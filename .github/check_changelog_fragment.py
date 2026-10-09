"""Check the towncrier changelog fragments a pull request would merge.

Towncrier compiles only fragments whose names carry a type from
``[tool.towncrier]`` in pyproject.toml, and leaves any other file in
changelog.d, so that entry never reaches CHANGELOG.md. This check accepts one
name form, ``changelog.d/<name>.<type>.md`` with an ASCII ``<name>``, which
towncrier and .github/bump_version.py both read as ``<type>``. It also rejects
two fragments that towncrier would file as the same entry, because towncrier
then stops versioning with an error.

A pull request that changes a file outside the exempt paths must add a typed
fragment of its own. The exempt paths are docs, CI and release tooling
(.github/), agent settings, offline tools, changelog fragments, root Markdown
files and .gitignore. That is a release-policy choice, not a guarantee: a
change under .github/ can still change how the next release is built.
Exempting these paths lets docs and CI changes merge without a data release.

Usage: python .github/check_changelog_fragment.py <base> <head>
"""

import os
import re
import subprocess
import sys
import tomllib
from collections.abc import Callable, Iterable
from pathlib import Path

FRAGMENT_DIR = "changelog.d/"
GITKEEP = "changelog.d/.gitkeep"
EXEMPT_DIRS = (".claude/", ".github/", "changelog.d/", "docs/", "tools/")
EXEMPT_ROOT_FILES = frozenset({".gitignore"})


def fragment_types(pyproject: Path) -> list[str]:
    config = tomllib.loads(pyproject.read_text())["tool"]["towncrier"]
    return [fragment_type["directory"] for fragment_type in config["type"]]


def fragment_pattern(types: Iterable[str]) -> re.Pattern:
    alternatives = "|".join(re.escape(fragment_type) for fragment_type in types)
    return re.compile(
        rf"changelog\.d/(?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)\.(?P<type>{alternatives})\.md"
    )


def towncrier_entry(match: re.Match) -> tuple[str, str]:
    """The entry towncrier files a fragment under: it drops a number's leading zeros."""
    name = match["name"]
    return (str(int(name)) if name.isdigit() else name, match["type"])


def needs_fragment(path: str) -> bool:
    if path.startswith(EXEMPT_DIRS) or path in EXEMPT_ROOT_FILES:
        return False
    return "/" in path or not path.endswith(".md")


def check(
    changes: Iterable[tuple[str, str]],
    types: list[str],
    read_bytes: Callable[[str], bytes],
    fragments_at_head: Iterable[str],
) -> list[str]:
    """Return one message per problem.

    ``changes`` holds ``(status, path)`` pairs as printed by
    ``git diff --name-status --no-renames``, so a rename is a deletion plus an
    addition. ``read_bytes`` returns a path's content at the pull request's
    head, and ``fragments_at_head`` lists every file in changelog.d there.
    """
    changes = list(changes)
    pattern = fragment_pattern(types)
    problems = []
    added_fragments = []
    for status, path in changes:
        if not path.startswith(FRAGMENT_DIR) or path == GITKEEP or status == "D":
            continue
        if not pattern.fullmatch(path):
            problems.append(
                f"{path}: name it changelog.d/<name>.<type>.md, where <name> uses "
                "only ASCII letters, digits, '.', '_' and '-', and <type> is one "
                f"of: {', '.join(types)}. Towncrier ignores a fragment without a "
                "type, so its entry would never reach CHANGELOG.md, and this is "
                "the form that towncrier and .github/bump_version.py both read "
                "the same way."
            )
            continue
        try:
            text = read_bytes(path).decode("utf-8")
        except UnicodeDecodeError:
            problems.append(f"{path} is not UTF-8 text.")
            continue
        if not text.strip():
            problems.append(f"{path} is empty. Describe the change in it.")
        elif status == "A":
            added_fragments.append(path)
    entries = {}
    for path in sorted(fragments_at_head):
        if match := pattern.fullmatch(path):
            entries.setdefault(towncrier_entry(match), []).append(path)
    for (name, fragment_type), paths in sorted(entries.items()):
        if len(paths) > 1:
            problems.append(
                f"{' and '.join(paths)}: towncrier files each as "
                f"{name}.{fragment_type} and stops versioning with an error. "
                "Rename all but one."
            )
    needing = sorted({path for _, path in changes if needs_fragment(path)})
    if needing and not added_fragments:
        listed = ", ".join(needing[:5]) + (", ..." if len(needing) > 5 else "")
        problems.append(
            f"This pull request changes files outside the exempt paths ({listed}) "
            "but adds no changelog fragment. Add one with: echo 'Description.' > "
            "changelog.d/<branch-name>.<type>.md, where <type> is one of: "
            f"{', '.join(types)}."
        )
    return problems


def changed_paths(base: str, head: str) -> list[tuple[str, str]]:
    output = subprocess.run(
        ["git", "diff", "--name-status", "--no-renames", "-z", base, head],
        check=True,
        capture_output=True,
    ).stdout
    fields = output.split(b"\0")[:-1]
    return [
        (status.decode(), os.fsdecode(path))
        for status, path in zip(fields[::2], fields[1::2])
    ]


def main(argv: list[str]) -> int:
    base, head = argv
    changes = changed_paths(base, head)
    types = fragment_types(Path("pyproject.toml"))
    fragments_at_head = [
        f"{FRAGMENT_DIR}{path.name}"
        for path in Path(FRAGMENT_DIR).iterdir()
        if path.is_file()
    ]
    problems = check(
        changes, types, lambda path: Path(path).read_bytes(), fragments_at_head
    )
    for problem in problems:
        print(f"::error::{problem}")
    if problems:
        return 1
    if any(needs_fragment(path) for _, path in changes):
        print("Changelog fragment check passed.")
    else:
        print(
            "::notice::No fragment needed: this pull request changes only exempt paths."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
