"""Check that a pull request adds a typed towncrier changelog fragment.

Towncrier compiles only fragments whose names carry a type from
``[tool.towncrier]`` in pyproject.toml, and it leaves any other file in
changelog.d untouched, so that entry never reaches CHANGELOG.md. This check
accepts one name form, ``changelog.d/<name>.<type>.md``, which towncrier and
.github/bump_version.py both read as ``<type>``.

A pull request that changes a file a release ships must add a typed fragment
of its own. One that changes only files no release ships (docs, CI, agent
settings, offline tools, changelog fragments, root Markdown files) needs none.

Usage: python .github/check_changelog_fragment.py <base> <head>
"""

import re
import subprocess
import sys
import tomllib
from collections.abc import Callable, Iterable
from pathlib import Path

FRAGMENT_DIR = "changelog.d/"
GITKEEP = "changelog.d/.gitkeep"
UNSHIPPED_DIRS = (".claude/", ".github/", "changelog.d/", "docs/", "tools/")
UNSHIPPED_ROOT_FILES = frozenset({".gitignore"})


def fragment_types(pyproject: Path) -> list[str]:
    config = tomllib.loads(pyproject.read_text())["tool"]["towncrier"]
    return [fragment_type["directory"] for fragment_type in config["type"]]


def fragment_pattern(types: Iterable[str]) -> re.Pattern:
    alternatives = "|".join(re.escape(fragment_type) for fragment_type in types)
    return re.compile(rf"changelog\.d/[^/]+\.(?:{alternatives})\.md")


def ships(path: str) -> bool:
    """Whether a release ships ``path``, so changing it needs a fragment."""
    if path.startswith(UNSHIPPED_DIRS) or path in UNSHIPPED_ROOT_FILES:
        return False
    return "/" in path or not path.endswith(".md")


def check(
    changes: Iterable[tuple[str, str]],
    types: list[str],
    read_text: Callable[[str], str],
) -> list[str]:
    """Return one message per problem in ``changes``.

    ``changes`` holds ``(status, path)`` pairs as printed by
    ``git diff --name-status --no-renames``, so a rename is a deletion plus an
    addition. ``read_text`` returns a path's content at the pull request's head.
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
                f"{path}: name it changelog.d/<name>.<type>.md, where <type> is "
                f"one of: {', '.join(types)}. Towncrier ignores a fragment without "
                "a type, so its entry would never reach CHANGELOG.md, and this is "
                "the one form that towncrier and .github/bump_version.py both read "
                "the same way."
            )
        elif not read_text(path).strip():
            problems.append(f"{path} is empty. Describe the change in it.")
        elif status == "A":
            added_fragments.append(path)
    shipped = sorted({path for _, path in changes if ships(path)})
    if shipped and not added_fragments:
        listed = ", ".join(shipped[:5]) + (", ..." if len(shipped) > 5 else "")
        problems.append(
            f"This pull request changes files that a release ships ({listed}) "
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
        text=True,
    ).stdout
    fields = output.split("\0")[:-1]
    return list(zip(fields[::2], fields[1::2]))


def main(argv: list[str]) -> int:
    base, head = argv
    changes = changed_paths(base, head)
    types = fragment_types(Path("pyproject.toml"))
    problems = check(changes, types, lambda path: Path(path).read_text())
    for problem in problems:
        print(f"::error::{problem}")
    if problems:
        return 1
    fragments = [
        path
        for status, path in changes
        if status == "A" and fragment_pattern(types).fullmatch(path)
    ]
    if fragments:
        print(f"Changelog fragments added: {', '.join(fragments)}")
    else:
        print(
            "::notice::No fragment needed: this pull request changes only files "
            "that no release ships."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
