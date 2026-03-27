"""Infer semver bump from towncrier fragment types and update version."""

import re
import sys
from pathlib import Path


def get_current_version(pyproject_path: Path) -> str:
    text = pyproject_path.read_text()
    match = re.search(r'^version\s*=\s*"(\d+\.\d+\.\d+)"', text, re.MULTILINE)
    if not match:
        print(
            "Could not find version in pyproject.toml",
            file=sys.stderr,
        )
        sys.exit(1)
    return match.group(1)


def infer_bump(changelog_dir: Path) -> str:
    fragments = [
        f
        for f in changelog_dir.rglob("*")
        if f.is_file() and f.name != ".gitkeep"
    ]
    if not fragments:
        print("No changelog fragments found", file=sys.stderr)
        sys.exit(1)

    categories = set()
    for f in fragments:
        # The fragment's parent directory name is the category (e.g. fixed, added)
        categories.add(f.parent.name)

    if "breaking" in categories:
        return "major"
    if "added" in categories or "removed" in categories:
        return "minor"
    return "patch"


def bump_version(version: str, bump: str) -> str:
    major, minor, patch = (int(x) for x in version.split("."))
    if bump == "major":
        return f"{major + 1}.0.0"
    elif bump == "minor":
        return f"{major}.{minor + 1}.0"
    else:
        return f"{major}.{minor}.{patch + 1}"


def update_file(path: Path, old_version: str, new_version: str):
    text = path.read_text()
    updated = text.replace(
        f'version = "{old_version}"',
        f'version = "{new_version}"',
    )
    if updated != text:
        path.write_text(updated)
        print(f"  Updated {path}")


def main():
    root = Path(__file__).resolve().parent.parent
    pyproject = root / "pyproject.toml"
    changelog_dir = root / "changelog.d"

    current = get_current_version(pyproject)
    bump = infer_bump(changelog_dir)
    new = bump_version(current, bump)

    print(f"Version: {current} -> {new} ({bump})")

    update_file(pyproject, current, new)


if __name__ == "__main__":
    main()
