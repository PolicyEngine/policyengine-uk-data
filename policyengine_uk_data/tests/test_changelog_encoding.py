"""Test that changelog files are valid UTF-8 and can be parsed by yaml-changelog."""

import pytest
from pathlib import Path
import yaml


def get_repo_root() -> Path:
    """Get the repository root directory."""
    current = Path(__file__).resolve()
    while current != current.parent:
        if (current / "pyproject.toml").exists():
            return current
        current = current.parent
    raise RuntimeError("Could not find repository root")


class TestChangelogEncoding:
    """Tests for changelog file encoding validation."""

    def test_changelog_entry_is_valid_utf8(self):
        """Test that changelog_entry.yaml is valid UTF-8.

        This prevents CI failures from the yaml-changelog tool which
        cannot parse files with non-UTF-8 characters (e.g., Latin-1
        encoded £ symbols).
        """
        repo_root = get_repo_root()
        changelog_entry_path = repo_root / "changelog_entry.yaml"

        if not changelog_entry_path.exists():
            pytest.skip("No changelog_entry.yaml file present")

        # Read as bytes and try to decode as UTF-8
        content_bytes = changelog_entry_path.read_bytes()

        # Skip if file is empty (happens after versioning workflow consumes it)
        if not content_bytes.strip():
            pytest.skip("changelog_entry.yaml is empty")

        try:
            content_bytes.decode("utf-8")
        except UnicodeDecodeError as e:
            # Find the problematic byte for better error message
            bad_byte = content_bytes[e.start : e.start + 1]
            pytest.fail(
                f"changelog_entry.yaml contains non-UTF-8 bytes at position "
                f"{e.start}: {bad_byte.hex()} (0x{bad_byte.hex()}). "
                f"This is likely a Latin-1 encoded character. "
                f"Please re-save the file with UTF-8 encoding."
            )

    def test_changelog_entry_is_valid_yaml(self):
        """Test that changelog_entry.yaml is valid YAML."""
        repo_root = get_repo_root()
        changelog_entry_path = repo_root / "changelog_entry.yaml"

        if not changelog_entry_path.exists():
            pytest.skip("No changelog_entry.yaml file present")

        content = changelog_entry_path.read_text(encoding="utf-8")

        # Skip if file is empty (happens after versioning workflow consumes it)
        if not content.strip():
            pytest.skip("changelog_entry.yaml is empty")

        try:
            data = yaml.safe_load(content)
        except yaml.YAMLError as e:
            pytest.fail(f"changelog_entry.yaml is not valid YAML: {e}")

        # Skip if data is None (empty YAML)
        if data is None:
            pytest.skip("changelog_entry.yaml is empty")

        # Verify structure
        assert isinstance(data, list), "changelog_entry.yaml must be a list"
        for entry in data:
            assert "bump" in entry, "Each entry must have a 'bump' field"
            assert entry["bump"] in (
                "patch",
                "minor",
                "major",
            ), f"Invalid bump type: {entry['bump']}"
            assert "changes" in entry, "Each entry must have a 'changes' field"

    def test_changelog_yaml_is_valid_utf8(self):
        """Test that changelog.yaml is valid UTF-8."""
        repo_root = get_repo_root()
        changelog_path = repo_root / "changelog.yaml"

        if not changelog_path.exists():
            pytest.skip("No changelog.yaml file present")

        content_bytes = changelog_path.read_bytes()

        try:
            content_bytes.decode("utf-8")
        except UnicodeDecodeError as e:
            bad_byte = content_bytes[e.start : e.start + 1]
            pytest.fail(
                f"changelog.yaml contains non-UTF-8 bytes at position "
                f"{e.start}: {bad_byte.hex()} (0x{bad_byte.hex()}). "
                f"Please re-save the file with UTF-8 encoding."
            )
