def fetch_version():
    """Fetch version from pyproject.toml."""
    import re
    from pathlib import Path

    pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
    content = pyproject_path.read_text()
    match = re.search(r'^version\s*=\s*"([^"]+)"', content, re.MULTILINE)
    if match:
        return match.group(1)
    return None


if __name__ == "__main__":
    print(fetch_version())
