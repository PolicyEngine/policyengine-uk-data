"""Shared utilities for target source modules."""

from pathlib import Path

import yaml

SOURCES_YAML = Path(__file__).parent.parent / "sources.yaml"
STORAGE = Path(__file__).parents[2] / "storage"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)" " AppleWebKit/537.36"
    ),
}


def load_config() -> dict:
    with open(SOURCES_YAML) as f:
        return yaml.safe_load(f)


def to_float(val) -> float:
    """Convert a cell value to float, handling suppressed markers."""
    if isinstance(val, (int, float)):
        return float(val)
    try:
        return float(val)
    except (ValueError, TypeError):
        return 0.0
