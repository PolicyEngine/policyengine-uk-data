"""Targets system: structured, source-traceable calibration targets."""

from policyengine_uk_data.targets.registry import get_all_targets
from policyengine_uk_data.targets.schema import (
    GeographicLevel,
    Target,
    Unit,
)

__all__ = [
    "get_all_targets",
    "GeographicLevel",
    "Target",
    "Unit",
]
