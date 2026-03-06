"""Target registry: discovers source modules and collects targets."""

import importlib
import pkgutil
from pathlib import Path

import yaml

from policyengine_uk_data.targets.schema import (
    GeographicLevel,
    Target,
)
from policyengine_uk_data.targets import sources as sources_pkg


def load_sources_config() -> dict:
    """Load the sources.yaml URL configuration."""
    config_path = Path(__file__).parent / "sources.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def discover_source_modules() -> list:
    """Import all modules under targets.sources."""
    modules = []
    package_path = Path(sources_pkg.__file__).parent
    for importer, modname, ispkg in pkgutil.iter_modules([str(package_path)]):
        mod = importlib.import_module(f"policyengine_uk_data.targets.sources.{modname}")
        if hasattr(mod, "get_targets"):
            modules.append(mod)
    return modules


def get_all_targets(
    year: int | None = None,
    geographic_level: GeographicLevel | None = None,
) -> list[Target]:
    """Collect targets from all source modules.

    Args:
        year: if provided, only return targets that have a value for
              this year.
        geographic_level: if provided, filter to this geographic level.

    Returns:
        De-duplicated list of Target objects.
    """
    all_targets: list[Target] = []
    seen_names: set[str] = set()

    for mod in discover_source_modules():
        for target in mod.get_targets():
            if target.name in seen_names:
                continue
            if year is not None and year not in target.values:
                continue
            if (
                geographic_level is not None
                and target.geographic_level != geographic_level
            ):
                continue
            seen_names.add(target.name)
            all_targets.append(target)

    return all_targets
