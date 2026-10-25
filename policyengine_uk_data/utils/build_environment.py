import sys
from typing import Iterable


def get_local_build_issues(
    python_version: tuple[int, ...] | None = None,
    variable_names: Iterable[str] | None = None,
    unprojected_corporate_sdlt: Iterable[str] | None = None,
) -> list[str]:
    """Problems with the build environment.

    ``unprojected_corporate_sdlt`` lists the corporate SDLT statistics the
    resolved policyengine-uk does not project
    (``obr.unprojected_corporate_sdlt``). It is checked only when given,
    which full builds do.
    """
    issues: list[str] = []

    if python_version is None:
        python_version = sys.version_info[:3]

    if tuple(python_version)[:2] != (3, 13):
        issues.append(
            "Use Python 3.13 for local dataset builds. Python 3.14 currently "
            "fails while loading PyTables/Blosc2 in this repo."
        )

    if variable_names is None:
        from policyengine_uk import CountryTaxBenefitSystem

        variable_names = CountryTaxBenefitSystem().variables

    if "num_vehicles" not in set(variable_names):
        issues.append(
            "The resolved policyengine-uk package does not expose the "
            "`num_vehicles` variable required by the UK data pipeline. For "
            "local builds, prefer the sibling `policyengine-uk` checkout on "
            "PYTHONPATH."
        )

    unprojected = list(unprojected_corporate_sdlt or [])
    if unprojected:
        issues.append(
            "The resolved policyengine-uk does not project "
            f"{' or '.join(unprojected)}: it carries the last outturn "
            "forward. Full builds need policyengine-uk 2.152.0 or later "
            "(PolicyEngine/policyengine-uk#2322), which uprates corporate SDLT "
            "with the OBR forecast: households' SDLT is calibrated to the OBR "
            "total less this statistic, so a stale one overstates it."
        )

    return issues


def assert_local_build_environment(full_build: bool = False) -> None:
    unprojected = None
    if full_build:
        from policyengine_uk_data.targets.sources.obr import (
            unprojected_corporate_sdlt,
        )

        unprojected = unprojected_corporate_sdlt()
    issues = get_local_build_issues(unprojected_corporate_sdlt=unprojected)
    if not issues:
        return

    raise RuntimeError(
        "Local UK data build environment is not compatible:\n- " + "\n- ".join(issues)
    )
