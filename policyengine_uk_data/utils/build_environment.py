import sys
from typing import Iterable


def get_local_build_issues(
    python_version: tuple[int, ...] | None = None,
    variable_names: Iterable[str] | None = None,
    corporate_sdlt: dict[int, float] | None = None,
) -> list[str]:
    """Problems with the build environment.

    ``corporate_sdlt`` is the resolved policyengine-uk's corporate SDLT
    statistic by year (``obr.corporate_sdlt_receipts``). It is checked only
    when given, which full builds do.
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

    if corporate_sdlt is not None:
        from policyengine_uk_data.targets.sources.obr import (
            corporate_sdlt_is_projected,
        )

        if not corporate_sdlt_is_projected(corporate_sdlt):
            issues.append(
                "The resolved policyengine-uk holds corporate SDLT at "
                f"£{next(iter(corporate_sdlt.values())) / 1e9:.2f}bn in every "
                f"year from {min(corporate_sdlt)} to {max(corporate_sdlt)}. "
                "Full builds need a version with "
                "PolicyEngine/policyengine-uk#2322, which projects it with the "
                "OBR forecast: households' SDLT is calibrated to the OBR total "
                "less this statistic, so a stale one overstates it."
            )

    return issues


def assert_local_build_environment(full_build: bool = False) -> None:
    corporate_sdlt = None
    if full_build:
        from policyengine_uk_data.targets.sources.obr import corporate_sdlt_receipts

        corporate_sdlt = corporate_sdlt_receipts()
    issues = get_local_build_issues(corporate_sdlt=corporate_sdlt)
    if not issues:
        return

    raise RuntimeError(
        "Local UK data build environment is not compatible:\n- " + "\n- ".join(issues)
    )
