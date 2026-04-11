import sys
from typing import Iterable


def get_local_build_issues(
    python_version: tuple[int, ...] | None = None,
    variable_names: Iterable[str] | None = None,
) -> list[str]:
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

    return issues


def assert_local_build_environment() -> None:
    issues = get_local_build_issues()
    if not issues:
        return

    raise RuntimeError(
        "Local UK data build environment is not compatible:\n- "
        + "\n- ".join(issues)
    )
