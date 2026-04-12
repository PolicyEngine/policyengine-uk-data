"""ONS National Balance Sheet land value targets."""

from policyengine_uk_data.targets.schema import Target, Unit
from policyengine_uk_data.targets.sources._land import (
    CORPORATE_LAND_VALUES,
    HOUSEHOLD_LAND_VALUES,
    TOTAL_LAND_VALUES,
    _REF_URL,
)


def get_targets() -> list[Target]:
    return [
        Target(
            name="ons/household_land_value",
            variable="household_land_value",
            source="ons",
            unit=Unit.GBP,
            values=HOUSEHOLD_LAND_VALUES,
            reference_url=_REF_URL,
        ),
        Target(
            name="ons/corporate_land_value",
            variable="corporate_land_value",
            source="ons",
            unit=Unit.GBP,
            values=CORPORATE_LAND_VALUES,
            reference_url=_REF_URL,
        ),
        Target(
            name="ons/land_value",
            variable="land_value",
            source="ons",
            unit=Unit.GBP,
            values=TOTAL_LAND_VALUES,
            reference_url=_REF_URL,
        ),
    ]
