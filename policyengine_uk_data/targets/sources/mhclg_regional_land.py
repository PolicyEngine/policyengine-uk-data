"""Regional household land value targets."""

import pandas as pd

from policyengine_uk_data.targets.schema import (
    GeographicLevel,
    Target,
    Unit,
)
from policyengine_uk_data.targets.sources._land import (
    HOUSEHOLD_LAND_VALUES,
    _REF_URL,
)
from policyengine_uk_data.targets.sources._common import STORAGE


def _compute_regional_shares() -> dict[str, float]:
    """Split household land totals across regions using fixed 2025 shares.

    Each region's share is proportional to its total property wealth
    (dwellings × avg_house_price). The shares are then scaled so the
    GB total sums to 1.
    """
    csv_path = STORAGE / "regional_land_values.csv"
    df = pd.read_csv(csv_path)

    df["property_wealth"] = df["dwellings"] * df["avg_house_price"]
    total = df["property_wealth"].sum()

    return dict(zip(df["region"], df["property_wealth"] / total))


def _compute_regional_targets() -> dict[str, dict[int, float]]:
    """Scale fixed regional shares by the national household-land series."""
    shares = _compute_regional_shares()
    return {
        region: {
            year: share * HOUSEHOLD_LAND_VALUES[year] for year in HOUSEHOLD_LAND_VALUES
        }
        for region, share in shares.items()
    }


def get_targets() -> list[Target]:
    csv_path = STORAGE / "regional_land_values.csv"
    if not csv_path.exists():
        return []

    regional = _compute_regional_targets()
    targets = []

    for region, value in regional.items():
        targets.append(
            Target(
                name=f"ons/household_land_value/{region}",
                variable="household_land_value",
                source="ons",
                unit=Unit.GBP,
                geographic_level=GeographicLevel.REGION,
                geo_name=region,
                values=value,
                reference_url=_REF_URL,
            )
        )

    return targets
