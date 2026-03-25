"""Regional household land value targets.

Splits the ONS National Balance Sheet household land total across
regions in proportion to total property wealth (dwellings × avg
house price from UK HPI Dec 2025).

The model's regional intensity ratios (in policyengine-uk) handle the
conversion from property wealth to land value per household. These
targets ensure the weighted regional totals match official estimates.

Sources:
  - UK House Price Index Dec 2025
    https://www.gov.uk/government/statistics/uk-house-price-index-for-december-2025
  - ONS National Balance Sheet 2025
    https://www.ons.gov.uk/economy/nationalaccounts/uksectoraccounts/bulletins/nationalbalancesheet/2025

See: https://github.com/PolicyEngine/policyengine-uk-data/issues/314
"""

import pandas as pd

from policyengine_uk_data.targets.schema import (
    GeographicLevel,
    Target,
    Unit,
)
from policyengine_uk_data.targets.sources._common import STORAGE

# ONS National Balance Sheet 2025 — household land value
_ONS_2020_HOUSEHOLD = 4.31e12
_ONS_2020_CORPORATE = 1.76e12
_ONS_2020_TOTAL = _ONS_2020_HOUSEHOLD + _ONS_2020_CORPORATE
_ONS_2024_TOTAL = 7.1e12
_SCALE = _ONS_2024_TOTAL / _ONS_2020_TOTAL
_ONS_2024_HOUSEHOLD = _ONS_2020_HOUSEHOLD * _SCALE

_ONS_REF = (
    "https://www.ons.gov.uk/economy/nationalaccounts/"
    "uksectoraccounts/bulletins/nationalbalancesheet/2025"
)


def _compute_regional_targets() -> dict[str, float]:
    """Split the ONS household land total across regions.

    Each region's share is proportional to its total property wealth
    (dwellings × avg_house_price). The shares are then scaled so the
    GB total matches the ONS national household land value.
    """
    csv_path = STORAGE / "regional_land_values.csv"
    df = pd.read_csv(csv_path)

    df["property_wealth"] = df["dwellings"] * df["avg_house_price"]
    total = df["property_wealth"].sum()

    return dict(zip(df["region"], df["property_wealth"] / total * _ONS_2024_HOUSEHOLD))


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
                values={
                    2024: value,
                    2025: value,
                    2026: value,
                },
                reference_url=_ONS_REF,
            )
        )

    return targets
