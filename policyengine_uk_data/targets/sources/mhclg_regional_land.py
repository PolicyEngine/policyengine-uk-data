"""Regional household land value targets.

Derived from MHCLG residential land value estimates (2023) and
UK House Price Index (Dec 2025), constrained to sum to the ONS
National Balance Sheet 2025 household land total.

The national flat intensity ratio (0.673) produces implausibly
uniform regional land values. These targets capture the true
regional variation so calibration can correct household weights.

Sources:
  - MHCLG Land Value Estimates for Policy Appraisal 2023
    https://www.gov.uk/government/publications/land-value-estimates-for-policy-appraisal-2023
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

_MHCLG_REF = (
    "https://www.gov.uk/government/publications/"
    "land-value-estimates-for-policy-appraisal-2023"
)


def _compute_regional_targets() -> dict[str, float]:
    """Derive regional household land value targets.

    For each region, estimate total residential land value as:
        dwellings × avg_house_price × land_intensity

    Then rescale so the GB total matches the ONS national household
    land value (£5.04tn for 2024). Northern Ireland is excluded as
    it is not in the FRS sample frame.
    """
    csv_path = STORAGE / "regional_land_values.csv"
    df = pd.read_csv(csv_path)

    df["raw_land"] = df["dwellings"] * df["avg_house_price"] * df["land_intensity"]
    raw_total = df["raw_land"].sum()

    # Rescale to match ONS national household land total
    scale = _ONS_2024_HOUSEHOLD / raw_total
    return dict(zip(df["region"], df["raw_land"] * scale))


def get_targets() -> list[Target]:
    csv_path = STORAGE / "regional_land_values.csv"
    if not csv_path.exists():
        return []

    regional = _compute_regional_targets()
    targets = []

    for region, value in regional.items():
        targets.append(
            Target(
                name=f"mhclg/household_land_value/{region}",
                variable="household_land_value",
                source="mhclg",
                unit=Unit.GBP,
                geographic_level=GeographicLevel.REGION,
                geo_name=region,
                values={
                    2024: value,
                    2025: value,
                    2026: value,
                },
                reference_url=_MHCLG_REF,
            )
        )

    return targets
