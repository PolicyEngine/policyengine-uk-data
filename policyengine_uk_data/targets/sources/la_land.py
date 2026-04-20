"""LA-level household land value targets.

Local authority generalisation of mhclg_regional_land.py. Each local
authority's share of national household land value is proportional to
its total property wealth (households x avg_house_price), then scaled
to the ONS National Balance Sheet household-land series.

Data sources:
- Average house price by LA: HM Land Registry UK HPI (Dec 2025).
  For LAs whose ONS code changed between releases, the CSV matches on
  LA name. For Northern Ireland LGDs missing from a specific month,
  the NI country-level HPI price is used as a fallback.
- Households by LA: derived from the policyengine-uk-data LA weight
  matrix (storage/local_authority_weights.h5), keeping the household
  count definition consistent with the rest of the LA calibration.
- National household land total: HOUSEHOLD_LAND_VALUES (ONS National
  Balance Sheet 2025, series AN.211 household sector).
"""

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


def _load_inputs() -> pd.DataFrame:
    csv_path = STORAGE / "la_land_values.csv"
    return pd.read_csv(csv_path)


def _compute_la_shares() -> pd.DataFrame:
    """Return a DataFrame with columns code, name, share.

    Each LA's share is proportional to households x avg_house_price,
    scaled to sum to 1 across all UK local authorities.
    """
    df = _load_inputs()
    df["property_wealth"] = df["households"] * df["avg_house_price"]
    total = df["property_wealth"].sum()
    df["share"] = df["property_wealth"] / total
    return df[["code", "name", "share"]]


def _compute_la_targets() -> dict[str, dict[int, float]]:
    """Scale per-LA shares by the national household-land series."""
    shares = _compute_la_shares().set_index("code")["share"]
    return {
        code: {
            year: float(share) * HOUSEHOLD_LAND_VALUES[year]
            for year in HOUSEHOLD_LAND_VALUES
        }
        for code, share in shares.items()
    }


def get_targets() -> list[Target]:
    csv_path = STORAGE / "la_land_values.csv"
    if not csv_path.exists():
        return []

    df = _load_inputs()
    la_targets = _compute_la_targets()

    targets: list[Target] = []
    for _, row in df.iterrows():
        code = row["code"]
        name = row["name"]
        targets.append(
            Target(
                name=f"ons/household_land_value/{code}",
                variable="household_land_value",
                source="ons",
                unit=Unit.GBP,
                geographic_level=GeographicLevel.LOCAL_AUTHORITY,
                geo_code=code,
                geo_name=name,
                values=la_targets[code],
                reference_url=_REF_URL,
            )
        )

    return targets
