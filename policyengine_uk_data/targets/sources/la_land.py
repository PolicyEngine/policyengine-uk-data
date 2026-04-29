"""LA-level main residence value targets (derived proxy).

This target is a **derived proxy**, not a directly observed LA total.
Per-LA target is constructed by multiplying three observed inputs:

    target_la = avg_house_price_la × ownership_share_la × n_households_la

Same multiplicative shape as the existing private-rent target
(median × pct × n_households).

Lineage caveat (flagged in PR review by @MaxGhenis):
- Matrix column ``main_residence_value`` in policyengine-uk is a
  **stock-wealth** quantity, imputed from the Wealth and Assets Survey
  (WAS) and uprated regionally via property-wealth intensity ratios.
- The target value uses HM Land Registry UK HPI "Average Price" — a
  **transaction-weighted geography-period price index**, not an
  observed stock total of owner-occupied main residences.
- The product avg_price × ownership × n_households is therefore a
  defensible identity ("if every owner-occupied dwelling were valued
  at the LA HPI average, the total would be £X") but the two sides
  of the calibration constraint reference different price concepts.

The target is treated as a soft-weighted/proxy training signal
relative to direct observed targets (HMRC SPI counts, ONS mid-year
population, DWP UC caseload, VOA dwelling band counts).

Data sources:
- Average house price by LA: HM Land Registry UK HPI (Dec 2025).
  For LAs whose ONS code changed between releases, the CSV matches on
  LA name. For Northern Ireland LGDs missing from a specific month,
  the NI country-level HPI price is used as a fallback.
- Ownership share by LA: English Housing Survey, via load_tenure_data
  (owned_outright_pct + owned_mortgage_pct). England-only.
- Households by LA: Census 2021, via load_household_counts.
"""

import pandas as pd

from policyengine_uk_data.targets.schema import (
    GeographicLevel,
    Target,
    Unit,
)
from policyengine_uk_data.targets.sources._common import STORAGE


_REF_URL_HMLR = (
    "https://www.gov.uk/government/statistical-data-sets/"
    "uk-house-price-index-data-downloads-december-2025"
)


def load_la_avg_prices() -> pd.DataFrame:
    """Load HMLR average house price by LA.

    Returns DataFrame with columns: code, name, avg_house_price.
    """
    csv_path = STORAGE / "la_land_values.csv"
    if not csv_path.exists():
        return pd.DataFrame(columns=["code", "name", "avg_house_price"])
    df = pd.read_csv(csv_path)
    return df[["code", "name", "avg_house_price"]]


def _compute_la_targets() -> dict[str, float]:
    """Per-LA main residence value target.

    target_la = avg_house_price_la × ownership_share_la × n_households_la

    Returns a dict ``code -> £``. LAs missing any input drop out and
    are handled in loss.py by the national-share fallback (same
    pattern as the tenure and rent targets).
    """
    from policyengine_uk_data.targets.sources.local_la_extras import (
        load_household_counts,
        load_tenure_data,
    )

    prices = load_la_avg_prices()
    tenure = load_tenure_data()
    households = load_household_counts()

    if prices.empty or tenure.empty or households.empty:
        return {}

    merged = prices.merge(tenure, left_on="code", right_on="la_code", how="left").merge(
        households, on="la_code", how="left"
    )

    ownership_share = (
        merged["owned_outright_pct"].fillna(0) + merged["owned_mortgage_pct"].fillna(0)
    ) / 100
    targets = merged["avg_house_price"] * ownership_share * merged["households"]

    return {
        code: float(value)
        for code, value in zip(merged["code"], targets)
        if pd.notna(value) and value > 0
    }


def get_targets() -> list[Target]:
    prices = load_la_avg_prices()
    if prices.empty:
        return []

    la_targets = _compute_la_targets()

    targets: list[Target] = []
    for _, row in prices.iterrows():
        code = row["code"]
        target_value = la_targets.get(code)
        if target_value is None:
            continue
        # HMLR Dec 2025 snapshot; same value across calibration years
        # until a year-varying HMLR series is wired in.
        values = {year: target_value for year in (2024, 2025, 2026)}
        targets.append(
            Target(
                name=f"housing/main_residence_value/{code}",
                variable="main_residence_value",
                source="hmlr",
                unit=Unit.GBP,
                geographic_level=GeographicLevel.LOCAL_AUTHORITY,
                geo_code=code,
                geo_name=row["name"],
                values=values,
                reference_url=_REF_URL_HMLR,
            )
        )

    return targets
