"""Local-authority council tax calibration targets.

Produces two kinds of LA-level calibration target from public data:

- ``ons/council_tax_band_d/{code}``: the average Band D council tax
  (inclusive of all precepts) each household pays in billing authority
  ``code``. Sourced from MHCLG, Welsh Government and Scottish
  Government annual publications.
- ``voa/council_tax/{code}/{band}``: the number of dwellings in band
  ``A``–``H`` (England) or ``A``–``I`` (Wales) for billing authority
  ``code``. Sourced from the VOA *Council Tax: Stock of Properties*
  summary tables.

Data for all 360 LAs in ``local_authorities_2021.csv`` is joined from
the committed canonical file ``storage/la_council_tax.csv``. Rows where
a source did not provide a value are omitted so calibrators cleanly
skip them.

Known coverage gaps (documented, not bugs):

- Northern Ireland is excluded because its domestic rates system is
  distinct from council tax.
- Band-count rows for Scottish LAs are absent because the VOA summary
  tables do not cover Scotland; Scottish Assessors publishes per-LA
  chargeable-dwellings data separately and is a follow-up.
- Band I only exists in Wales (introduced in the 2005 Welsh revaluation);
  English rows leave it null.
- City of London has Band A suppressed by VOA for disclosure control;
  its other bands are populated.

Sources:
- MHCLG *Council Tax levels set by local authorities in England 2026-27*
  https://www.gov.uk/government/statistics/council-tax-levels-set-by-local-authorities-in-england-2026-to-2027
- Welsh Government *Council Tax levels: April 2026 to March 2027*
  https://www.gov.wales/council-tax-levels-april-2026-march-2027-html
- Scottish Government *Council Tax Assumptions 2025* (CT by Band, 2025-26)
  https://www.gov.scot/publications/council-tax-datasets/
- VOA *Council Tax: Stock of Properties, 2025*
  https://www.gov.uk/government/statistics/council-tax-stock-of-properties-2025
"""

from __future__ import annotations

from functools import lru_cache

import pandas as pd

from policyengine_uk_data.targets.schema import (
    GeographicLevel,
    Target,
    Unit,
)
from policyengine_uk_data.targets.sources._common import STORAGE


_CSV_NAME = "la_council_tax.csv"

# Latest fiscal years covered by each source. The LA Band D amounts are
# structurally single-year snapshots; callers that need longer time
# series should uprate via the existing council-tax uprating index.
_YEAR_BAND_D_ENGLAND = 2026
_YEAR_BAND_D_WALES = 2026
_YEAR_BAND_D_SCOTLAND = 2025
_YEAR_BAND_COUNT = 2025

_BAND_COUNT_COLUMNS = {band: f"count_band_{band}" for band in "ABCDEFGHI"}

_ENGLAND_REF = (
    "https://www.gov.uk/government/statistics/"
    "council-tax-levels-set-by-local-authorities-in-england-2026-to-2027"
)
_WALES_REF = "https://www.gov.wales/council-tax-levels-april-2026-march-2027-html"
_SCOTLAND_REF = "https://www.gov.scot/publications/council-tax-datasets/"
_VOA_REF = (
    "https://www.gov.uk/government/statistics/council-tax-stock-of-properties-2025"
)
# Net council tax requirement per LA. England derived from MHCLG
# Council Taxbase 2025 Table 1.35 ("Tax base after allowance for council
# tax support") × LA Band D amount. Wales sourced directly from the
# Welsh Government Table 3 "Council tax income (£m)" — already net.
_NET_CT_REF_ENG = (
    "https://www.gov.uk/government/statistics/council-taxbase-2025-in-england"
)
_NET_CT_REF_WAL = _WALES_REF


@lru_cache(maxsize=1)
def _load_table() -> pd.DataFrame | None:
    """Return the committed LA council-tax table, or ``None`` if missing."""
    csv_path = STORAGE / _CSV_NAME
    if not csv_path.exists():
        return None
    return pd.read_csv(csv_path)


def load_la_net_council_tax() -> pd.DataFrame:
    """Load per-LA net council tax requirement (£, after CTR support).

    Returns a DataFrame with columns ``code, total_council_tax_net``
    for LAs where a directly-observed net figure is available
    (England + Wales). Scotland and NI are absent and handled by the
    loss-matrix national-share fallback — same pattern as the rent
    and tenure targets.
    """
    df = _load_table()
    if df is None or df.empty:
        return pd.DataFrame(columns=["code", "total_council_tax_net"])
    if "total_council_tax_net" not in df.columns:
        return pd.DataFrame(columns=["code", "total_council_tax_net"])
    return df.loc[
        df["total_council_tax_net"].notna(),
        ["code", "total_council_tax_net"],
    ].reset_index(drop=True)


def _year_for_band_d(country: str) -> int:
    if country == "WALES":
        return _YEAR_BAND_D_WALES
    if country == "SCOTLAND":
        return _YEAR_BAND_D_SCOTLAND
    return _YEAR_BAND_D_ENGLAND


def _ref_for_band_d(country: str) -> str:
    if country == "WALES":
        return _WALES_REF
    if country == "SCOTLAND":
        return _SCOTLAND_REF
    return _ENGLAND_REF


def get_targets() -> list[Target]:
    """Emit LA-level Band D amount + band-count targets."""
    df = _load_table()
    if df is None or df.empty:
        return []

    targets: list[Target] = []

    # Band D amount targets — one per LA with a reported value.
    for _, row in df.iterrows():
        amount = row.get("band_d_amount")
        if pd.isna(amount):
            continue
        code = str(row["code"])
        country = str(row["country"])
        targets.append(
            Target(
                name=f"ons/council_tax_band_d/{code}",
                variable="council_tax_band_d_amount",
                source="ons",
                unit=Unit.GBP,
                geographic_level=GeographicLevel.LOCAL_AUTHORITY,
                geo_code=code,
                geo_name=str(row["name"]),
                values={_year_for_band_d(country): float(amount)},
                reference_url=_ref_for_band_d(country),
            )
        )

    # Band count targets — one per (LA, band) where VOA has a value.
    for _, row in df.iterrows():
        code = str(row["code"])
        name = str(row["name"])
        for band, col in _BAND_COUNT_COLUMNS.items():
            count = row.get(col)
            if pd.isna(count):
                continue
            targets.append(
                Target(
                    name=f"voa/council_tax/{code}/{band}",
                    variable="council_tax_band",
                    source="voa",
                    unit=Unit.COUNT,
                    geographic_level=GeographicLevel.LOCAL_AUTHORITY,
                    geo_code=code,
                    geo_name=name,
                    values={_YEAR_BAND_COUNT: float(count)},
                    is_count=True,
                    reference_url=_VOA_REF,
                )
            )

    # Net council tax £ targets — one per LA with an observed value.
    # Mirrors the FRS net-of-CTR amount; pairs with the band targets
    # above to cover both FRS council-tax data points.
    if "total_council_tax_net" in df.columns:
        for _, row in df.iterrows():
            net = row.get("total_council_tax_net")
            if pd.isna(net):
                continue
            country = str(row["country"])
            ref = _NET_CT_REF_WAL if country == "WALES" else _NET_CT_REF_ENG
            targets.append(
                Target(
                    name=f"housing/council_tax_net/{row['code']}",
                    variable="council_tax_less_benefit",
                    source="mhclg" if country == "ENGLAND" else "stats_wales",
                    unit=Unit.GBP,
                    geographic_level=GeographicLevel.LOCAL_AUTHORITY,
                    geo_code=str(row["code"]),
                    geo_name=str(row["name"]),
                    values={_YEAR_BAND_D_ENGLAND: float(net)},
                    reference_url=ref,
                )
            )

    return targets
