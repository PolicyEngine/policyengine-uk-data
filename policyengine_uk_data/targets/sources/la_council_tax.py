"""Local-authority council tax calibration targets (derived proxies).

Produces two kinds of LA-level calibration target from public data:

- ``voa/council_tax/{code}/{band}``: the number of dwellings in band
  ``A``–``H`` (England) or ``A``–``I`` (Wales) for billing authority
  ``code``. Sourced from the VOA *Council Tax: Stock of Properties*
  summary tables.
- ``housing/council_tax_net/{code}``: net council tax requirement per
  LA (net of CTR support). England derived from MHCLG taxbase × Band D;
  Wales sourced directly from WG Council Tax Income (Table 3).

No target is emitted from the ``band_d_amount`` column. A Band D
amount is a **rate**, not an additive household control, and the loss
matrix is an additive objective — see the note in ``get_targets`` and
issue #483.

Data for all 360 LAs in ``local_authorities_2021.csv`` is joined from
the committed canonical file ``storage/la_council_tax.csv``. Rows where
a source did not provide a value are omitted so calibrators cleanly
skip them.

Lineage caveats (flagged in PR review by @MaxGhenis):

- ``voa/council_tax/{A..H}`` is a **derived proxy**, not a direct
  match for the matrix-side household ``council_tax_band``:
  * Target counts VOA dwellings; matrix counts policyengine-uk
    households. A household ≠ a dwelling in general.
  * VOA stock includes exempt, empty, and second-home dwellings,
    which contribute zero to the matrix-side sum (no household lives
    in them in the FRS).
  * VOA covers England and Wales only. Scotland and NI cells are
    masked out of the loss matrix unless a direct source is available.
  * Banding ratios differ: Scotland diverged from the standard
    6/9–18/9 E&W ratios after the 2017 reform; Wales has Band I,
    England does not.

- ``housing/council_tax_net`` is a **derived proxy**:
  * Target value (England) is MHCLG ``taxbase × Band D``, where
    taxbase is Band D equivalent dwellings adjusted for ~7
    discount, premium, and exemption classes (single-person,
    disabled relief, second-home, empty-home premium, family
    annexe, etc.). Wales uses WG-published net council tax income
    direct.
  * Matrix col is FRS-reported ``council_tax_less_benefit``
    (household-reported gross less reported CTB).
  * Same intent (what households pay net of CTR), different
    construction paths and underlying microdata sources.

Known coverage gaps:

- Northern Ireland is excluded because its domestic rates system is
  distinct from council tax. ``loss.py`` masks NI cells rather than
  fabricating a fallback.
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
- MHCLG *Council Taxbase 2025 in England* (Table 1.35 taxbase after CTR)
  https://www.gov.uk/government/statistics/council-taxbase-2025-in-england
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

# Latest fiscal years covered by each source. These are structurally
# single-year snapshots; callers that need longer time series should
# uprate via the existing council-tax uprating index.
# ``_YEAR_BAND_D_ENGLAND`` is retained because the England net
# council-tax requirement is derived from the 2026-27 Band D level.
_YEAR_BAND_D_ENGLAND = 2026
_YEAR_BAND_COUNT = 2025

_BAND_COUNT_COLUMNS = {band: f"count_band_{band}" for band in "ABCDEFGHI"}

_WALES_REF = "https://www.gov.wales/council-tax-levels-april-2026-march-2027-html"
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
    (England + Wales). Scotland and NI are absent; loss-matrix callers
    should mask those cells rather than fabricating fallback values.
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


def get_targets() -> list[Target]:
    """Emit LA-level band-count and net council-tax targets."""
    df = _load_table()
    if df is None or df.empty:
        return []

    targets: list[Target] = []

    # NO Band D amount target is emitted here. Do not re-add one.
    #
    # A Band D council tax amount is a *rate* (£ per Band D dwelling),
    # not an additive household control. The loss matrix is an additive
    # objective: every target is compared against a weighted SUM over
    # households. Binding a ~£2,500 per-dwelling rate into that objective
    # is category-wrong — neither as a sum (which would scale with the
    # number of households in the LA) nor as a mean (which would still be
    # a rate inside an additive objective).
    #
    # Microcosm reached the same verdict independently: its
    # ``uk_data_target_parity.json`` concern ``local_council_tax_band_d_rate``
    # is ``reviewed_exclusion`` / ``non_linear``, "Band D currency amounts
    # are per-rate, not additive household controls", to be kept excluded
    # "until a rate-aware non-linear objective exists".
    #
    # ``band_d_amount`` deliberately REMAINS in ``storage/la_council_tax.csv``
    # and is loadable: it is needed as data for household liability
    # computation, and upstream when building the CSV's
    # ``total_council_tax_net`` column (England = MHCLG taxbase x Band D).
    # Only the target emission is dropped. See PolicyEngine/policyengine-uk-data#483.

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
