"""Local area UC household targets from DWP Stat-Xplore.

UC household counts by parliamentary constituency and local authority,
loaded from pre-downloaded Stat-Xplore exports and scaled to match
national UC payment distribution totals.

Also provides UC household counts split by number of children, using
country-level shares last observed in Stat-Xplore in November 2023,
scaled to the latest GB-wide UC household totals by children count.
This keeps the local split aligned to current national family-size
totals without requiring a fresh protected Stat-Xplore country export
for every release.

Source: DWP Stat-Xplore
https://stat-xplore.dwp.gov.uk
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_REF = "https://stat-xplore.dwp.gov.uk"

# Last observed country-level UC household counts by number of children
# from the November 2023 Stat-Xplore household export. Keys:
# (0 children, 1 child, 2 children, 3+ children)
_UC_CHILDREN_BY_COUNTRY_BASE_2023 = {
    "E": np.array([2_411_993, 948_304, 802_992, 495_279], dtype=float),
    "W": np.array([141_054, 52_953, 44_348, 26_372], dtype=float),
    "S": np.array([253_609, 86_321, 66_829, 35_036], dtype=float),
}

# Latest GB-wide UC household totals by number of children from the
# 2025 national claimant counts in dwp.py, with the 0-children bucket
# inferred from the current GB total household count.
_GB_UC_2025_CHILDREN_BUCKETS = np.array(
    [1_222_944, 1_058_967, 473_500 + 166_790 + 74_050 + 1_860],
    dtype=float,
)


def _scaled_uc_children_by_country(gb_total_households: float) -> dict[str, np.ndarray]:
    zero_children_total = gb_total_households - _GB_UC_2025_CHILDREN_BUCKETS.sum()
    gb_bucket_totals = np.array(
        [zero_children_total, *_GB_UC_2025_CHILDREN_BUCKETS],
        dtype=float,
    )
    base_totals = sum(_UC_CHILDREN_BY_COUNTRY_BASE_2023.values())
    scaled = {}
    for country, base_counts in _UC_CHILDREN_BY_COUNTRY_BASE_2023.items():
        shares = np.divide(
            base_counts,
            base_totals,
            out=np.zeros_like(base_counts),
            where=base_totals > 0,
        )
        scaled[country] = np.round(shares * gb_bucket_totals).astype(float)

    gb_sum = sum(scaled.values())
    rounding_diff = gb_bucket_totals - gb_sum
    # Keep the GB totals exact after per-country rounding drift.
    scaled["E"] = scaled["E"] + rounding_diff

    # Northern Ireland still falls back to GB-wide proportions because the
    # public export in this repo does not include a children-count split.
    scaled["N"] = gb_bucket_totals.copy()
    return scaled


def get_constituency_uc_targets() -> pd.Series:
    """UC household counts for 650 constituencies (positional order).

    Returns Series of household_count values, aligned to the same
    ordering as the constituency age.csv.
    """
    from policyengine_uk_data.utils.uc_data import uc_pc_households

    return uc_pc_households.household_count


def get_constituency_uc_by_children_targets() -> pd.DataFrame:
    """UC households split by 0, 1, 2, 3+ children for 650 constituencies.

    Applies country-level proportions from Stat-Xplore to each
    constituency's total UC count.  Returns a DataFrame with columns
    ``uc_hh_0_children``, ``uc_hh_1_child``, ``uc_hh_2_children``,
    ``uc_hh_3plus_children``, in the same positional order as
    :func:`get_constituency_uc_targets`.
    """
    from policyengine_uk_data.utils.uc_data import uc_pc_households
    from policyengine_uk_data.storage import STORAGE_FOLDER

    codes = pd.read_csv(STORAGE_FOLDER / "constituencies_2024.csv")["code"]
    totals = uc_pc_households.household_count.values.astype(float)

    result = pd.DataFrame(index=range(len(totals)))
    cols = [
        "uc_hh_0_children",
        "uc_hh_1_child",
        "uc_hh_2_children",
        "uc_hh_3plus_children",
    ]
    for col in cols:
        result[col] = 0.0

    gb_total = totals[codes.str[0].isin(["E", "W", "S"])].sum()
    country_buckets = _scaled_uc_children_by_country(gb_total)

    for i, (total, code) in enumerate(zip(totals, codes)):
        country_prefix = code[0]
        proportions = country_buckets.get(
            country_prefix,
            country_buckets["N"],  # fallback
        )
        shares = proportions / proportions.sum()
        for j, col in enumerate(cols):
            result.loc[i, col] = round(total * shares[j])

    return result


def get_la_uc_targets() -> pd.Series:
    """UC household counts for 360 local authorities (positional order).

    Returns Series of household_count values, aligned to the same
    ordering as the LA age.csv.
    """
    from policyengine_uk_data.utils.uc_data import uc_la_households

    return uc_la_households.household_count


REFERENCE_URL = _REF
