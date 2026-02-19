"""Local area UC household targets from DWP Stat-Xplore.

UC household counts by parliamentary constituency and local authority,
loaded from pre-downloaded Stat-Xplore exports and scaled to match
national UC payment distribution totals.

Also provides UC household counts split by number of children, using
country-level proportions from Stat-Xplore (November 2023) applied to
each constituency's total.  This ensures the reweighting algorithm
places adequate weight on larger families in every constituency.

Source: DWP Stat-Xplore
https://stat-xplore.dwp.gov.uk
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_REF = "https://stat-xplore.dwp.gov.uk"

# Country-level UC households by number of children (Nov 2023, Stat-Xplore).
# Used to split each constituency's UC total into children-count buckets.
# Keys: (0 children, 1 child, 2 children, 3+ children)
_UC_CHILDREN_BY_COUNTRY = {
    "E": np.array([2_411_993, 948_304, 802_992, 495_279], dtype=float),
    "W": np.array([141_054, 52_953, 44_348, 26_372], dtype=float),
    "S": np.array([253_609, 86_321, 66_829, 35_036], dtype=float),
    # Northern Ireland: use GB-wide proportions as fallback
    "N": np.array(
        [
            2_411_993 + 141_054 + 253_609,
            948_304 + 52_953 + 86_321,
            802_992 + 44_348 + 66_829,
            495_279 + 26_372 + 35_036,
        ],
        dtype=float,
    ),
}


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

    for i, (total, code) in enumerate(zip(totals, codes)):
        country_prefix = code[0]
        proportions = _UC_CHILDREN_BY_COUNTRY.get(
            country_prefix,
            _UC_CHILDREN_BY_COUNTRY["N"],  # fallback
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
