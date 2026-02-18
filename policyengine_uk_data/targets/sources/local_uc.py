"""Local area UC household targets from DWP Stat-Xplore.

UC household counts by parliamentary constituency and local authority,
loaded from pre-downloaded Stat-Xplore exports and scaled to match
national UC payment distribution totals.

Source: DWP Stat-Xplore
https://stat-xplore.dwp.gov.uk
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)

_REF = "https://stat-xplore.dwp.gov.uk"


def get_constituency_uc_targets() -> pd.Series:
    """UC household counts for 650 constituencies (positional order).

    Returns Series of household_count values, aligned to the same
    ordering as the constituency age.csv.
    """
    from policyengine_uk_data.utils.uc_data import uc_pc_households

    return uc_pc_households.household_count


def get_la_uc_targets() -> pd.Series:
    """UC household counts for 360 local authorities (positional order).

    Returns Series of household_count values, aligned to the same
    ordering as the LA age.csv.
    """
    from policyengine_uk_data.utils.uc_data import uc_la_households

    return uc_la_households.household_count


REFERENCE_URL = _REF
