"""Official data source assets."""

from policyengine_uk_data.assets.sources.obr import obr_receipts_observations
from policyengine_uk_data.assets.sources.dwp_stat_xplore import (
    dwp_stat_xplore_observations,
)

__all__ = [
    "obr_receipts_observations",
    "dwp_stat_xplore_observations",
]
