"""Backward-compatible aliases for the current public UK transfer builder.

The checked-in ``policybench_transfer_2025`` artifacts remain the original
1,000-household proof-of-method files for historical comparison. The Python
entry points below intentionally alias the current 28,532-household
``enhanced_cps_2025`` builder instead of recreating those legacy artifacts.
"""

from policyengine_uk_data.datasets.enhanced_cps import (
    ENHANCED_CPS_SOURCE_FILE as POLICYBENCH_TRANSFER_SOURCE_FILE,
    create_enhanced_cps as create_policybench_transfer,
    save_enhanced_cps as save_policybench_transfer,
)

__all__ = [
    "POLICYBENCH_TRANSFER_SOURCE_FILE",
    "create_policybench_transfer",
    "save_policybench_transfer",
]
