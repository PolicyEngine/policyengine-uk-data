"""Backward-compatible aliases for the public UK enhanced CPS dataset."""

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
