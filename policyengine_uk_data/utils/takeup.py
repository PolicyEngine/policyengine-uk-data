"""Shared take-up draw logic with reported-recipient anchoring.

Ported from ``policyengine_us_data/utils/takeup.py``. The core idea: when a
survey respondent reports receiving a benefit, they are by construction a
taker-up; they should be assigned takeup=True with certainty, and the
remaining random fill should hit the target aggregate takeup rate across the
non-reporting eligibles. Pure random draws (the previous UK pattern) ignore
this information and produce noisier calibration.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def assign_takeup_with_reported_anchors(
    draws: np.ndarray,
    rate: float,
    reported_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Apply the SSI/SNAP-style reported-first takeup pattern.

    Reported recipients are always assigned ``takeup=True``. Remaining
    non-reporters are filled probabilistically to reach the target count
    implied by ``rate`` across the full population.

    Args:
        draws: Uniform draws in [0, 1), one per entity.
        rate: Target aggregate takeup rate in [0, 1].
        reported_mask: Boolean array, same length as ``draws``. ``True``
            where the survey reports a positive benefit amount. If ``None``,
            the function falls back to a plain ``draws < rate`` fill.

    Returns:
        Boolean array of the same length as ``draws``, ``True`` for entities
        that take up.
    """
    draws = np.asarray(draws, dtype=np.float64)
    rate = float(rate)

    if reported_mask is None:
        return draws < rate

    reported_mask = np.asarray(reported_mask, dtype=bool)
    if len(reported_mask) != len(draws):
        raise ValueError("reported_mask and draws must align")

    result = reported_mask.copy()
    target_count = int(rate * len(draws))
    remaining_needed = max(0, target_count - int(reported_mask.sum()))
    non_reporters = ~reported_mask
    if not non_reporters.any() or remaining_needed == 0:
        return result

    adjusted_rate = remaining_needed / int(non_reporters.sum())
    result |= non_reporters & (draws < adjusted_rate)
    return result
