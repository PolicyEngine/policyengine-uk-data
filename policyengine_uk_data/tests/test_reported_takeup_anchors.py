"""Unit tests for reported-anchor takeup logic."""

from __future__ import annotations

import numpy as np

from policyengine_uk_data.utils.takeup import assign_takeup_with_reported_anchors


def test_no_reported_mask_falls_back_to_draws_less_than_rate():
    rng = np.random.default_rng(0)
    draws = rng.random(1000)
    result = assign_takeup_with_reported_anchors(draws, 0.3)
    # Expected share close to rate
    assert abs(result.mean() - 0.3) < 0.05
    # Identical to plain draws < rate
    assert (result == (draws < 0.3)).all()


def test_reported_anchor_forces_true_for_reporters():
    rng = np.random.default_rng(1)
    draws = rng.random(1000)
    reported_mask = np.zeros(1000, dtype=bool)
    reported_mask[:100] = True
    result = assign_takeup_with_reported_anchors(
        draws, 0.3, reported_mask=reported_mask
    )
    # Every reporter is True
    assert result[:100].all()


def test_reported_anchor_hits_target_rate():
    rng = np.random.default_rng(2)
    draws = rng.random(10000)
    reported_mask = np.zeros(10000, dtype=bool)
    reported_mask[:1000] = True  # 10% reporters
    result = assign_takeup_with_reported_anchors(
        draws, 0.3, reported_mask=reported_mask
    )
    # Overall rate should be close to 30%
    assert abs(result.mean() - 0.3) < 0.02


def test_reported_anchor_when_reporters_exceed_target():
    rng = np.random.default_rng(3)
    draws = rng.random(1000)
    reported_mask = np.zeros(1000, dtype=bool)
    reported_mask[:500] = True  # 50% reporters
    # Target 30% but reporters already at 50% — everyone reporting stays in.
    result = assign_takeup_with_reported_anchors(
        draws, 0.3, reported_mask=reported_mask
    )
    assert result[:500].all()
    assert not result[500:].any()


def test_reported_mask_length_validation():
    draws = np.random.default_rng(4).random(100)
    reported_mask = np.zeros(50, dtype=bool)
    try:
        assign_takeup_with_reported_anchors(draws, 0.3, reported_mask=reported_mask)
    except ValueError as exc:
        assert "must align" in str(exc)
    else:
        raise AssertionError("expected ValueError for misaligned reported_mask")
