"""Tests for SLC student loan calibration targets."""

import pytest


def test_slc_targets_registered():
    """SLC targets appear in the target registry."""
    from policyengine_uk_data.targets.registry import get_all_targets

    targets = {t.name: t for t in get_all_targets()}
    assert "slc/plan_2_borrowers_above_threshold" in targets
    assert "slc/plan_5_borrowers_above_threshold" in targets
    assert "slc/plan_2_borrowers_liable" in targets
    assert "slc/plan_5_borrowers_liable" in targets


def test_slc_plan2_above_threshold_values():
    """Plan 2 above-threshold values match SLC Table 6a HE total."""
    from policyengine_uk_data.targets.registry import get_all_targets

    targets = {t.name: t for t in get_all_targets()}
    p2 = targets["slc/plan_2_borrowers_above_threshold"]
    # Values from Row 11 (HE total, above threshold)
    assert p2.values[2025] == 3_985_000
    assert p2.values[2026] == 4_460_000
    assert p2.values[2030] == 5_205_000


def test_slc_plan5_above_threshold_values():
    """Plan 5 above-threshold values match SLC Table 6a HE total."""
    from policyengine_uk_data.targets.registry import get_all_targets

    targets = {t.name: t for t in get_all_targets()}
    p5 = targets["slc/plan_5_borrowers_above_threshold"]
    # Values from Row 11 (HE total, above threshold)
    assert 2025 not in p5.values  # 0 in 2024-25
    assert p5.values[2026] == 35_000
    assert p5.values[2030] == 1_235_000


def test_slc_plan2_liable_values():
    """Plan 2 liable-to-repay values match SLC Table 6a HE total."""
    from policyengine_uk_data.targets.registry import get_all_targets

    targets = {t.name: t for t in get_all_targets()}
    p2 = targets["slc/plan_2_borrowers_liable"]
    # Values from Row 10 (HE total, liable to repay)
    assert p2.values[2025] == 8_940_000
    assert p2.values[2026] == 9_710_000
    assert p2.values[2030] == 10_525_000


def test_slc_plan5_liable_values():
    """Plan 5 liable-to-repay values match SLC Table 6a HE total."""
    from policyengine_uk_data.targets.registry import get_all_targets

    targets = {t.name: t for t in get_all_targets()}
    p5 = targets["slc/plan_5_borrowers_liable"]
    # Values from Row 10 (HE total, liable to repay)
    assert p5.values[2025] == 10_000
    assert p5.values[2026] == 230_000
    assert p5.values[2030] == 3_400_000


def test_liable_exceeds_above_threshold():
    """Liable-to-repay counts exceed above-threshold counts."""
    from policyengine_uk_data.targets.registry import get_all_targets

    targets = {t.name: t for t in get_all_targets()}
    p2_liable = targets["slc/plan_2_borrowers_liable"]
    p2_above = targets["slc/plan_2_borrowers_above_threshold"]

    for year in p2_above.values:
        if year in p2_liable.values:
            assert p2_liable.values[year] > p2_above.values[year]
