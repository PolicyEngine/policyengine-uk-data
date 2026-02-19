"""Test UC households by number of children calibration targets.

Validates that the weighted count of UC households split by number of
children (0, 1, 2, 3+) matches DWP Stat-Xplore country-level totals
(November 2023).

Source: DWP Stat-Xplore, UC Households dataset
https://stat-xplore.dwp.gov.uk/
"""

import pytest

# DWP Stat-Xplore November 2023 national totals (GB)
# England + Wales + Scotland
_TARGETS = {
    "0_children": 2_411_993 + 141_054 + 253_609,  # 2,806,656
    "1_child": 948_304 + 52_953 + 86_321,  # 1,087,578
    "2_children": 802_992 + 44_348 + 66_829,  # 914,169
    "3plus_children": 495_279 + 26_372 + 35_036,  # 556,687
}

TOLERANCE = 0.30  # 30% relative tolerance


@pytest.mark.xfail(
    reason="Will pass after recalibration with UC-by-children constituency targets"
)
@pytest.mark.parametrize(
    "bucket,target",
    list(_TARGETS.items()),
    ids=list(_TARGETS.keys()),
)
def test_uc_households_by_children(baseline, bucket, target):
    """Test that UC households by children count matches Stat-Xplore data."""
    uc = baseline.calculate("universal_credit", period=2025).values
    on_uc = baseline.map_result(uc > 0, "benunit", "household") > 0

    is_child = baseline.calculate(
        "is_child", map_to="person", period=2025
    ).values
    children_per_hh = baseline.map_result(is_child, "person", "household")

    if bucket == "0_children":
        match = on_uc & (children_per_hh == 0)
    elif bucket == "1_child":
        match = on_uc & (children_per_hh == 1)
    elif bucket == "2_children":
        match = on_uc & (children_per_hh == 2)
    else:  # 3plus_children
        match = on_uc & (children_per_hh >= 3)

    household_weight = baseline.calculate(
        "household_weight", period=2025
    ).values
    actual = (household_weight * match).sum()

    assert abs(actual / target - 1) < TOLERANCE, (
        f"UC households with {bucket}: expected {target/1e3:.0f}k, "
        f"got {actual/1e3:.0f}k ({actual/target*100:.0f}% of target)"
    )
