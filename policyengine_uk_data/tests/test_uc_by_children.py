"""Test UC households by number of children calibration targets.

Validates that the weighted count of UC households split by number of
children (0, 1, 2, 3+) matches the latest GB-wide UC household totals
used for local calibration targets.

Source: DWP Stat-Xplore, UC Households dataset
https://stat-xplore.dwp.gov.uk/
"""

import pytest
from policyengine_uk_data.targets.sources.local_uc import (
    _scaled_uc_children_by_country,
)

# Latest GB UC household totals by children count used by local_uc.py.
_TARGETS = {
    "0_children": 2_937_389,
    "1_child": 1_222_944,
    "2_children": 1_058_967,
    "3plus_children": 716_200,
}

TOLERANCE = 0.30  # 30% relative tolerance


def test_scaled_country_children_buckets_match_latest_gb_totals():
    """Scaled country buckets should recover the latest GB child-count totals."""
    country_buckets = _scaled_uc_children_by_country(5_935_500)
    gb_totals = country_buckets["E"] + country_buckets["W"] + country_buckets["S"]
    assert gb_totals.tolist() == [
        _TARGETS["0_children"],
        _TARGETS["1_child"],
        _TARGETS["2_children"],
        _TARGETS["3plus_children"],
    ]


@pytest.mark.parametrize(
    "bucket,target",
    list(_TARGETS.items()),
    ids=list(_TARGETS.keys()),
)
def test_uc_households_by_children(baseline, bucket, target):
    """Test that UC households by children count matches Stat-Xplore data."""
    uc = baseline.calculate("universal_credit", period=2025).values
    on_uc = baseline.map_result(uc > 0, "benunit", "household") > 0

    is_child = baseline.calculate("is_child", map_to="person", period=2025).values
    children_per_hh = baseline.map_result(is_child, "person", "household")

    if bucket == "0_children":
        match = on_uc & (children_per_hh == 0)
    elif bucket == "1_child":
        match = on_uc & (children_per_hh == 1)
    elif bucket == "2_children":
        match = on_uc & (children_per_hh == 2)
    else:  # 3plus_children
        match = on_uc & (children_per_hh >= 3)

    household_weight = baseline.calculate("household_weight", period=2025).values
    actual = (household_weight * match).sum()

    assert abs(actual / target - 1) < TOLERANCE, (
        f"UC households with {bucket}: expected {target / 1e3:.0f}k, "
        f"got {actual / 1e3:.0f}k ({actual / target * 100:.0f}% of target)"
    )
