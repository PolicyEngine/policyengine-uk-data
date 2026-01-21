"""Test Scotland UC households with child under 1 calibration target.

Source: DWP Stat-Xplore, UC Households dataset, November 2023
https://stat-xplore.dwp.gov.uk/
Filters: Scotland, Age of Youngest Child = 0
Result: 13,992 households (~14k)
"""

import pytest


@pytest.mark.xfail(
    reason="Will pass after recalibration with new scotland_uc_households_child_under_1 target"
)
def test_scotland_uc_households_child_under_1(baseline):
    """Test that UC households in Scotland with child under 1 matches DWP data.

    Target: ~14,000 households (13,992 from Stat-Xplore November 2023)
    Source: DWP Stat-Xplore UC Households dataset
    """
    region = baseline.calculate(
        "region", map_to="household", period=2025
    )
    uc = baseline.calculate("universal_credit", period=2025).values
    household_weight = baseline.calculate(
        "household_weight", map_to="household", period=2025
    ).values

    # Check if household has child under 1
    is_child = baseline.calculate(
        "is_child", map_to="person", period=2025
    ).values
    age = baseline.calculate("age", map_to="person", period=2025).values

    child_under_1 = is_child & (age < 1)
    has_child_under_1 = (
        baseline.map_result(child_under_1, "person", "household") > 0
    )

    scotland_uc_child_under_1 = (
        (region.values == "SCOTLAND") & (uc > 0) & has_child_under_1
    )
    total = (household_weight * scotland_uc_child_under_1).sum()

    TARGET = 14_000  # DWP Stat-Xplore November 2023: 13,992 rounded to 14k
    TOLERANCE = 0.15  # 15% tolerance

    assert abs(total / TARGET - 1) < TOLERANCE, (
        f"Expected ~{TARGET/1000:.0f}k UC households with child under 1 in Scotland, "
        f"got {total/1000:.0f}k ({total/TARGET*100:.0f}% of target)"
    )
