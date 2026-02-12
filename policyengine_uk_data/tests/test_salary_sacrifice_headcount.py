"""Test salary sacrifice headcount calibration targets.

Source: HMRC, "Salary sacrifice reform for pension contributions"
https://www.gov.uk/government/publications/salary-sacrifice-reform-for-pension-contributions-effective-from-6-april-2029
7.7mn total SS users (3.3mn above 2k cap, 4.3mn below 2k cap)
"""

import pytest

TOLERANCE = 0.15  # 15% relative tolerance


@pytest.mark.xfail(
    reason="Will pass after recalibration with new headcount targets"
)
def test_salary_sacrifice_total_users(baseline):
    """Test that total SS user count is close to 7.7mn."""
    ss = baseline.calculate(
        "pension_contributions_via_salary_sacrifice",
        map_to="person",
        period=2025,
    )
    person_weight = baseline.calculate(
        "person_weight", map_to="person", period=2025
    ).values

    total_users = (person_weight * (ss.values > 0)).sum()
    TARGET = 7_700_000

    assert abs(total_users / TARGET - 1) < TOLERANCE, (
        f"Expected ~{TARGET/1e6:.1f}mn SS users, "
        f"got {total_users/1e6:.1f}mn ({total_users/TARGET*100:.0f}% of target)"
    )


@pytest.mark.xfail(
    reason="Will pass after recalibration with new headcount targets"
)
def test_salary_sacrifice_below_cap_users(baseline):
    """Test that below-cap (<=2k) SS users are close to 4.3mn."""
    ss = baseline.calculate(
        "pension_contributions_via_salary_sacrifice",
        map_to="person",
        period=2025,
    )
    person_weight = baseline.calculate(
        "person_weight", map_to="person", period=2025
    ).values

    below_cap = (ss.values > 0) & (ss.values <= 2000)
    total_below_cap = (person_weight * below_cap).sum()
    TARGET = 4_300_000

    assert abs(total_below_cap / TARGET - 1) < TOLERANCE, (
        f"Expected ~{TARGET/1e6:.1f}mn below-cap SS users, "
        f"got {total_below_cap/1e6:.1f}mn ({total_below_cap/TARGET*100:.0f}% of target)"
    )


@pytest.mark.xfail(
    reason="Will pass after recalibration with new headcount targets"
)
def test_salary_sacrifice_above_cap_users(baseline):
    """Test that above-cap (>2k) SS users are close to 3.3mn."""
    ss = baseline.calculate(
        "pension_contributions_via_salary_sacrifice",
        map_to="person",
        period=2025,
    )
    person_weight = baseline.calculate(
        "person_weight", map_to="person", period=2025
    ).values

    above_cap = ss.values > 2000
    total_above_cap = (person_weight * above_cap).sum()
    TARGET = 3_300_000

    assert abs(total_above_cap / TARGET - 1) < TOLERANCE, (
        f"Expected ~{TARGET/1e6:.1f}mn above-cap SS users, "
        f"got {total_above_cap/1e6:.1f}mn ({total_above_cap/TARGET*100:.0f}% of target)"
    )
