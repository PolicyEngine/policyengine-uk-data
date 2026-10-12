"""Test salary sacrifice headcount calibration targets.

Source: HMRC, "Salary sacrifice reform for pension contributions"
https://www.gov.uk/government/publications/salary-sacrifice-reform-for-pension-contributions-effective-from-6-april-2029
7.7mn total SS users (3.3mn above 2k cap, 4.3mn below 2k cap)
"""

import os

from policyengine_uk_data.datasets.frs_release import CURRENT_FRS_RELEASE

REDUCED_BUILD = os.environ.get("TESTING") == "1"
# The 32-epoch reduced build stops well short of the OBR salary-sacrifice
# counts that calibration targets: in the seed-0 full build of #529 the users
# are 4.5m at epoch 0, 5.5m at epoch 30 and 7.9m at epoch 510. Before #529,
# reduced builds met the old bounds (20% total, 25% below cap) only because
# SPI-synthetic children were imputed salary sacrifice (about 1.35m users on
# the seed-0 reduced build of main, which held 5.0m users without them).
# #529 gives those children no pay, so the reduced bounds widen to 40% and
# 45%. They still catch a collapse; full builds keep the strict bounds.
# The total was earlier widened from 0.16 after the household-weight
# alignment fix (#436).
TOTAL_TOLERANCE = 0.40 if REDUCED_BUILD else 0.20
TOLERANCE = 0.45 if REDUCED_BUILD else 0.15
# Widened under the reduced-epoch CI build after the benunit-table sort fix
# (#462) shifted the calibration starting point (observed 20.4% on a
# TESTING=1 run), following the precedent of TOTAL_TOLERANCE (#436) and
# TOLERANCE above. Full builds keep the strict 20%.
ABOVE_CAP_TOLERANCE = 0.25 if REDUCED_BUILD else 0.20
PERIOD = CURRENT_FRS_RELEASE.calibration_year


def test_salary_sacrifice_total_users(baseline):
    """Test that total SS user count is close to 7.7mn."""
    ss = baseline.calculate(
        "pension_contributions_via_salary_sacrifice",
        map_to="person",
        period=PERIOD,
    )
    person_weight = baseline.calculate(
        "person_weight", map_to="person", period=PERIOD
    ).values

    total_users = (person_weight * (ss.values > 0)).sum()
    TARGET = 7_700_000

    assert abs(total_users / TARGET - 1) < TOTAL_TOLERANCE, (
        f"Expected ~{TARGET / 1e6:.1f}mn SS users, "
        f"got {total_users / 1e6:.1f}mn ({total_users / TARGET * 100:.0f}% of target)"
    )


def test_salary_sacrifice_below_cap_users(baseline):
    """Test that below-cap (<=2k) SS users are close to 4.3mn."""
    ss = baseline.calculate(
        "pension_contributions_via_salary_sacrifice",
        map_to="person",
        period=PERIOD,
    )
    person_weight = baseline.calculate(
        "person_weight", map_to="person", period=PERIOD
    ).values

    below_cap = (ss.values > 0) & (ss.values <= 2000)
    total_below_cap = (person_weight * below_cap).sum()
    TARGET = 4_300_000

    assert abs(total_below_cap / TARGET - 1) < TOLERANCE, (
        f"Expected ~{TARGET / 1e6:.1f}mn below-cap SS users, "
        f"got {total_below_cap / 1e6:.1f}mn ({total_below_cap / TARGET * 100:.0f}% of target)"
    )


def test_salary_sacrifice_above_cap_users(baseline):
    """Test that above-cap (>2k) SS users are close to 3.3mn."""
    ss = baseline.calculate(
        "pension_contributions_via_salary_sacrifice",
        map_to="person",
        period=PERIOD,
    )
    person_weight = baseline.calculate(
        "person_weight", map_to="person", period=PERIOD
    ).values

    above_cap = ss.values > 2000
    total_above_cap = (person_weight * above_cap).sum()
    TARGET = 3_300_000

    assert abs(total_above_cap / TARGET - 1) < ABOVE_CAP_TOLERANCE, (
        f"Expected ~{TARGET / 1e6:.1f}mn above-cap SS users, "
        f"got {total_above_cap / 1e6:.1f}mn ({total_above_cap / TARGET * 100:.0f}% of target)"
    )
