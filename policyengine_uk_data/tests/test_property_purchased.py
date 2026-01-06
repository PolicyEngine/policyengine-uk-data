"""
Test that property_purchased is set correctly in the enhanced FRS dataset.

The property_purchased variable should be stochastically set based on
UK housing transaction rates (~3.85% of households per year).

Sources:
- Transactions: HMRC 2024 - 1.1m/year
  https://www.gov.uk/government/statistics/monthly-property-transactions-completed-in-the-uk-with-value-40000-or-above
- Households: ONS 2024 - 28.6m
  https://www.ons.gov.uk/peoplepopulationandcommunity/birthsdeathsandmarriages/families/bulletins/familiesandhouseholds/2024
- Rate: 1.1m / 28.6m = 3.85%

Verification against official SDLT revenue (2024-25):
- Official SDLT: £13.9bn (https://www.gov.uk/government/statistics/uk-stamp-tax-statistics)
- With fix (3.85%): ~£15bn (close to official)
- Without fix (100%): £370bn (26x too high)
"""

import pytest


PROPERTY_PURCHASE_RATE = 0.0385


def test_property_purchased_rate(baseline):
    """Test that property_purchased rate is approximately 3.85%."""
    property_purchased = baseline.calculate("property_purchased", 2025).values

    n_households = len(property_purchased)
    true_count = property_purchased.sum()
    actual_rate = true_count / n_households

    # Rate should be approximately 3.85% (allow for random variation)
    target_rate = PROPERTY_PURCHASE_RATE
    tolerance = 0.02

    assert (
        abs(actual_rate - target_rate) < tolerance
    ), f"property_purchased rate {actual_rate:.2%} is not close to target {target_rate:.2%}"


def test_property_purchased_not_all_true(baseline):
    """Test that not all households have property_purchased = True."""
    property_purchased = baseline.calculate("property_purchased", 2025).values

    true_count = property_purchased.sum()
    n_households = len(property_purchased)

    # Should NOT be 100% True (the bug we fixed)
    assert (
        true_count < n_households * 0.1
    ), f"Too many households have property_purchased=True ({true_count}/{n_households})"


def test_property_purchased_has_some_true(baseline):
    """Test that some households have property_purchased = True."""
    property_purchased = baseline.calculate("property_purchased", 2025).values

    true_count = property_purchased.sum()

    # Should have some True values (realistic purchasing rate)
    assert true_count > 0, "No households have property_purchased=True"


def test_sdlt_total_reasonable(baseline):
    """Test that total SDLT revenue is realistic.

    Official SDLT revenue (2024-25): £13.9bn
    Source: https://www.gov.uk/government/statistics/uk-stamp-tax-statistics

    Without fix (100% property_purchased=True): £370bn (26x too high)
    With fix (3.85% rate): ~£15bn (close to official)
    """
    expected_sdlt = baseline.calculate("expected_sdlt", 2025).values
    household_weight = baseline.calculate("household_weight", 2025).values
    total_sdlt = (expected_sdlt * household_weight).sum()

    # Total SDLT should be within reasonable range of official figures
    min_sdlt = 5e9  # £5bn minimum
    max_sdlt = 50e9  # £50bn maximum (official is ~£14bn)

    assert total_sdlt > min_sdlt, (
        f"Total SDLT £{total_sdlt/1e9:.1f}bn is too low "
        f"(minimum expected: £{min_sdlt/1e9:.1f}bn)"
    )

    assert total_sdlt < max_sdlt, (
        f"Total SDLT £{total_sdlt/1e9:.1f}bn is unrealistically high "
        f"(maximum expected: £{max_sdlt/1e9:.1f}bn). "
        f"Official SDLT is ~£14bn. "
        "This suggests property_purchased may be incorrectly set to True for all households."
    )
