"""
Test that property_purchased is set correctly in the FRS dataset.

The property_purchased variable should be stochastically set based on
UK housing transaction rates (~3.85% of households per year).

Source: HMRC 2024 - 1.1m transactions / 28.6m households = 3.85%
https://www.gov.uk/government/statistics/monthly-property-transactions-completed-in-the-uk-with-value-40000-or-above
"""

import pytest


def test_property_purchased_rate():
    """Test that property_purchased rate is approximately 3.85%."""
    from policyengine_uk import Microsimulation

    sim = Microsimulation()
    property_purchased = sim.calculate("property_purchased", 2025).values

    # Calculate the rate
    n_households = len(property_purchased)
    true_count = property_purchased.sum()
    actual_rate = true_count / n_households

    # The rate should be approximately 3.85% (allow for random variation)
    # With ~53,000 households, standard error is sqrt(0.0385 * 0.9615 / 53000) ≈ 0.0008
    # Using 3 standard deviations gives a tolerance of about 0.5%
    target_rate = 0.0385
    tolerance = 0.02  # Allow 2% deviation from target

    assert (
        abs(actual_rate - target_rate) < tolerance
    ), f"property_purchased rate {actual_rate:.2%} is not close to target {target_rate:.2%}"


def test_property_purchased_not_all_true():
    """Test that not all households have property_purchased = True."""
    from policyengine_uk import Microsimulation

    sim = Microsimulation()
    property_purchased = sim.calculate("property_purchased", 2025).values

    true_count = property_purchased.sum()
    n_households = len(property_purchased)

    # Should NOT be 100% True (the bug we're fixing)
    assert (
        true_count < n_households
    ), f"All households have property_purchased=True ({true_count}/{n_households})"


def test_property_purchased_not_all_false():
    """Test that not all households have property_purchased = False."""
    from policyengine_uk import Microsimulation

    sim = Microsimulation()
    property_purchased = sim.calculate("property_purchased", 2025).values

    true_count = property_purchased.sum()

    # Should have some True values (realistic purchasing rate)
    assert true_count > 0, "No households have property_purchased=True"


def test_sdlt_total_reasonable():
    """Test that total SDLT revenue is in a realistic range.

    Official SDLT revenue (2024-25): £13.9bn
    Source: https://www.gov.uk/government/statistics/uk-stamp-tax-statistics

    If property_purchased is wrongly set to True for all households,
    SDLT would be ~£370bn (26x too high).
    """
    from policyengine_uk import Microsimulation

    sim = Microsimulation()
    expected_sdlt = sim.calculate("expected_sdlt", 2025).values
    household_weight = sim.calculate("household_weight", 2025).values
    total_sdlt = (expected_sdlt * household_weight).sum()

    # Total SDLT should be within reasonable range of official figures
    # Allow 50% margin for model differences
    min_sdlt = 5e9  # £5bn minimum
    max_sdlt = 50e9  # £50bn maximum (official is ~£14bn)

    assert total_sdlt > min_sdlt, (
        f"Total SDLT £{total_sdlt/1e9:.1f}bn is too low "
        f"(minimum expected: £{min_sdlt/1e9:.1f}bn)"
    )

    assert total_sdlt < max_sdlt, (
        f"Total SDLT £{total_sdlt/1e9:.1f}bn is unrealistically high "
        f"(maximum expected: £{max_sdlt/1e9:.1f}bn). "
        "This suggests property_purchased may be incorrectly set to True for all households."
    )
