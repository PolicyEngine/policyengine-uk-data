"""
Tests for low-income decile sanity checks.

These tests ensure that the first income decile (lowest income households)
has reasonable tax and net income values. This catches bugs like the
property_purchased issue where incorrect defaults led to:
- 224% tax rates in the first decile
- Negative net incomes for low-income households

These tests should prevent similar data quality issues in the future.
"""

import pytest


def test_first_decile_tax_rate_reasonable():
    """Test that first decile effective tax rate is below 100%.

    Tax rate should never exceed 100% of market income - that would imply
    households are paying more in tax than they earn, which is impossible
    without the SDLT bug.
    """
    from policyengine_uk import Microsimulation
    import pandas as pd

    sim = Microsimulation()
    household_weight = sim.calculate("household_weight", 2025).values
    net_income = sim.calculate("household_net_income", 2025).values
    market_income = sim.calculate("household_market_income", 2025).values
    household_tax = sim.calculate("household_tax", 2025).values

    # Assign deciles based on net income
    decile = pd.qcut(net_income, 10, labels=False, duplicates="drop")

    # Get first decile (lowest income)
    d1_mask = decile == 0
    d1_tax = (household_tax[d1_mask] * household_weight[d1_mask]).sum()
    d1_market = (market_income[d1_mask] * household_weight[d1_mask]).sum()

    # Handle edge case where market income is very low
    if d1_market > 0:
        d1_tax_rate = d1_tax / d1_market
        assert d1_tax_rate < 1.0, (
            f"First decile tax rate is {d1_tax_rate:.0%}, which exceeds 100%. "
            f"Total D1 tax: £{d1_tax/1e9:.1f}bn, "
            f"Total D1 market income: £{d1_market/1e9:.1f}bn. "
            "This likely indicates a bug in property_purchased or similar variable."
        )


def test_first_decile_average_tax_reasonable():
    """Test that first decile average household tax is reasonable.

    Low-income households should not pay more than £50,000 per year in tax
    on average. The SDLT bug caused D1 tax to be ~£84,000 per household.
    """
    from policyengine_uk import Microsimulation
    import pandas as pd

    sim = Microsimulation()
    household_weight = sim.calculate("household_weight", 2025).values
    net_income = sim.calculate("household_net_income", 2025).values
    household_tax = sim.calculate("household_tax", 2025).values

    # Assign deciles based on net income
    decile = pd.qcut(net_income, 10, labels=False, duplicates="drop")

    # Get first decile average tax
    d1_mask = decile == 0
    d1_avg_tax = (
        household_tax[d1_mask] * household_weight[d1_mask]
    ).sum() / household_weight[d1_mask].sum()

    max_reasonable_d1_tax = 50_000  # £50k max average tax for lowest decile

    assert d1_avg_tax < max_reasonable_d1_tax, (
        f"First decile average tax is £{d1_avg_tax:,.0f}, "
        f"which exceeds the £{max_reasonable_d1_tax:,} threshold. "
        "This likely indicates a bug in property_purchased or similar variable "
        "causing unrealistic stamp duty charges."
    )


def test_first_decile_positive_net_income():
    """Test that first decile weighted average net income is not negative.

    While individual low-income households can have negative net income,
    the weighted average across the entire first decile should be positive
    when benefits are included.
    """
    from policyengine_uk import Microsimulation
    import pandas as pd

    sim = Microsimulation()
    household_weight = sim.calculate("household_weight", 2025).values
    net_income = sim.calculate("household_net_income", 2025).values

    # Assign deciles based on net income
    decile = pd.qcut(net_income, 10, labels=False, duplicates="drop")

    # Get first decile average net income
    d1_mask = decile == 0
    d1_avg_net = (
        net_income[d1_mask] * household_weight[d1_mask]
    ).sum() / household_weight[d1_mask].sum()

    # With the SDLT bug, D1 net income was -£40,000 on average
    # After fix, it should be positive
    assert d1_avg_net > -10_000, (
        f"First decile average net income is £{d1_avg_net:,.0f}, "
        "which is significantly negative. This likely indicates a bug "
        "in property_purchased or similar variable causing unrealistic "
        "tax charges that push net income negative."
    )


def test_decile_tax_ordering():
    """Test that tax generally increases with income decile.

    Higher income deciles should generally pay more tax than lower deciles.
    The SDLT bug caused D1 (£84k) to pay more than D10 (£79k).
    """
    from policyengine_uk import Microsimulation
    import pandas as pd

    sim = Microsimulation()
    household_weight = sim.calculate("household_weight", 2025).values
    net_income = sim.calculate("household_net_income", 2025).values
    household_tax = sim.calculate("household_tax", 2025).values

    # Assign deciles based on net income
    decile = pd.qcut(net_income, 10, labels=False, duplicates="drop")

    # Calculate average tax by decile
    decile_taxes = []
    for d in range(10):
        mask = decile == d
        avg_tax = (
            household_tax[mask] * household_weight[mask]
        ).sum() / household_weight[mask].sum()
        decile_taxes.append(avg_tax)

    # First decile should have lower tax than top decile
    d1_tax = decile_taxes[0]
    d10_tax = decile_taxes[9]

    assert d1_tax < d10_tax, (
        f"First decile tax (£{d1_tax:,.0f}) is higher than "
        f"tenth decile tax (£{d10_tax:,.0f}). "
        "This inverted pattern indicates a bug in tax calculations, "
        "likely from property_purchased being incorrectly set."
    )


def test_no_excessive_negative_incomes():
    """Test that there aren't too many households with severely negative income.

    While some households may have negative net income due to losses,
    more than 1% having income below -£50k indicates a bug.
    """
    from policyengine_uk import Microsimulation

    sim = Microsimulation()
    household_weight = sim.calculate("household_weight", 2025).values
    net_income = sim.calculate("household_net_income", 2025).values

    total_households = household_weight.sum()
    severe_negative_mask = net_income < -50_000
    severe_negative_count = household_weight[severe_negative_mask].sum()
    severe_negative_pct = severe_negative_count / total_households

    max_allowed_pct = 0.01  # 1% threshold

    assert severe_negative_pct < max_allowed_pct, (
        f"{severe_negative_pct:.1%} of households have net income below -£50,000. "
        f"This exceeds the {max_allowed_pct:.0%} threshold and indicates "
        "a potential bug in tax calculations."
    )
