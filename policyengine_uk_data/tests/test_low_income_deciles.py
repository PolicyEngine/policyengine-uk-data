"""
Tests for low-income decile sanity checks.

These tests ensure that the first income decile (lowest income households)
has reasonable tax and net income values. This catches bugs like the
property_purchased issue where incorrect defaults led to:
- 224% tax rates in the first decile
- Negative net incomes for low-income households

These tests confirm the fix works and prevent similar issues in the future.
"""

import pytest
import pandas as pd


def test_first_decile_tax_rate_reasonable(baseline):
    """Test that first decile effective tax rate is below 100%.

    Without fix: 224% tax rate (impossible)
    With fix: Should be well below 100%
    """
    household_weight = baseline.calculate("household_weight", 2025).values
    net_income = baseline.calculate("household_net_income", 2025).values
    market_income = baseline.calculate("household_market_income", 2025).values
    household_tax = baseline.calculate("household_tax", 2025).values

    decile = pd.qcut(net_income, 10, labels=False, duplicates="drop")

    d1_mask = decile == 0
    d1_tax = (household_tax[d1_mask] * household_weight[d1_mask]).sum()
    d1_market = (market_income[d1_mask] * household_weight[d1_mask]).sum()

    if d1_market > 0:
        d1_tax_rate = d1_tax / d1_market
        assert d1_tax_rate < 1.0, (
            f"First decile tax rate is {d1_tax_rate:.0%}, which exceeds 100%. "
            f"Total D1 tax: £{d1_tax/1e9:.1f}bn, "
            f"Total D1 market income: £{d1_market/1e9:.1f}bn. "
            "This likely indicates a bug in property_purchased or similar variable."
        )


def test_first_decile_average_tax_reasonable(baseline):
    """Test that first decile average household tax is reasonable.

    Without fix: £90,988 average tax (unrealistic)
    With fix: Should be below £50,000
    """
    household_weight = baseline.calculate("household_weight", 2025).values
    net_income = baseline.calculate("household_net_income", 2025).values
    household_tax = baseline.calculate("household_tax", 2025).values

    decile = pd.qcut(net_income, 10, labels=False, duplicates="drop")

    d1_mask = decile == 0
    d1_avg_tax = (
        household_tax[d1_mask] * household_weight[d1_mask]
    ).sum() / household_weight[d1_mask].sum()

    max_reasonable_d1_tax = 50_000

    assert d1_avg_tax < max_reasonable_d1_tax, (
        f"First decile average tax is £{d1_avg_tax:,.0f}, "
        f"which exceeds £{max_reasonable_d1_tax:,}. "
        "This likely indicates a bug in property_purchased or similar variable "
        "causing unrealistic stamp duty charges."
    )


def test_first_decile_net_income_not_severely_negative(baseline):
    """Test that first decile net income is not severely negative.

    Without fix: -£37,452 average (due to massive SDLT)
    With fix: Should be above -£10,000
    """
    household_weight = baseline.calculate("household_weight", 2025).values
    net_income = baseline.calculate("household_net_income", 2025).values

    decile = pd.qcut(net_income, 10, labels=False, duplicates="drop")

    d1_mask = decile == 0
    d1_avg_net = (
        net_income[d1_mask] * household_weight[d1_mask]
    ).sum() / household_weight[d1_mask].sum()

    assert d1_avg_net > -10_000, (
        f"First decile average net income is £{d1_avg_net:,.0f}, "
        "which is significantly negative. This likely indicates a bug "
        "in property_purchased or similar variable causing unrealistic "
        "tax charges that push net income negative."
    )


def test_decile_tax_ordering(baseline):
    """Test that higher deciles pay more tax than lower deciles.

    Without fix: D1 (£90k) > D10 (£79k) - inverted!
    With fix: D1 < D10 (correct ordering)
    """
    household_weight = baseline.calculate("household_weight", 2025).values
    net_income = baseline.calculate("household_net_income", 2025).values
    household_tax = baseline.calculate("household_tax", 2025).values

    decile = pd.qcut(net_income, 10, labels=False, duplicates="drop")

    decile_taxes = []
    for d in range(10):
        mask = decile == d
        avg_tax = (
            household_tax[mask] * household_weight[mask]
        ).sum() / household_weight[mask].sum()
        decile_taxes.append(avg_tax)

    d1_tax = decile_taxes[0]
    d10_tax = decile_taxes[9]

    assert d1_tax < d10_tax, (
        f"First decile tax (£{d1_tax:,.0f}) is higher than "
        f"tenth decile tax (£{d10_tax:,.0f}). "
        "This inverted pattern indicates a bug in tax calculations, "
        "likely from property_purchased being incorrectly set."
    )


def test_no_excessive_negative_incomes(baseline):
    """Test that excessive negative incomes are limited.

    Without fix: 2.3% of households below -£50k
    With fix: Should be below 1%
    """
    household_weight = baseline.calculate("household_weight", 2025).values
    net_income = baseline.calculate("household_net_income", 2025).values

    total_households = household_weight.sum()
    severe_negative_mask = net_income < -50_000
    severe_negative_count = household_weight[severe_negative_mask].sum()
    severe_negative_pct = severe_negative_count / total_households

    max_allowed_pct = 0.01

    assert severe_negative_pct < max_allowed_pct, (
        f"{severe_negative_pct:.1%} of households have net income "
        f"below -£50,000. This exceeds {max_allowed_pct:.0%} and indicates "
        "a potential bug in tax calculations."
    )
