import numpy as np
import pytest


def test_pension_contributions_via_salary_sacrifice(baseline):
    """Test that pension_contributions_via_salary_sacrifice loads and has reasonable values."""
    values = baseline.calculate(
        "pension_contributions_via_salary_sacrifice", period=2025
    )

    # Basic validation: all values should be non-negative
    assert (values >= 0).all(), (
        "Salary sacrifice pension contributions must be non-negative"
    )

    # Should have some non-zero values (not everyone uses salary sacrifice, but some do)
    total = values.sum()
    assert total > 0, f"Expected some salary sacrifice contributions, got {total}"

    # Reasonableness check: total should be less than total employment income
    # This is a very loose check just to catch major issues
    employment_income = baseline.calculate("employment_income", period=2025)
    total_employment = employment_income.sum()
    assert total < total_employment, (
        f"Salary sacrifice contributions ({total / 1e9:.1f}B) cannot exceed total employment income ({total_employment / 1e9:.1f}B)"
    )


def test_salary_sacrifice_needs_pay(baseline):
    """A salary sacrifice comes out of pay: none without pay, none above it."""
    pay = baseline.calculate("employment_income_before_lsr", period=2025).values
    ss = baseline.calculate(
        "pension_contributions_via_salary_sacrifice", period=2025
    ).values
    weight = baseline.calculate("person_weight", period=2025).values

    without_pay = (ss > 0) & (pay <= 0)
    assert not without_pay.any(), (
        f"{without_pay.sum()} people ({weight[without_pay].sum() / 1e3:.0f}k "
        "weighted) have salary sacrifice but no pay"
    )
    # Both columns uprate by the same index, so only rounding can separate them.
    above_pay = ss > pay * (1 + 1e-9) + 1e-6
    assert not above_pay.any(), (
        f"{above_pay.sum()} people sacrifice more than their pay"
    )


def test_salary_sacrifice_cap_gives_nobody_pay_they_do_not_have(baseline):
    """From 2029 the excess over the cap returns to pay, so it needs pay."""
    pay = baseline.calculate("employment_income_before_lsr", period=2029).values
    employment_income = baseline.calculate("employment_income", period=2029).values
    phantom = (pay <= 0) & (employment_income > 0)
    assert not phantom.any(), (
        f"{phantom.sum()} people without pay have employment income in 2029"
    )


def test_model_pay_limit_never_binds_on_the_dataset(baseline):
    """policyengine-uk limits salary sacrifice to pay as the imputation does."""
    limited = "pension_contributions_via_salary_sacrifice_from_pay"
    if limited not in baseline.tax_benefit_system.variables:
        pytest.skip(f"policyengine-uk has no {limited} variable")
    for period in (2025, 2029):
        ss = baseline.calculate(
            "pension_contributions_via_salary_sacrifice", period=period
        ).values
        from_pay = baseline.calculate(limited, period=period).values
        differing = ~np.isclose(from_pay, ss, rtol=1e-6, atol=1e-2)
        assert not differing.any(), (
            f"policyengine-uk limits the salary sacrifice of {differing.sum()} "
            f"people in {period}"
        )
