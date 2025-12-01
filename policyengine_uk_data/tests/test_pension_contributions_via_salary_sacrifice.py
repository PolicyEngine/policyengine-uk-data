def test_pension_contributions_via_salary_sacrifice(baseline):
    """Test that pension_contributions_via_salary_sacrifice loads and has reasonable values."""
    values = baseline.calculate(
        "pension_contributions_via_salary_sacrifice", period=2025
    )

    # Basic validation: all values should be non-negative
    assert (
        values >= 0
    ).all(), "Salary sacrifice pension contributions must be non-negative"

    # Should have some non-zero values (not everyone uses salary sacrifice, but some do)
    total = values.sum()
    assert (
        total > 0
    ), f"Expected some salary sacrifice contributions, got {total}"

    # Reasonableness check: total should be less than total employment income
    # This is a very loose check just to catch major issues
    employment_income = baseline.calculate("employment_income", period=2025)
    total_employment = employment_income.sum()
    assert (
        total < total_employment
    ), f"Salary sacrifice contributions ({total/1e9:.1f}B) cannot exceed total employment income ({total_employment/1e9:.1f}B)"
