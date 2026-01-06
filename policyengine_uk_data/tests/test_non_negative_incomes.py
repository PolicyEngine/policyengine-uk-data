import pytest

INCOME_VARIABLES = [
    "employment_income",
    "self_employment_income",
    "tax_free_savings_income",
    "savings_interest_income",
    "dividend_income",
    "private_pension_income",
    "property_income",
    "maintenance_income",
    "miscellaneous_income",
]


@pytest.mark.parametrize("variable", INCOME_VARIABLES)
def test_income_non_negative(frs, variable: str):
    """Test that income variables have no negative values."""
    values = frs.person[variable]
    min_value = values.min()
    assert (
        min_value >= 0
    ), f"{variable} has negative values (min = {min_value:.2f})"
