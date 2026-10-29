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
    "income_not_otherwise_charged",
]


@pytest.mark.parametrize("variable", INCOME_VARIABLES)
def test_income_non_negative(frs, variable: str):
    """Test that income variables have no negative values.

    The test counts the negative values rather than reporting the smallest, so
    a failure prints no single record's amount.
    """
    negative = int((frs.person[variable] < 0).sum())
    assert negative == 0, f"{variable} has {negative} negative values"
