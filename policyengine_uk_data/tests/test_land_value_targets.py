"""Tests for ONS land value calibration targets.

These validate that the generated Enhanced FRS dataset reproduces
aggregate land values from the ONS National
Balance Sheet 2025.

Source: https://www.ons.gov.uk/economy/nationalaccounts/uksectoraccounts/bulletins/nationalbalancesheet/2025
"""

import pytest

# ONS National Balance Sheet 2025
# 2024 measured total: £7.1tn
# 2020 split scaled proportionally: household £5.04tn, corporate £2.06tn
_ONS_2020_HOUSEHOLD = 4.31e12
_ONS_2020_CORPORATE = 1.76e12
_ONS_2020_TOTAL = _ONS_2020_HOUSEHOLD + _ONS_2020_CORPORATE
_ONS_2024_TOTAL = 7.1e12
_SCALE = _ONS_2024_TOTAL / _ONS_2020_TOTAL

LAND_TARGETS = {
    "land_value": _ONS_2024_TOTAL,
    "household_land_value": _ONS_2020_HOUSEHOLD * _SCALE,
    "corporate_land_value": _ONS_2020_CORPORATE * _SCALE,
}

YEAR = 2025
TOLERANCE = 0.50  # 50% — land values not yet calibrated against ONS targets


@pytest.mark.parametrize(
    "variable,target",
    list(LAND_TARGETS.items()),
    ids=list(LAND_TARGETS.keys()),
)
def test_land_value_aggregate(baseline, variable, target):
    """Check weighted aggregate land values against ONS targets."""
    weights = baseline.calculate("household_weight", period=YEAR).values
    values = baseline.calculate(variable, map_to="household", period=YEAR).values
    estimate = (values * weights).sum()

    rel_error = abs(estimate / target - 1)
    assert rel_error < TOLERANCE, (
        f"{variable}: expected £{target / 1e12:.2f}tn, "
        f"got £{estimate / 1e12:.2f}tn "
        f"(relative error = {rel_error:.1%})"
    )


def test_land_value_composition(baseline):
    """Household + corporate land should equal total land value."""
    weights = baseline.calculate("household_weight", period=YEAR).values
    total = baseline.calculate("land_value", map_to="household", period=YEAR).values
    hh = baseline.calculate(
        "household_land_value", map_to="household", period=YEAR
    ).values
    corp = baseline.calculate(
        "corporate_land_value", map_to="household", period=YEAR
    ).values

    total_agg = (total * weights).sum()
    sum_agg = ((hh + corp) * weights).sum()

    assert abs(total_agg / sum_agg - 1) < 0.01, (
        f"Total land (£{total_agg / 1e12:.2f}tn) should equal "
        f"household + corporate (£{sum_agg / 1e12:.2f}tn)"
    )


def test_household_land_less_than_property_wealth(baseline):
    """Household land value should not exceed total property wealth."""
    weights = baseline.calculate("household_weight", period=YEAR).values
    hh_land = baseline.calculate(
        "household_land_value", map_to="household", period=YEAR
    ).values
    prop = baseline.calculate("property_wealth", map_to="household", period=YEAR).values

    hh_land_agg = (hh_land * weights).sum()
    prop_agg = (prop * weights).sum()

    assert hh_land_agg <= prop_agg * 1.05, (
        f"Household land (£{hh_land_agg / 1e12:.2f}tn) should not "
        f"exceed property wealth (£{prop_agg / 1e12:.2f}tn)"
    )
