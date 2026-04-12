"""Tests for ONS land value calibration targets."""

import pytest
from policyengine_uk_data.targets.sources._land import (
    CORPORATE_LAND_VALUES,
    HOUSEHOLD_LAND_VALUES,
    TOTAL_LAND_VALUES,
)

LAND_TARGETS = {
    "land_value": TOTAL_LAND_VALUES,
    "household_land_value": HOUSEHOLD_LAND_VALUES,
    "corporate_land_value": CORPORATE_LAND_VALUES,
}

# The target series is backfilled to 2021, but the enhanced 2023/24 simulation
# fixture is only a stable regression base from its dataset year onward.
# Keep the broader year coverage in the target-registry tests, and only run the
# simulation-vs-target aggregate check for years the fixture can represent.
MODEL_CHECK_YEARS = [2023, 2025]

TOLERANCES = {
    "land_value": 0.65,
    "household_land_value": 0.65,
    # Corporate land is not directly calibrated and currently drifts more
    # than household land as other admin targets move the weights.
    "corporate_land_value": 0.70,
}


@pytest.mark.parametrize("year", MODEL_CHECK_YEARS, ids=["2023", "2025"])
@pytest.mark.parametrize("variable", list(LAND_TARGETS), ids=list(LAND_TARGETS))
def test_land_value_aggregate(baseline, variable, year):
    """Check weighted aggregate land values against ONS targets."""
    target = LAND_TARGETS[variable][year]
    weights = baseline.calculate("household_weight", period=year).values
    values = baseline.calculate(variable, map_to="household", period=year).values
    estimate = (values * weights).sum()

    tolerance = TOLERANCES[variable]
    rel_error = abs(estimate / target - 1)
    assert rel_error < tolerance, (
        f"{variable}: expected £{target / 1e12:.2f}tn, "
        f"got £{estimate / 1e12:.2f}tn "
        f"(relative error = {rel_error:.1%}, tolerance = {tolerance:.0%})"
    )


def test_land_value_composition(baseline):
    """Household + corporate land should equal total land value."""
    year = 2025
    weights = baseline.calculate("household_weight", period=year).values
    total = baseline.calculate("land_value", map_to="household", period=year).values
    hh = baseline.calculate(
        "household_land_value", map_to="household", period=year
    ).values
    corp = baseline.calculate(
        "corporate_land_value", map_to="household", period=year
    ).values

    total_agg = (total * weights).sum()
    sum_agg = ((hh + corp) * weights).sum()

    assert abs(total_agg / sum_agg - 1) < 0.01, (
        f"Total land (£{total_agg / 1e12:.2f}tn) should equal "
        f"household + corporate (£{sum_agg / 1e12:.2f}tn)"
    )


def test_household_land_less_than_property_wealth(baseline):
    """Household land value should not exceed total property wealth."""
    year = 2025
    weights = baseline.calculate("household_weight", period=year).values
    hh_land = baseline.calculate(
        "household_land_value", map_to="household", period=year
    ).values
    prop = baseline.calculate("property_wealth", map_to="household", period=year).values

    hh_land_agg = (hh_land * weights).sum()
    prop_agg = (prop * weights).sum()

    assert hh_land_agg <= prop_agg * 1.05, (
        f"Household land (£{hh_land_agg / 1e12:.2f}tn) should not "
        f"exceed property wealth (£{prop_agg / 1e12:.2f}tn)"
    )
