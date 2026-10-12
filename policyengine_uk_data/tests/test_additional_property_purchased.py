"""Additional-property purchases are reproducible, stock capped and calibrated.

Synthetic counts tolerate 10% error and value fits have relative tolerance
1e-8. Saved enhanced FRS aggregates tolerate 20% after weight calibration.
Dataset assertions report only aggregate diagnostics, never household rows.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from policyengine_uk_data.datasets.frs_release import CURRENT_FRS_RELEASE
from policyengine_uk_data.datasets.imputations.property_purchases import (
    impute_property_purchase_values,
    impute_property_purchases,
)
from policyengine_uk_data.parameters import load_parameter


PROPERTY_TYPES = (
    (
        "additional_residential",
        "other_residential_property_value",
        "additional_residential_property_purchased",
    ),
    (
        "non_residential",
        "non_residential_property_value",
        "non_residential_property_purchased",
    ),
)
PROPERTY_TYPE_NAMES = tuple(item[0] for item in PROPERTY_TYPES)
RELEASE_YEARS = sorted(
    {CURRENT_FRS_RELEASE.base_year, CURRENT_FRS_RELEASE.calibration_year}
)
SYNTHETIC_YEARS = sorted({2023, *RELEASE_YEARS})


@pytest.fixture
def synthetic_households():
    """Heterogeneous owners, weights and nations with ample purchase capacity."""
    stock = np.geomspace(50_000.0, 50_000_000.0, 4_000)
    stock[::5] = 0.0
    stock[1::41] = 10_000.0
    weights = np.linspace(100.0, 500.0, len(stock))
    weights[::11] = 0.0
    sdlt_mask = np.arange(len(stock)) % 7 != 0
    return stock, weights, sdlt_mask


def _targets(property_type, year):
    return load_parameter("stochastic", "additional_property_purchases", year)[
        property_type
    ]


def _assert_aggregate_targets(
    prices, weights, sdlt_mask, targets, tolerance, value_tolerance=None
):
    value_tolerance = tolerance if value_tolerance is None else value_tolerance
    transactions = float(np.sum(weights[(prices > 0) & sdlt_mask]))
    value = float(np.dot(prices[sdlt_mask], weights[sdlt_mask]))
    assert transactions == pytest.approx(targets["transactions"], rel=tolerance), (
        f"Weighted transactions {transactions:,.0f}; "
        f"target {targets['transactions']:,.0f}; tolerance {tolerance:.0%}"
    )
    assert value == pytest.approx(targets["value"], rel=value_tolerance), (
        f"Weighted purchase value £{value / 1e9:.3f}bn; "
        f"target £{targets['value'] / 1e9:.3f}bn; "
        f"relative tolerance {value_tolerance:g}"
    )


@pytest.mark.unit
@pytest.mark.parametrize("property_type", PROPERTY_TYPE_NAMES)
def test_purchase_imputation_is_independent_of_global_rng(
    synthetic_households, property_type
):
    stock, weights, sdlt_mask = synthetic_households
    state = np.random.get_state()
    try:
        np.random.seed(1)
        first = impute_property_purchase_values(
            stock, weights, sdlt_mask, 2024, property_type
        )
        np.random.seed(999)
        np.random.random(10_000)
        second = impute_property_purchase_values(
            stock, weights, sdlt_mask, 2024, property_type
        )
    finally:
        np.random.set_state(state)
    np.testing.assert_array_equal(first, second)


@pytest.mark.unit
@pytest.mark.parametrize("property_type", PROPERTY_TYPE_NAMES)
@pytest.mark.parametrize("year", SYNTHETIC_YEARS)
def test_purchase_prices_and_weighted_targets(
    synthetic_households, property_type, year
):
    stock, weights, sdlt_mask = synthetic_households
    prices = impute_property_purchase_values(
        stock, weights, sdlt_mask, year, property_type
    )
    assert np.all(np.isfinite(prices))
    assert np.all(prices >= 0)
    assert np.all(prices <= stock)
    assert np.all(prices[(stock < 40_000) | (weights == 0)] == 0)
    assert np.all(prices[prices > 0] >= 40_000)
    assert np.any(prices > 0)
    assert np.any((prices > 0) & (prices < stock))
    _assert_aggregate_targets(
        prices,
        weights,
        sdlt_mask,
        _targets(property_type, year),
        tolerance=0.10,
        value_tolerance=1e-8,
    )


@pytest.mark.unit
def test_property_types_have_independent_purchase_draws(
    synthetic_households, monkeypatch
):
    from policyengine_uk_data.datasets.imputations import property_purchases

    original = load_parameter("stochastic", "additional_property_purchases", 2024)
    # Equal targets and price bands isolate the independence of event seeds.
    assumptions = {
        property_type: {
            **original["additional_residential"],
            "purchase_seed": original[property_type]["purchase_seed"],
            "price_seed": original[property_type]["price_seed"],
        }
        for property_type, _, _ in PROPERTY_TYPES
    }
    monkeypatch.setattr(property_purchases, "load_parameter", lambda *args: assumptions)
    stock, weights, sdlt_mask = synthetic_households
    residential = impute_property_purchase_values(
        stock, weights, sdlt_mask, 2024, "additional_residential"
    )
    non_residential = impute_property_purchase_values(
        stock, weights, sdlt_mask, 2024, "non_residential"
    )
    assert np.any((residential > 0) != (non_residential > 0))


@dataclass
class _HouseholdDataset:
    household: pd.DataFrame
    time_period: int = 2024

    def copy(self):
        return _HouseholdDataset(self.household.copy(), self.time_period)


@pytest.mark.unit
def test_purchase_inputs_do_not_depend_on_main_residence_purchase(
    synthetic_households,
):
    stock, weights, sdlt_mask = synthetic_households
    original = _HouseholdDataset(
        pd.DataFrame(
            {
                "region": np.where(sdlt_mask, "LONDON", "SCOTLAND"),
                "household_weight": weights,
                "other_residential_property_value": stock,
                "non_residential_property_value": stock,
                "property_purchased": False,
            }
        )
    )
    changed = original.copy()
    changed.household["property_purchased"] = True
    first = impute_property_purchases(original)
    second = impute_property_purchases(changed)
    for property_type, _, purchase_name in PROPERTY_TYPES:
        assert purchase_name not in original.household
        np.testing.assert_array_equal(
            first.household[purchase_name], second.household[purchase_name]
        )
        _assert_aggregate_targets(
            first.household[purchase_name].to_numpy(),
            weights,
            sdlt_mask,
            _targets(property_type, 2024),
            tolerance=0.10,
            value_tolerance=1e-8,
        )
    assert not original.household.property_purchased.any()
    assert second.household.property_purchased.all()


@pytest.mark.unit
@pytest.mark.parametrize("property_type", PROPERTY_TYPE_NAMES)
def test_insufficient_owner_population_fails(property_type):
    with pytest.raises(ValueError, match="Insufficient weighted property owners"):
        impute_property_purchase_values(
            np.full(100, 1_000_000.0),
            np.ones(100),
            np.ones(100, dtype=bool),
            2024,
            property_type,
        )


@pytest.mark.unit
@pytest.mark.parametrize("property_type", PROPERTY_TYPE_NAMES)
def test_stock_cap_prevents_infeasible_value_target(
    synthetic_households, property_type
):
    stock, weights, sdlt_mask = synthetic_households
    with pytest.raises(ValueError, match="stock cannot support"):
        impute_property_purchase_values(
            np.where(stock > 0, 40_000.0, 0.0),
            weights,
            sdlt_mask,
            2024,
            property_type,
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    "stock,weights",
    [
        (np.array([np.nan]), np.ones(1)),
        (np.ones(1), np.array([np.inf])),
        (np.array([-1.0]), np.ones(1)),
        (np.ones(1), np.array([-1.0])),
    ],
)
def test_invalid_stock_or_weights_fail(stock, weights):
    with pytest.raises(ValueError, match="finite and non-negative"):
        impute_property_purchase_values(
            stock, weights, np.ones(1, dtype=bool), 2024, "additional_residential"
        )


@pytest.mark.integration
@pytest.mark.parametrize("property_type,stock_name,purchase_name", PROPERTY_TYPES)
@pytest.mark.parametrize("year", RELEASE_YEARS)
def test_saved_enhanced_frs_purchase_inputs(
    enhanced_frs, baseline, property_type, stock_name, purchase_name, year
):
    """A rebuilt dataset retains the input and targets after calibration."""
    input_columns = set(enhanced_frs.household.columns)
    assert purchase_name in input_columns, (
        f"Enhanced FRS lacks {purchase_name}; rebuild with purchase imputation"
    )
    prices = baseline.calculate(purchase_name, year).values
    stock = baseline.calculate(stock_name, year).values
    weights = baseline.calculate("household_weight", year).values
    country = baseline.calculate("country", year, map_to="household").values
    sdlt_mask = np.isin(country, ["ENGLAND", "NORTHERN_IRELAND"])
    # Store only aggregate booleans in assertions: pytest otherwise includes
    # the household arrays in its rewritten assertion failure diagnostics.
    finite = bool(np.all(np.isfinite(prices)))
    non_negative = bool(np.all(prices >= 0))
    stock_capped = bool(np.all(prices <= stock + 0.01))
    owners_only = bool(np.all(prices[stock <= 0] == 0))
    assert finite, "Non-finite purchase inputs"
    assert non_negative, "Negative purchase prices"
    assert stock_capped, "Purchase prices exceed property stocks"
    assert owners_only, "Non-owners purchase property"
    _assert_aggregate_targets(
        prices, weights, sdlt_mask, _targets(property_type, year), tolerance=0.20
    )


@pytest.mark.integration
@pytest.mark.parametrize("year", RELEASE_YEARS)
def test_saved_enhanced_frs_higher_rates_sdlt(baseline, year):
    """Gross HRAD SDLT is non-zero and near the allocated household benchmark.

    The 30% tolerance allows price-distribution and calibration differences, and
    the annual model scale versus HMRC's fiscal-year rate changes and refunds.
    """
    prices = baseline.calculate(
        "additional_residential_property_purchased", year
    ).values
    weights = baseline.calculate("household_weight", year).values
    country = baseline.calculate("country", year, map_to="household").values
    sdlt_mask = np.isin(country, ["ENGLAND", "NORTHERN_IRELAND"])
    additional = baseline.tax_benefit_system.parameters(
        f"{year}-01-01"
    ).gov.hmrc.stamp_duty.residential.purchase.additional
    liability = np.where(prices >= additional.min, additional.rate.calc(prices), 0.0)
    revenue = float(np.dot(liability[sdlt_mask], weights[sdlt_mask]))
    target = _targets("additional_residential", year)["revenue"]
    assert revenue > 0, "Additional-dwelling purchases have no higher-rates SDLT"
    assert revenue == pytest.approx(target, rel=0.30), (
        f"Weighted higher-rates SDLT £{revenue / 1e9:.3f}bn; "
        f"household benchmark £{target / 1e9:.3f}bn; tolerance 30%"
    )
