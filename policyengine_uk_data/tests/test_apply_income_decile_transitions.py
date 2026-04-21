"""Tests for apply_income_decile_transitions."""

from __future__ import annotations

import numpy as np
import pandas as pd
from policyengine_uk.data import UKSingleYearDataset

from policyengine_uk_data.utils.demographic_ageing import (
    AGE_COLUMN,
    FEMALE_VALUE,
    MALE_VALUE,
)
from policyengine_uk_data.utils.household_transitions import (
    apply_income_decile_transitions,
)


def _pop_with_income(n: int, income_min: float, income_max: float):
    ages = [35] * n
    sexes = [MALE_VALUE] * n
    incomes = np.linspace(income_min, income_max, n).tolist()
    person = pd.DataFrame(
        {
            "person_id": list(range(1, n + 1)),
            "person_benunit_id": list(range(1, n + 1)),
            "person_household_id": list(range(1, n + 1)),
            AGE_COLUMN: ages,
            "gender": sexes,
            "employment_income": incomes,
        }
    )
    benunit = pd.DataFrame(
        {"benunit_id": list(range(1, n + 1)), "benunit_weight": [1.0] * n}
    )
    household = pd.DataFrame(
        {
            "household_id": list(range(1, n + 1)),
            "household_weight": [1.0] * n,
            "region": ["LONDON"] * n,
        }
    )
    return UKSingleYearDataset(
        person=person, benunit=benunit, household=household, fiscal_year=2023
    )


def test_no_rates_is_no_op():
    ds = _pop_with_income(100, 10_000, 100_000)
    out = apply_income_decile_transitions(
        ds, decile_rates={}, rng=np.random.default_rng(0)
    )
    pd.testing.assert_series_equal(
        out.person["employment_income"], ds.person["employment_income"]
    )


def test_stay_in_same_decile_leaves_income_close():
    ds = _pop_with_income(100, 10_000, 100_000)
    # Everyone stays in their current decile with probability 1.
    rates = {("35-39", "MALE", d): {d: 1.0} for d in range(1, 11)}
    out = apply_income_decile_transitions(
        ds, decile_rates=rates, rng=np.random.default_rng(0)
    )
    # No one moves, so incomes should be identical.
    pd.testing.assert_series_equal(
        out.person["employment_income"], ds.person["employment_income"]
    )


def test_move_from_bottom_to_top_lifts_incomes():
    ds = _pop_with_income(200, 10_000, 200_000)
    # Everyone in decile 1 → decile 10 with probability 1, others stay.
    rates = {("35-39", "MALE", 1): {10: 1.0}}
    for d in range(2, 11):
        rates[("35-39", "MALE", d)] = {d: 1.0}
    out = apply_income_decile_transitions(
        ds, decile_rates=rates, rng=np.random.default_rng(0)
    )
    # Bottom 10% should now have incomes close to the original top-decile
    # median (~190k), much higher than their starting incomes.
    bottom_idx = ds.person["employment_income"].rank() <= 20
    lifted = out.person.loc[bottom_idx, "employment_income"]
    assert lifted.min() > 100_000


def test_probabilities_normalise_if_not_already():
    ds = _pop_with_income(100, 10_000, 100_000)
    # Deliberately unnormalised (sum to 2): the function should still run.
    rates = {("35-39", "MALE", 1): {1: 1.0, 2: 1.0}}
    for d in range(2, 11):
        rates[("35-39", "MALE", d)] = {d: 1.0}
    out = apply_income_decile_transitions(
        ds, decile_rates=rates, rng=np.random.default_rng(0)
    )
    assert len(out.person) == 100


def test_missing_cell_passes_through():
    ds = _pop_with_income(50, 10_000, 50_000)
    # Rates exist only for women, but our pop is men.
    rates = {("35-39", "FEMALE", 5): {6: 1.0}}
    out = apply_income_decile_transitions(
        ds, decile_rates=rates, rng=np.random.default_rng(0)
    )
    pd.testing.assert_series_equal(
        out.person["employment_income"], ds.person["employment_income"]
    )


def test_under_min_age_untouched():
    ds = _pop_with_income(50, 10_000, 50_000)
    ds.person[AGE_COLUMN] = 15
    rates = {("35-39", "MALE", 1): {10: 1.0}}
    out = apply_income_decile_transitions(
        ds,
        decile_rates=rates,
        rng=np.random.default_rng(0),
        min_age=18,
    )
    pd.testing.assert_series_equal(
        out.person["employment_income"], ds.person["employment_income"]
    )


def test_deterministic_under_same_seed():
    ds = _pop_with_income(100, 10_000, 100_000)
    rates = {
        ("35-39", "MALE", 1): {5: 1.0},
        ("35-39", "MALE", 10): {1: 1.0},
    }
    for d in range(2, 10):
        rates[("35-39", "MALE", d)] = {d: 1.0}
    a = apply_income_decile_transitions(
        ds, decile_rates=rates, rng=np.random.default_rng(3)
    )
    b = apply_income_decile_transitions(
        ds, decile_rates=rates, rng=np.random.default_rng(3)
    )
    pd.testing.assert_frame_equal(a.person, b.person)
