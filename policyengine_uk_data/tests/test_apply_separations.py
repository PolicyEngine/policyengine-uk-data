"""Tests for the separation transition in household_transitions.py."""

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
    DEFAULT_SEPARATION_RATES,
    apply_separations,
)


def _couple_with_children(
    *,
    male_age: int = 40,
    female_age: int = 38,
    child_ages: list[int] | None = None,
) -> UKSingleYearDataset:
    child_ages = child_ages or []
    ages = [male_age, female_age, *child_ages]
    sexes = [MALE_VALUE, FEMALE_VALUE] + [MALE_VALUE] * len(child_ages)
    n = len(ages)
    person = pd.DataFrame(
        {
            "person_id": list(range(1, n + 1)),
            "person_benunit_id": [1] * n,  # one couple benunit with kids
            "person_household_id": [1] * n,
            AGE_COLUMN: ages,
            "gender": sexes,
            "employment_income": [0.0] * n,
        }
    )
    benunit = pd.DataFrame({"benunit_id": [1], "benunit_weight": [1.0]})
    household = pd.DataFrame(
        {"household_id": [1], "household_weight": [1.0], "region": ["LONDON"]}
    )
    return UKSingleYearDataset(
        person=person, benunit=benunit, household=household, fiscal_year=2023
    )


def test_rate_one_splits_every_couple():
    ds = _couple_with_children(male_age=40, female_age=38)
    rates = {a: 1.0 for a in range(120)}
    out = apply_separations(ds, separation_rates=rates, rng=np.random.default_rng(0))
    # Two benunits now (one per adult).
    assert len(out.benunit) == 2
    # Each benunit has 1 adult.
    adults = out.person[out.person[AGE_COLUMN] >= 18]
    assert adults.groupby("person_benunit_id").size().max() == 1


def test_rate_zero_keeps_couple_intact():
    ds = _couple_with_children(male_age=40, female_age=38)
    rates = {a: 0.0 for a in range(120)}
    out = apply_separations(ds, separation_rates=rates, rng=np.random.default_rng(0))
    pd.testing.assert_frame_equal(
        out.person.sort_values("person_id").reset_index(drop=True),
        ds.person.sort_values("person_id").reset_index(drop=True),
    )


def test_children_default_to_mother():
    ds = _couple_with_children(male_age=40, female_age=38, child_ages=[3, 8])
    rates = {a: 1.0 for a in range(120)}
    out = apply_separations(ds, separation_rates=rates, rng=np.random.default_rng(0))
    # Find the female's benunit and verify the children are with her.
    female_row = out.person[out.person["gender"] == FEMALE_VALUE].iloc[0]
    female_bu = int(female_row["person_benunit_id"])
    children = out.person[out.person[AGE_COLUMN] < 18]
    assert set(children["person_benunit_id"]) == {female_bu}
    # Male is in a different benunit and household.
    male_row = out.person[out.person["gender"] == MALE_VALUE].iloc[0]
    assert int(male_row["person_benunit_id"]) != female_bu
    assert int(male_row["person_household_id"]) != int(
        female_row["person_household_id"]
    )


def test_children_stay_with_male_when_overridden():
    ds = _couple_with_children(male_age=40, female_age=38, child_ages=[5])
    rates = {a: 1.0 for a in range(120)}
    out = apply_separations(
        ds,
        separation_rates=rates,
        rng=np.random.default_rng(0),
        children_stay_with_female=False,
    )
    male_row = out.person[out.person["gender"] == MALE_VALUE].iloc[0]
    child = out.person[out.person[AGE_COLUMN] < 18].iloc[0]
    assert int(child["person_benunit_id"]) == int(male_row["person_benunit_id"])


def test_new_benunit_and_household_rows_added():
    ds = _couple_with_children(male_age=40, female_age=38)
    rates = {a: 1.0 for a in range(120)}
    out = apply_separations(ds, separation_rates=rates, rng=np.random.default_rng(0))
    assert len(out.benunit) == 2
    assert len(out.household) == 2
    assert set(out.household["region"].tolist()) == {"LONDON"}


def test_singles_are_not_touched():
    # A man alone in benunit 1, a woman alone in benunit 2; no couple.
    person = pd.DataFrame(
        {
            "person_id": [1, 2],
            "person_benunit_id": [1, 2],
            "person_household_id": [1, 2],
            AGE_COLUMN: [40, 38],
            "gender": [MALE_VALUE, FEMALE_VALUE],
            "employment_income": [0.0, 0.0],
        }
    )
    benunit = pd.DataFrame({"benunit_id": [1, 2], "benunit_weight": [1.0, 1.0]})
    household = pd.DataFrame(
        {
            "household_id": [1, 2],
            "household_weight": [1.0, 1.0],
            "region": ["LONDON", "LONDON"],
        }
    )
    ds = UKSingleYearDataset(
        person=person, benunit=benunit, household=household, fiscal_year=2023
    )
    rates = {a: 1.0 for a in range(120)}
    out = apply_separations(ds, separation_rates=rates, rng=np.random.default_rng(0))
    pd.testing.assert_frame_equal(out.person, ds.person)


def test_deterministic_under_same_seed():
    ds = _couple_with_children(male_age=40, female_age=38, child_ages=[3])
    a = apply_separations(
        ds, separation_rates=DEFAULT_SEPARATION_RATES, rng=np.random.default_rng(7)
    )
    b = apply_separations(
        ds, separation_rates=DEFAULT_SEPARATION_RATES, rng=np.random.default_rng(7)
    )
    pd.testing.assert_frame_equal(a.person, b.person)
    pd.testing.assert_frame_equal(a.benunit, b.benunit)
    pd.testing.assert_frame_equal(a.household, b.household)


def test_default_rates_produce_some_splits_over_many_couples():
    # 500 couples, default ONS rates — expect a few splits per run.
    n_couples = 500
    ages_m = [32] * n_couples
    ages_f = [30] * n_couples
    bu_ids = list(range(1, n_couples + 1))
    hh_ids = list(range(1, n_couples + 1))
    person = pd.DataFrame(
        {
            "person_id": list(range(1, 2 * n_couples + 1)),
            "person_benunit_id": bu_ids + bu_ids,
            "person_household_id": hh_ids + hh_ids,
            AGE_COLUMN: ages_m + ages_f,
            "gender": [MALE_VALUE] * n_couples + [FEMALE_VALUE] * n_couples,
            "employment_income": [0.0] * (2 * n_couples),
        }
    )
    benunit = pd.DataFrame({"benunit_id": bu_ids, "benunit_weight": [1.0] * n_couples})
    household = pd.DataFrame(
        {
            "household_id": hh_ids,
            "household_weight": [1.0] * n_couples,
            "region": ["LONDON"] * n_couples,
        }
    )
    ds = UKSingleYearDataset(
        person=person, benunit=benunit, household=household, fiscal_year=2023
    )
    out = apply_separations(ds, separation_rates=None, rng=np.random.default_rng(0))
    # 2 % rate × 500 couples → ~10 splits expected.
    assert len(out.benunit) > n_couples
