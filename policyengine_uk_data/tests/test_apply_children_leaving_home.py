"""Tests for the leaving-home transition in household_transitions.py."""

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
    DEFAULT_LEAVING_HOME_RATES,
    apply_children_leaving_home,
)


def _family(
    *,
    parent_ages=(45, 43),
    child_ages=(19,),
    adult_child_on_separate_benunit: bool = False,
) -> UKSingleYearDataset:
    """Build a household with two parents and some (possibly adult) children.

    ``adult_child_on_separate_benunit`` models the FRS shape where
    a grown child living with parents has their own single benunit
    inside the parental household.
    """
    ages = [*parent_ages, *child_ages]
    sexes = [MALE_VALUE, FEMALE_VALUE] + [MALE_VALUE] * len(child_ages)
    n = len(ages)

    if adult_child_on_separate_benunit and child_ages:
        benunit_ids = [1, 1] + [2] * len(child_ages)
    else:
        benunit_ids = [1] * n

    person = pd.DataFrame(
        {
            "person_id": list(range(1, n + 1)),
            "person_benunit_id": benunit_ids,
            "person_household_id": [1] * n,
            AGE_COLUMN: list(ages),
            "gender": sexes,
            "employment_income": [0.0] * n,
        }
    )
    benunit_rows = sorted(set(benunit_ids))
    benunit = pd.DataFrame(
        {"benunit_id": benunit_rows, "benunit_weight": [1.0] * len(benunit_rows)}
    )
    household = pd.DataFrame(
        {"household_id": [1], "household_weight": [1.0], "region": ["LONDON"]}
    )
    return UKSingleYearDataset(
        person=person, benunit=benunit, household=household, fiscal_year=2023
    )


def test_rate_one_moves_every_eligible_adult_child():
    ds = _family(parent_ages=(45, 43), child_ages=(19, 22))
    rates = {a: 1.0 for a in range(120)}
    out = apply_children_leaving_home(
        ds, leaving_home_rates=rates, rng=np.random.default_rng(0)
    )
    # Parents stay; each child gets own benunit + household.
    adults_in_parents_bu = out.person[out.person["person_benunit_id"] == 1]
    assert sorted(adults_in_parents_bu[AGE_COLUMN].tolist()) == [43, 45]
    children_moved = out.person[out.person["person_benunit_id"] != 1]
    assert set(children_moved[AGE_COLUMN]) == {19, 22}
    # Each child in a distinct household.
    assert children_moved["person_household_id"].nunique() == 2


def test_rate_zero_leaves_everyone_in_place():
    ds = _family(parent_ages=(45, 43), child_ages=(19,))
    rates = {a: 0.0 for a in range(120)}
    out = apply_children_leaving_home(
        ds, leaving_home_rates=rates, rng=np.random.default_rng(0)
    )
    pd.testing.assert_frame_equal(
        out.person.sort_values("person_id").reset_index(drop=True),
        ds.person.sort_values("person_id").reset_index(drop=True),
    )


def test_young_children_are_not_moved():
    ds = _family(parent_ages=(45, 43), child_ages=(5, 10, 14))
    rates = {a: 1.0 for a in range(120)}
    out = apply_children_leaving_home(
        ds, leaving_home_rates=rates, rng=np.random.default_rng(0)
    )
    # Nothing changes: min_age default is 16.
    pd.testing.assert_frame_equal(out.person, ds.person)


def test_custom_min_age_threshold():
    ds = _family(parent_ages=(45, 43), child_ages=(17,))
    rates = {a: 1.0 for a in range(120)}
    # At min_age=18, the 17-year-old stays put.
    out = apply_children_leaving_home(
        ds, leaving_home_rates=rates, rng=np.random.default_rng(0), min_age=18
    )
    pd.testing.assert_frame_equal(out.person, ds.person)
    # At min_age=16, the 17-year-old moves.
    out2 = apply_children_leaving_home(
        ds, leaving_home_rates=rates, rng=np.random.default_rng(0), min_age=16
    )
    moved = out2.person[out2.person["person_benunit_id"] != 1]
    assert len(moved) == 1
    assert int(moved[AGE_COLUMN].iloc[0]) == 17


def test_adult_child_on_separate_benunit_also_moves():
    """FRS shape: adult child has their own benunit inside the parental household."""
    ds = _family(
        parent_ages=(45, 43), child_ages=(24,), adult_child_on_separate_benunit=True
    )
    rates = {a: 1.0 for a in range(120)}
    out = apply_children_leaving_home(
        ds, leaving_home_rates=rates, rng=np.random.default_rng(0)
    )
    # Adult child should now live in a different household from the parents.
    child = out.person[out.person[AGE_COLUMN] == 24].iloc[0]
    parents = out.person[out.person[AGE_COLUMN] >= 40]
    assert int(child["person_household_id"]) not in set(
        parents["person_household_id"].tolist()
    )


def test_couple_without_children_untouched():
    ds = _family(parent_ages=(45, 43), child_ages=())
    rates = {a: 1.0 for a in range(120)}
    out = apply_children_leaving_home(
        ds, leaving_home_rates=rates, rng=np.random.default_rng(0)
    )
    pd.testing.assert_frame_equal(out.person, ds.person)


def test_deterministic_under_same_seed():
    ds = _family(parent_ages=(45, 43), child_ages=(19, 22, 25))
    a = apply_children_leaving_home(
        ds, leaving_home_rates=DEFAULT_LEAVING_HOME_RATES, rng=np.random.default_rng(3)
    )
    b = apply_children_leaving_home(
        ds, leaving_home_rates=DEFAULT_LEAVING_HOME_RATES, rng=np.random.default_rng(3)
    )
    pd.testing.assert_frame_equal(a.person, b.person)
    pd.testing.assert_frame_equal(a.benunit, b.benunit)
    pd.testing.assert_frame_equal(a.household, b.household)


def test_new_household_inherits_region():
    ds = _family(parent_ages=(45, 43), child_ages=(19,))
    ds.household.loc[0, "region"] = "SCOTLAND"
    rates = {a: 1.0 for a in range(120)}
    out = apply_children_leaving_home(
        ds, leaving_home_rates=rates, rng=np.random.default_rng(0)
    )
    assert set(out.household["region"].tolist()) == {"SCOTLAND"}
