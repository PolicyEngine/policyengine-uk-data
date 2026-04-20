"""Tests for the marriage transition in household_transitions.py."""

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
    DEFAULT_MARRIAGE_RATES,
    apply_marriages,
)


def _build_dataset(
    *,
    ages: list[int],
    sexes: list[str],
    benunit_ids: list[int],
    household_ids: list[int],
    region: str = "LONDON",
) -> UKSingleYearDataset:
    n = len(ages)
    person = pd.DataFrame(
        {
            "person_id": list(range(1, n + 1)),
            "person_benunit_id": benunit_ids,
            "person_household_id": household_ids,
            AGE_COLUMN: ages,
            "gender": sexes,
            "employment_income": [0.0] * n,
        }
    )
    benunit = pd.DataFrame(
        {
            "benunit_id": sorted(set(benunit_ids)),
            "benunit_weight": [1.0] * len(set(benunit_ids)),
        }
    )
    household = pd.DataFrame(
        {
            "household_id": sorted(set(household_ids)),
            "household_weight": [1.0] * len(set(household_ids)),
            "region": [region] * len(set(household_ids)),
        }
    )
    return UKSingleYearDataset(
        person=person, benunit=benunit, household=household, fiscal_year=2023
    )


def test_rate_one_marries_everyone_possible():
    # Three single men and three single women, each in their own benunit
    # and household. Every age has rate 1.0 → three pairs formed.
    ages = [30, 32, 34, 29, 31, 33]
    sexes = [MALE_VALUE] * 3 + [FEMALE_VALUE] * 3
    ds = _build_dataset(
        ages=ages,
        sexes=sexes,
        benunit_ids=list(range(1, 7)),
        household_ids=list(range(1, 7)),
    )
    rates = {
        MALE_VALUE: {a: 1.0 for a in range(120)},
        FEMALE_VALUE: {a: 1.0 for a in range(120)},
    }
    out = apply_marriages(ds, marriage_rates=rates, rng=np.random.default_rng(0))

    bu_adults = (
        out.person.assign(_adult=(out.person[AGE_COLUMN] >= 18).astype(int))
        .groupby("person_benunit_id")["_adult"]
        .sum()
    )
    # Exactly 3 benunits with 2 adults each (the new couples).
    assert (bu_adults == 2).sum() == 3


def test_rate_zero_changes_nothing():
    ages = [30, 29, 35, 33]
    sexes = [MALE_VALUE, FEMALE_VALUE, MALE_VALUE, FEMALE_VALUE]
    ds = _build_dataset(
        ages=ages,
        sexes=sexes,
        benunit_ids=[1, 2, 3, 4],
        household_ids=[1, 2, 3, 4],
    )
    rates = {MALE_VALUE: {}, FEMALE_VALUE: {}}
    out = apply_marriages(ds, marriage_rates=rates, rng=np.random.default_rng(0))
    pd.testing.assert_frame_equal(
        out.person.sort_values("person_id").reset_index(drop=True),
        ds.person.sort_values("person_id").reset_index(drop=True),
    )


def test_default_rates_used_when_none():
    """Passing ``None`` uses :data:`DEFAULT_MARRIAGE_RATES`.

    With ~3 % per-person marriage rate in the 25-35 cohort and 200
    singles of each sex, expect ~6 pairs per year on average — the
    test passes as long as at least one pair forms.
    """
    n = 200
    ages_m = [28 + (i % 8) for i in range(n)]
    ages_f = [27 + (i % 8) for i in range(n)]
    ds = _build_dataset(
        ages=ages_m + ages_f,
        sexes=[MALE_VALUE] * n + [FEMALE_VALUE] * n,
        benunit_ids=list(range(1, 2 * n + 1)),
        household_ids=list(range(1, 2 * n + 1)),
    )
    out = apply_marriages(ds, marriage_rates=None, rng=np.random.default_rng(0))
    assert len(out.benunit) < 2 * n, "Default rates should produce at least one pair"


def test_same_sex_pool_does_not_produce_a_pair():
    # Two men, no women: no pair possible even at rate 1.
    ds = _build_dataset(
        ages=[30, 33],
        sexes=[MALE_VALUE, MALE_VALUE],
        benunit_ids=[1, 2],
        household_ids=[1, 2],
    )
    rates = {
        MALE_VALUE: {a: 1.0 for a in range(120)},
        FEMALE_VALUE: {a: 1.0 for a in range(120)},
    }
    out = apply_marriages(ds, marriage_rates=rates, rng=np.random.default_rng(0))
    assert len(out.benunit) == 2  # no merge


def test_married_benunit_is_skipped():
    # A man + woman already in one benunit (two adults) plus an outside
    # single woman. The existing couple must NOT be touched, and rate=1
    # on the single woman should produce no pair (no eligible man).
    ages = [35, 34, 29]
    sexes = [MALE_VALUE, FEMALE_VALUE, FEMALE_VALUE]
    ds = _build_dataset(
        ages=ages,
        sexes=sexes,
        benunit_ids=[1, 1, 2],
        household_ids=[1, 1, 2],
    )
    rates = {
        MALE_VALUE: {a: 1.0 for a in range(120)},
        FEMALE_VALUE: {a: 1.0 for a in range(120)},
    }
    out = apply_marriages(ds, marriage_rates=rates, rng=np.random.default_rng(0))
    # The couple benunit still has 2 adults; the single woman's benunit
    # still has 1 adult.
    adult_counts = (
        out.person.assign(_a=(out.person[AGE_COLUMN] >= 18).astype(int))
        .groupby("person_benunit_id")["_a"]
        .sum()
    )
    assert set(adult_counts.values) == {1, 2}


def test_region_preference_keeps_pool_within_region():
    # One man in London, one woman in Scotland. Even at rate 1, they
    # should not marry across regions because matching is in-region.
    person = pd.DataFrame(
        {
            "person_id": [1, 2],
            "person_benunit_id": [10, 20],
            "person_household_id": [100, 200],
            AGE_COLUMN: [30, 29],
            "gender": [MALE_VALUE, FEMALE_VALUE],
            "employment_income": [0.0, 0.0],
        }
    )
    benunit = pd.DataFrame({"benunit_id": [10, 20], "benunit_weight": [1.0, 1.0]})
    household = pd.DataFrame(
        {
            "household_id": [100, 200],
            "household_weight": [1.0, 1.0],
            "region": ["LONDON", "SCOTLAND"],
        }
    )
    ds = UKSingleYearDataset(
        person=person, benunit=benunit, household=household, fiscal_year=2023
    )
    rates = {
        MALE_VALUE: {a: 1.0 for a in range(120)},
        FEMALE_VALUE: {a: 1.0 for a in range(120)},
    }
    out = apply_marriages(ds, marriage_rates=rates, rng=np.random.default_rng(0))
    # Still two separate benunits, no marriage.
    assert len(out.benunit) == 2


def test_matched_partner_has_closest_available_age():
    # One man aged 30; two women aged 22 and 29. The 29-year-old should
    # be matched (closer age).
    ages = [30, 22, 29]
    sexes = [MALE_VALUE, FEMALE_VALUE, FEMALE_VALUE]
    ds = _build_dataset(
        ages=ages,
        sexes=sexes,
        benunit_ids=[1, 2, 3],
        household_ids=[1, 2, 3],
    )
    rates = {
        MALE_VALUE: {a: 1.0 for a in range(120)},
        FEMALE_VALUE: {a: 1.0 for a in range(120)},
    }
    out = apply_marriages(ds, marriage_rates=rates, rng=np.random.default_rng(0))
    # Find the benunit that now has 2 adults, inspect the ages.
    adult_counts = (
        out.person.assign(_a=(out.person[AGE_COLUMN] >= 18).astype(int))
        .groupby("person_benunit_id")["_a"]
        .sum()
    )
    couple_bu = adult_counts[adult_counts == 2].index[0]
    couple_ages = sorted(
        out.person[out.person["person_benunit_id"] == couple_bu][AGE_COLUMN].tolist()
    )
    assert couple_ages == [29, 30]


def test_benunit_weights_preserved_on_merge():
    # Two singles, each benunit weight 1.0. After merge, remaining
    # benunit weight should be 2.0 (summed).
    ds = _build_dataset(
        ages=[30, 29],
        sexes=[MALE_VALUE, FEMALE_VALUE],
        benunit_ids=[1, 2],
        household_ids=[1, 2],
    )
    ds.benunit["benunit_weight"] = [1.0, 1.0]
    rates = {
        MALE_VALUE: {a: 1.0 for a in range(120)},
        FEMALE_VALUE: {a: 1.0 for a in range(120)},
    }
    out = apply_marriages(ds, marriage_rates=rates, rng=np.random.default_rng(0))
    assert len(out.benunit) == 1
    assert float(out.benunit["benunit_weight"].iloc[0]) == 2.0


def test_deterministic_under_same_seed():
    ds = _build_dataset(
        ages=[30, 29, 35, 36],
        sexes=[MALE_VALUE, FEMALE_VALUE, MALE_VALUE, FEMALE_VALUE],
        benunit_ids=[1, 2, 3, 4],
        household_ids=[1, 2, 3, 4],
    )
    a = apply_marriages(
        ds, marriage_rates=DEFAULT_MARRIAGE_RATES, rng=np.random.default_rng(42)
    )
    b = apply_marriages(
        ds, marriage_rates=DEFAULT_MARRIAGE_RATES, rng=np.random.default_rng(42)
    )
    pd.testing.assert_frame_equal(a.person, b.person)
    pd.testing.assert_frame_equal(a.benunit, b.benunit)
    pd.testing.assert_frame_equal(a.household, b.household)
