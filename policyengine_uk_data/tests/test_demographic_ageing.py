"""Tests for demographic ageing (step 3 of #345)."""

import pandas as pd
import pytest
from policyengine_uk.data import UKSingleYearDataset

from policyengine_uk_data.utils.demographic_ageing import (
    AGE_COLUMN,
    DEFAULT_FERTILITY_RATES_PLACEHOLDER,
    DEFAULT_MORTALITY_RATES_PLACEHOLDER,
    FEMALE_VALUE,
    MALE_VALUE,
    age_dataset,
)
from policyengine_uk_data.utils.panel_ids import classify_panel_ids


def _build_dataset(
    ages, genders, *, person_ids=None, benunit_ids=None, household_ids=None
) -> UKSingleYearDataset:
    """Build a minimal ``UKSingleYearDataset`` keyed on a list of ages."""
    n = len(ages)
    person_ids = list(person_ids or range(1001, 1001 + n))
    benunit_ids = list(benunit_ids or [101] * n)
    household_ids = list(household_ids or [1] * n)

    unique_benunits = sorted(set(benunit_ids))
    unique_households = sorted(set(household_ids))

    person = pd.DataFrame(
        {
            "person_id": person_ids,
            "person_benunit_id": benunit_ids,
            "person_household_id": household_ids,
            AGE_COLUMN: list(ages),
            "gender": list(genders),
            "employment_income": [0.0] * n,
        }
    )
    benunit = pd.DataFrame({"benunit_id": unique_benunits})
    household = pd.DataFrame(
        {
            "household_id": unique_households,
            "household_weight": [1.0] * len(unique_households),
        }
    )
    return UKSingleYearDataset(
        person=person, benunit=benunit, household=household, fiscal_year=2023
    )


# ---------------------------------------------------------------------------
# Identity / age-increment behaviour
# ---------------------------------------------------------------------------


def test_years_zero_returns_equivalent_dataset():
    base = _build_dataset(
        ages=[30, 35, 5], genders=[MALE_VALUE, FEMALE_VALUE, MALE_VALUE]
    )
    aged = age_dataset(base, years=0)
    pd.testing.assert_frame_equal(
        aged.person.reset_index(drop=True),
        base.person.reset_index(drop=True),
    )


def test_years_negative_raises():
    base = _build_dataset(ages=[30], genders=[MALE_VALUE])
    with pytest.raises(ValueError, match="non-negative"):
        age_dataset(base, years=-1)


def test_age_increments_when_no_mortality_or_fertility():
    base = _build_dataset(
        ages=[30, 35, 5], genders=[MALE_VALUE, FEMALE_VALUE, MALE_VALUE]
    )
    aged = age_dataset(base, years=5, mortality_rates={}, fertility_rates={})
    assert sorted(aged.person[AGE_COLUMN].tolist()) == [10, 35, 40]


def test_age_increment_preserves_row_count_under_no_demographics():
    base = _build_dataset(
        ages=[30, 40, 50], genders=[MALE_VALUE, FEMALE_VALUE, MALE_VALUE]
    )
    aged = age_dataset(base, years=3, mortality_rates={}, fertility_rates={})
    assert len(aged.person) == len(base.person)


def test_base_dataset_is_not_mutated():
    base = _build_dataset(ages=[30, 40], genders=[MALE_VALUE, FEMALE_VALUE])
    before_ages = base.person[AGE_COLUMN].tolist()
    age_dataset(base, years=10, mortality_rates={}, fertility_rates={})
    assert base.person[AGE_COLUMN].tolist() == before_ages


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_same_seed_produces_identical_output():
    base = _build_dataset(
        ages=list(range(20, 80)),
        genders=[FEMALE_VALUE if i % 2 else MALE_VALUE for i in range(60)],
    )
    a = age_dataset(base, years=5, seed=42)
    b = age_dataset(base, years=5, seed=42)
    pd.testing.assert_frame_equal(
        a.person.reset_index(drop=True),
        b.person.reset_index(drop=True),
    )


def test_different_seeds_produce_different_outputs():
    base = _build_dataset(
        ages=list(range(20, 80)),
        genders=[FEMALE_VALUE if i % 2 else MALE_VALUE for i in range(60)],
    )
    a = age_dataset(base, years=5, seed=1)
    b = age_dataset(base, years=5, seed=2)
    # Some randomness must show up — either survivor set or births differ.
    assert len(a.person) != len(b.person) or not a.person["person_id"].reset_index(
        drop=True
    ).equals(b.person["person_id"].reset_index(drop=True))


# ---------------------------------------------------------------------------
# Mortality
# ---------------------------------------------------------------------------


def test_mortality_rate_one_kills_everyone():
    base = _build_dataset(
        ages=[30, 40, 50], genders=[MALE_VALUE, FEMALE_VALUE, MALE_VALUE]
    )
    aged = age_dataset(
        base,
        years=1,
        mortality_rates={age: 1.0 for age in range(0, 121)},
        fertility_rates={},
    )
    assert len(aged.person) == 0


def test_mortality_rate_zero_keeps_everyone():
    base = _build_dataset(
        ages=[30, 40, 50], genders=[MALE_VALUE, FEMALE_VALUE, MALE_VALUE]
    )
    aged = age_dataset(
        base,
        years=10,
        mortality_rates={age: 0.0 for age in range(0, 121)},
        fertility_rates={},
    )
    assert len(aged.person) == 3


def test_mortality_reduces_population_with_placeholder_rates():
    # Large elderly cohort so placeholder mortality has a visible effect.
    base = _build_dataset(ages=[80] * 200, genders=[MALE_VALUE] * 200)
    aged = age_dataset(
        base,
        years=5,
        seed=0,
        mortality_rates=DEFAULT_MORTALITY_RATES_PLACEHOLDER,
        fertility_rates={},
    )
    assert len(aged.person) < len(base.person)


def test_mortality_only_affects_person_ids_subset_of_base():
    base = _build_dataset(ages=[70] * 50, genders=[MALE_VALUE] * 50)
    aged = age_dataset(
        base,
        years=5,
        seed=0,
        mortality_rates=DEFAULT_MORTALITY_RATES_PLACEHOLDER,
        fertility_rates={},
    )
    base_ids = set(base.person["person_id"].tolist())
    aged_ids = set(aged.person["person_id"].tolist())
    assert aged_ids <= base_ids


# ---------------------------------------------------------------------------
# Fertility
# ---------------------------------------------------------------------------


def test_fertility_rate_one_produces_one_birth_per_fertile_woman_per_year():
    # Three fertile women, one year, rate 1.0 → three births.
    base = _build_dataset(
        ages=[25, 30, 35, 50],
        genders=[FEMALE_VALUE, FEMALE_VALUE, FEMALE_VALUE, MALE_VALUE],
    )
    aged = age_dataset(
        base,
        years=1,
        mortality_rates={},
        fertility_rates={age: 1.0 for age in range(15, 50)},
    )
    # The man also ages but cannot give birth.
    assert len(aged.person) == 7


def test_no_fertility_rates_means_no_births():
    base = _build_dataset(
        ages=[25, 30, 35],
        genders=[FEMALE_VALUE, FEMALE_VALUE, FEMALE_VALUE],
    )
    aged = age_dataset(base, years=3, mortality_rates={}, fertility_rates={})
    assert len(aged.person) == len(base.person)


def test_fertility_does_not_affect_men():
    base = _build_dataset(
        ages=[25, 30, 35],
        genders=[MALE_VALUE, MALE_VALUE, MALE_VALUE],
    )
    aged = age_dataset(
        base,
        years=5,
        mortality_rates={},
        fertility_rates={age: 1.0 for age in range(15, 50)},
    )
    # No women → no babies.
    assert len(aged.person) == len(base.person)


def test_fertility_respects_reproductive_age_window():
    base = _build_dataset(ages=[10, 55], genders=[FEMALE_VALUE, FEMALE_VALUE])
    aged = age_dataset(
        base,
        years=1,
        mortality_rates={},
        fertility_rates={age: 1.0 for age in range(0, 121)},
    )
    # Neither woman is 15-49 this year.
    assert len(aged.person) == 2


def test_newborns_have_age_zero():
    base = _build_dataset(ages=[25], genders=[FEMALE_VALUE])
    aged = age_dataset(
        base,
        years=1,
        mortality_rates={},
        fertility_rates={age: 1.0 for age in range(15, 50)},
    )
    # Mother aged to 26; newborn created that year then aged 1 year at
    # end of step ⇒ newborn age 1 at end of the first aged year? No — the
    # implementation increments age AFTER births, so the newborn starts
    # this year at 0 and is 1 at the end. The mother started at 25 and is
    # 26. Expected ages: [26, 1].
    assert sorted(aged.person[AGE_COLUMN].tolist()) == [1, 26]


def test_newborns_have_unique_person_ids():
    base = _build_dataset(
        ages=[25] * 10,
        genders=[FEMALE_VALUE] * 10,
        person_ids=range(5000, 5010),
    )
    aged = age_dataset(
        base,
        years=3,
        seed=0,
        mortality_rates={},
        fertility_rates={age: 1.0 for age in range(15, 50)},
    )
    ids = aged.person["person_id"].tolist()
    assert len(ids) == len(set(ids)), "Newborn person_ids collided"


def test_newborns_inherit_mother_benunit_and_household():
    base = _build_dataset(
        ages=[30],
        genders=[FEMALE_VALUE],
        person_ids=[1001],
        benunit_ids=[777],
        household_ids=[42],
    )
    aged = age_dataset(
        base,
        years=1,
        mortality_rates={},
        fertility_rates={age: 1.0 for age in range(15, 50)},
    )
    newborns = aged.person[aged.person["person_id"] != 1001]
    assert newborns["person_benunit_id"].tolist() == [777]
    assert newborns["person_household_id"].tolist() == [42]


def test_newborns_have_valid_gender():
    base = _build_dataset(
        ages=[25] * 50,
        genders=[FEMALE_VALUE] * 50,
    )
    aged = age_dataset(
        base,
        years=1,
        seed=0,
        mortality_rates={},
        fertility_rates={age: 1.0 for age in range(15, 50)},
    )
    newborns = aged.person[~aged.person["person_id"].isin(base.person["person_id"])]
    assert set(newborns["gender"].unique()) <= {MALE_VALUE, FEMALE_VALUE}


# ---------------------------------------------------------------------------
# Combined and multi-year behaviour
# ---------------------------------------------------------------------------


def test_multi_year_age_compounds_correctly_without_demographics():
    base = _build_dataset(ages=[10, 40], genders=[MALE_VALUE, FEMALE_VALUE])
    aged = age_dataset(base, years=7, mortality_rates={}, fertility_rates={})
    assert sorted(aged.person[AGE_COLUMN].tolist()) == [17, 47]


def test_combined_mortality_and_fertility_run_cleanly():
    base = _build_dataset(
        ages=list(range(15, 80)),
        genders=[FEMALE_VALUE if i % 2 else MALE_VALUE for i in range(65)],
    )
    aged = age_dataset(
        base,
        years=5,
        seed=123,
        mortality_rates=DEFAULT_MORTALITY_RATES_PLACEHOLDER,
        fertility_rates=DEFAULT_FERTILITY_RATES_PLACEHOLDER,
    )
    # Smoke assertions: the run produced a valid dataset.
    assert AGE_COLUMN in aged.person.columns
    assert aged.person["person_id"].is_unique
    assert (aged.person[AGE_COLUMN] >= 0).all()


# ---------------------------------------------------------------------------
# Panel ID classification helper
# ---------------------------------------------------------------------------


def test_classify_panel_ids_identifies_survivors_deaths_and_births():
    base = _build_dataset(
        ages=[30, 40, 50],
        genders=[FEMALE_VALUE, FEMALE_VALUE, MALE_VALUE],
        person_ids=[1001, 1002, 1003],
    )

    # Force one death (id 1003) and one birth.
    aged = base.copy()
    aged.person = aged.person.loc[aged.person["person_id"] != 1003].copy()
    new_row = aged.person.iloc[0:1].copy()
    new_row["person_id"] = 9999
    new_row["age"] = 0
    aged.person = pd.concat([aged.person, new_row], ignore_index=True)
    transition = classify_panel_ids(base, aged)
    assert set(transition.survivors.tolist()) == {1001, 1002}
    assert set(transition.deaths.tolist()) == {1003}
    assert set(transition.births.tolist()) == {9999}
