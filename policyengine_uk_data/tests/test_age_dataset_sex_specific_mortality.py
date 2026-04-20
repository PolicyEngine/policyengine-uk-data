"""Tests for sex-specific mortality rates in ``age_dataset``.

Covers the new ``Mapping[str, Mapping[int, float]]`` accepted shape so
that callers can pass ONS life-table output directly without having to
collapse males and females into one rate.
"""

from __future__ import annotations

import pandas as pd
from policyengine_uk.data import UKSingleYearDataset

from policyengine_uk_data.utils.demographic_ageing import (
    AGE_COLUMN,
    FEMALE_VALUE,
    MALE_VALUE,
    age_dataset,
)


def _population(n_male: int, n_female: int, age: int = 70) -> UKSingleYearDataset:
    """Build a flat dataset with only elderly men and women at one age."""
    ages = [age] * (n_male + n_female)
    sexes = [MALE_VALUE] * n_male + [FEMALE_VALUE] * n_female
    n = len(ages)
    person = pd.DataFrame(
        {
            "person_id": list(range(1, n + 1)),
            "person_benunit_id": list(range(1, n + 1)),
            "person_household_id": list(range(1, n + 1)),
            AGE_COLUMN: ages,
            "gender": sexes,
            "employment_income": [0.0] * n,
        }
    )
    benunit = pd.DataFrame({"benunit_id": list(range(1, n + 1))})
    household = pd.DataFrame(
        {
            "household_id": list(range(1, n + 1)),
            "household_weight": [1.0] * n,
        }
    )
    return UKSingleYearDataset(
        person=person, benunit=benunit, household=household, fiscal_year=2023
    )


def test_age_only_rates_still_work():
    """Backward compat: a plain ``Mapping[int, float]`` must still apply."""
    base = _population(n_male=500, n_female=500, age=70)
    rates = {70: 1.0, 71: 1.0}  # everyone dies
    aged = age_dataset(
        base,
        years=1,
        seed=0,
        mortality_rates=rates,
        fertility_rates={},
    )
    assert len(aged.person) == 0


def test_sex_specific_rates_kill_males_spare_females():
    base = _population(n_male=1_000, n_female=1_000, age=70)
    rates = {
        MALE_VALUE: {70: 1.0},  # every male aged 70 dies
        FEMALE_VALUE: {70: 0.0},  # every female aged 70 survives
    }
    aged = age_dataset(
        base,
        years=1,
        seed=0,
        mortality_rates=rates,
        fertility_rates={},
    )
    assert (aged.person["gender"] == MALE_VALUE).sum() == 0
    assert (aged.person["gender"] == FEMALE_VALUE).sum() == 1_000


def test_sex_specific_rates_without_sex_column_defaults_to_male():
    """Belt-and-braces guard for future entities where sex may not exist."""
    base = _population(n_male=500, n_female=500, age=70)
    base.person = base.person.drop(columns=["gender"])
    rates = {MALE_VALUE: {70: 1.0}, FEMALE_VALUE: {70: 0.0}}
    aged = age_dataset(
        base,
        years=1,
        seed=0,
        mortality_rates=rates,
        fertility_rates={},
    )
    # With every person treated as male and male rate = 1.0, everyone dies.
    assert len(aged.person) == 0


def test_missing_age_in_sex_block_defaults_to_zero():
    """Ages absent from the table must not raise — treated as 0 probability."""
    base = _population(n_male=100, n_female=100, age=42)
    rates = {MALE_VALUE: {70: 1.0}, FEMALE_VALUE: {70: 1.0}}
    aged = age_dataset(
        base,
        years=1,
        seed=0,
        mortality_rates=rates,
        fertility_rates={},
    )
    # Nobody at age 42 in the rate table → nobody dies.
    assert len(aged.person) == 200


def test_real_ons_rates_apply_sensibly_to_toy_pop(tmp_path):
    """Smoke-test that real ONS rates plug in without shape errors.

    Uses the synthetic workbook from ``test_ons_mortality`` so CI stays
    hermetic; the asserted behaviour (lower female death rate than male
    at every tested age) mirrors the real UK pattern and holds in the
    fixture too.
    """
    from policyengine_uk_data.targets.sources.ons_mortality import (
        get_mortality_rates,
        load_ons_life_tables,
    )
    from policyengine_uk_data.tests.test_ons_mortality import _synthetic_workbook

    load_ons_life_tables.cache_clear()
    path = _synthetic_workbook(tmp_path)
    tables = load_ons_life_tables(path=str(path))
    rates = get_mortality_rates(2023, tables=tables)

    # In both the real ONS data and the fixture, female qx < male qx at
    # every age. Use a large toy population at a single age and confirm
    # females have more survivors than males.
    base = _population(n_male=10_000, n_female=10_000, age=5)
    # Scale up the rates so the Monte-Carlo separation is clearly visible
    # in a small sample; the shape (male > female) is preserved.
    scaled = {
        sex: {age: min(1.0, qx * 50) for age, qx in ages.items()}
        for sex, ages in rates.items()
    }
    aged = age_dataset(
        base,
        years=1,
        seed=7,
        mortality_rates=scaled,
        fertility_rates={},
    )
    male_survivors = int((aged.person["gender"] == MALE_VALUE).sum())
    female_survivors = int((aged.person["gender"] == FEMALE_VALUE).sum())
    assert female_survivors > male_survivors
