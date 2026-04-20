"""Tests for the migration transition in household_transitions.py."""

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
    DEFAULT_NET_MIGRATION_RATES,
    apply_migration,
)


def _population(n: int, age: int = 22, region: str = "LONDON") -> UKSingleYearDataset:
    person = pd.DataFrame(
        {
            "person_id": list(range(1, n + 1)),
            "person_benunit_id": list(range(1, n + 1)),
            "person_household_id": list(range(1, n + 1)),
            AGE_COLUMN: [age] * n,
            "gender": [MALE_VALUE if i % 2 == 0 else FEMALE_VALUE for i in range(n)],
            "employment_income": [0.0] * n,
        }
    )
    benunit = pd.DataFrame(
        {"benunit_id": list(range(1, n + 1)), "benunit_weight": [1.0] * n}
    )
    household = pd.DataFrame(
        {
            "household_id": list(range(1, n + 1)),
            "household_weight": [1.0] * n,
            "region": [region] * n,
        }
    )
    return UKSingleYearDataset(
        person=person, benunit=benunit, household=household, fiscal_year=2023
    )


def test_large_positive_rate_grows_population():
    ds = _population(n=500, age=22)
    # Rate 0.5 → expected 250 immigrants at age 22.
    rates = {22: 0.5}
    out = apply_migration(ds, net_migration_rates=rates, rng=np.random.default_rng(0))
    assert len(out.person) > len(ds.person)
    # New IDs are contiguous with the original max.
    assert int(out.person["person_id"].max()) > int(ds.person["person_id"].max())


def test_large_negative_rate_shrinks_population():
    ds = _population(n=500, age=22)
    rates = {22: -0.3}
    out = apply_migration(ds, net_migration_rates=rates, rng=np.random.default_rng(0))
    assert len(out.person) < len(ds.person)


def test_zero_rate_is_noop():
    ds = _population(n=100, age=22)
    rates = {a: 0.0 for a in range(120)}
    out = apply_migration(ds, net_migration_rates=rates, rng=np.random.default_rng(0))
    pd.testing.assert_frame_equal(
        out.person.sort_values("person_id").reset_index(drop=True),
        ds.person.sort_values("person_id").reset_index(drop=True),
    )


def test_empty_rates_is_noop():
    ds = _population(n=100, age=22)
    out = apply_migration(ds, net_migration_rates={}, rng=np.random.default_rng(0))
    pd.testing.assert_frame_equal(
        out.person.sort_values("person_id").reset_index(drop=True),
        ds.person.sort_values("person_id").reset_index(drop=True),
    )


def test_emigrants_leave_no_orphaned_benunits_or_households():
    ds = _population(n=200, age=22)
    rates = {22: -0.5}
    out = apply_migration(ds, net_migration_rates=rates, rng=np.random.default_rng(0))
    # Every remaining benunit / household has at least one person.
    assert set(out.benunit["benunit_id"]) == set(out.person["person_benunit_id"])
    assert set(out.household["household_id"]) == set(out.person["person_household_id"])


def test_immigrants_get_fresh_ids_and_bu_hh_rows():
    ds = _population(n=200, age=22)
    rates = {22: 0.5}
    max_before = (
        int(ds.person["person_id"].max()),
        int(ds.person["person_benunit_id"].max()),
        int(ds.person["person_household_id"].max()),
    )
    out = apply_migration(ds, net_migration_rates=rates, rng=np.random.default_rng(0))
    assert int(out.person["person_id"].max()) > max_before[0]
    assert int(out.person["person_benunit_id"].max()) > max_before[1]
    assert int(out.person["person_household_id"].max()) > max_before[2]
    # Every new person has a matching benunit and household row.
    assert set(out.benunit["benunit_id"]).issuperset(
        set(out.person["person_benunit_id"])
    )
    assert set(out.household["household_id"]).issuperset(
        set(out.person["person_household_id"])
    )


def test_deterministic_under_same_seed():
    ds = _population(n=200, age=22)
    rates = {22: 0.1}
    a = apply_migration(ds, net_migration_rates=rates, rng=np.random.default_rng(9))
    b = apply_migration(ds, net_migration_rates=rates, rng=np.random.default_rng(9))
    pd.testing.assert_frame_equal(a.person, b.person)
    pd.testing.assert_frame_equal(a.benunit, b.benunit)
    pd.testing.assert_frame_equal(a.household, b.household)


def test_immigrant_inherits_donor_region():
    # Two subpopulations in different regions; migration draws from
    # donors in the same age cohort, so regions should be preserved.
    ds = _population(n=200, age=22, region="LONDON")
    rates = {22: 0.2}
    out = apply_migration(ds, net_migration_rates=rates, rng=np.random.default_rng(0))
    assert set(out.household["region"].unique()) == {"LONDON"}


def test_default_rates_only_touch_working_ages():
    # Build two cohorts: one at age 5 (rate ~0.001), one at 22 (rate ~0.012).
    person = pd.DataFrame(
        {
            "person_id": list(range(1, 2001)),
            "person_benunit_id": list(range(1, 2001)),
            "person_household_id": list(range(1, 2001)),
            AGE_COLUMN: [5] * 1000 + [22] * 1000,
            "gender": [MALE_VALUE] * 2000,
            "employment_income": [0.0] * 2000,
        }
    )
    benunit = pd.DataFrame(
        {"benunit_id": list(range(1, 2001)), "benunit_weight": [1.0] * 2000}
    )
    household = pd.DataFrame(
        {
            "household_id": list(range(1, 2001)),
            "household_weight": [1.0] * 2000,
            "region": ["LONDON"] * 2000,
        }
    )
    ds = UKSingleYearDataset(
        person=person, benunit=benunit, household=household, fiscal_year=2023
    )

    # Aggregate over many seeds to smooth the Poisson noise.
    total_child_adds = 0
    total_adult_adds = 0
    for seed in range(20):
        out = apply_migration(
            ds,
            net_migration_rates=DEFAULT_NET_MIGRATION_RATES,
            rng=np.random.default_rng(seed),
        )
        delta_child = int((out.person[AGE_COLUMN] == 5).sum()) - 1000
        delta_adult = int((out.person[AGE_COLUMN] == 22).sum()) - 1000
        total_child_adds += delta_child
        total_adult_adds += delta_adult
    # Adults should get materially more inflow than children.
    assert total_adult_adds > total_child_adds
