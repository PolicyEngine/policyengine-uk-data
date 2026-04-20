"""Tests for the rule-based employment / income transitions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from policyengine_uk.data import UKSingleYearDataset

from policyengine_uk_data.utils.demographic_ageing import (
    AGE_COLUMN,
    FEMALE_VALUE,
    MALE_VALUE,
)
from policyengine_uk_data.utils.household_transitions import (
    DEFAULT_STATE_PENSION_AGE,
    apply_employment_transitions,
)


def _population(
    *,
    ages: list[int],
    incomes: list[float],
    self_incomes: list[float] | None = None,
    with_status: bool = False,
) -> UKSingleYearDataset:
    n = len(ages)
    data = {
        "person_id": list(range(1, n + 1)),
        "person_benunit_id": list(range(1, n + 1)),
        "person_household_id": list(range(1, n + 1)),
        AGE_COLUMN: ages,
        "gender": [MALE_VALUE] * n,
        "employment_income": incomes,
    }
    if self_incomes is not None:
        data["self_employment_income"] = self_incomes
    if with_status:
        data["employment_status"] = [""] * n
    person = pd.DataFrame(data)
    benunit = pd.DataFrame(
        {"benunit_id": data["person_benunit_id"], "benunit_weight": [1.0] * n}
    )
    household = pd.DataFrame(
        {
            "household_id": data["person_household_id"],
            "household_weight": [1.0] * n,
            "region": ["LONDON"] * n,
        }
    )
    return UKSingleYearDataset(
        person=person, benunit=benunit, household=household, fiscal_year=2023
    )


def test_retirement_zeros_labour_income_at_spa():
    ds = _population(ages=[65, 66, 67], incomes=[40_000.0, 45_000.0, 38_000.0])
    out = apply_employment_transitions(
        ds,
        state_pension_age=66,
        job_loss_rate=0.0,
        job_gain_rate=0.0,
        wage_drift=0.0,
        rng=np.random.default_rng(0),
    )
    # 65 (under SPA) keeps their income; 66 and 67 (at/above) zeroed.
    incomes = out.person.sort_values(AGE_COLUMN)["employment_income"].tolist()
    assert incomes == [40_000.0, 0.0, 0.0]


def test_self_employment_income_also_zeroed_on_retirement():
    ds = _population(
        ages=[65, 67],
        incomes=[0.0, 0.0],
        self_incomes=[20_000.0, 15_000.0],
    )
    out = apply_employment_transitions(
        ds,
        state_pension_age=66,
        job_loss_rate=0.0,
        job_gain_rate=0.0,
        wage_drift=0.0,
        rng=np.random.default_rng(0),
    )
    assert out.person.sort_values(AGE_COLUMN)["self_employment_income"].tolist() == [
        20_000.0,
        0.0,
    ]


def test_employment_status_is_set_to_retired():
    ds = _population(ages=[67], incomes=[40_000.0], with_status=True)
    out = apply_employment_transitions(
        ds,
        state_pension_age=66,
        job_loss_rate=0.0,
        job_gain_rate=0.0,
        wage_drift=0.0,
        rng=np.random.default_rng(0),
    )
    assert out.person["employment_status"].iloc[0] == "RETIRED"


def test_wage_drift_scales_active_workers_only():
    ds = _population(ages=[30, 40], incomes=[20_000.0, 30_000.0])
    out = apply_employment_transitions(
        ds,
        state_pension_age=66,
        job_loss_rate=0.0,
        job_gain_rate=0.0,
        wage_drift=0.05,
        rng=np.random.default_rng(0),
    )
    assert out.person["employment_income"].tolist() == [21_000.0, 31_500.0]


def test_wage_drift_does_not_touch_retirees():
    ds = _population(ages=[67, 70], incomes=[0.0, 0.0])
    out = apply_employment_transitions(
        ds,
        state_pension_age=66,
        job_loss_rate=0.0,
        job_gain_rate=0.0,
        wage_drift=0.05,
        rng=np.random.default_rng(0),
    )
    assert out.person["employment_income"].tolist() == [0.0, 0.0]


def test_job_loss_zeros_incomes_at_expected_rate():
    n = 1000
    ds = _population(ages=[35] * n, incomes=[25_000.0] * n)
    out = apply_employment_transitions(
        ds,
        state_pension_age=66,
        job_loss_rate=0.10,
        job_gain_rate=0.0,
        wage_drift=0.0,
        rng=np.random.default_rng(0),
    )
    zero_after = int((out.person["employment_income"] == 0).sum())
    # With p=0.10 and n=1,000 expect ~100 zeros; allow a generous 30 margin.
    assert 70 <= zero_after <= 130


def test_job_gain_creates_income_for_unemployed():
    # Half the population has income, half doesn't. Gain rate should
    # turn some zeros into non-zero income at roughly gain_rate.
    n = 1000
    ds = _population(
        ages=[35] * n,
        incomes=[25_000.0] * (n // 2) + [0.0] * (n // 2),
    )
    out = apply_employment_transitions(
        ds,
        state_pension_age=66,
        job_loss_rate=0.0,
        job_gain_rate=0.20,
        wage_drift=0.0,
        rng=np.random.default_rng(0),
    )
    previously_unemployed = ds.person[ds.person["employment_income"] == 0]
    now_employed = out.person[
        (out.person["person_id"].isin(previously_unemployed["person_id"]))
        & (out.person["employment_income"] > 0)
    ]
    # ~100 gainers expected.
    assert 60 <= len(now_employed) <= 140


def test_deterministic_under_same_seed():
    ds = _population(ages=[30, 40, 50, 67], incomes=[20_000.0, 30_000.0, 40_000.0, 0.0])
    a = apply_employment_transitions(ds, rng=np.random.default_rng(5))
    b = apply_employment_transitions(ds, rng=np.random.default_rng(5))
    pd.testing.assert_frame_equal(a.person, b.person)


def test_missing_employment_columns_does_not_crash():
    # Minimal schema without employment_status / self_employment_income.
    ds = _population(ages=[30], incomes=[20_000.0])
    out = apply_employment_transitions(
        ds,
        state_pension_age=66,
        job_loss_rate=0.0,
        job_gain_rate=0.0,
        wage_drift=0.03,
        rng=np.random.default_rng(0),
    )
    assert out.person["employment_income"].iloc[0] == pytest.approx(20_600.0)


def test_default_spa_constant_is_66():
    """Sanity-check the default matches current UK SPA."""
    assert DEFAULT_STATE_PENSION_AGE == 66
