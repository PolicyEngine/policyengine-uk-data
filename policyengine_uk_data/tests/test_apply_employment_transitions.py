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


# -- UKHLS-rates branch ------------------------------------------------------


def test_ukhls_rates_override_rule_based_path():
    """When ``ukhls_rates`` is supplied, the rule-based job_loss/gain is skipped.

    Use UKHLS rates that *keep* everyone in work with probability 1 — if
    the rule-based path were running in parallel, some workers would
    still lose their jobs at the default rate.
    """
    n = 1_000
    ds = _population(ages=[35] * n, incomes=[25_000.0] * n)
    # Everyone stays employed regardless of age band.
    ukhls_rates = {
        (f"{lo}-{lo + 4}", "MALE", "IN_WORK"): {"IN_WORK": 1.0}
        for lo in range(16, 76, 5)
    }
    ukhls_rates.update(
        {
            (f"{lo}-{lo + 4}", "FEMALE", "IN_WORK"): {"IN_WORK": 1.0}
            for lo in range(16, 76, 5)
        }
    )
    out = apply_employment_transitions(
        ds,
        ukhls_rates=ukhls_rates,
        job_loss_rate=0.99,  # would zero out nearly everyone if applied
        wage_drift=0.0,
        rng=np.random.default_rng(0),
    )
    zero_after = int((out.person["employment_income"] == 0).sum())
    assert zero_after == 0


def test_ukhls_rates_drive_people_into_work():
    """If rates say UNEMPLOYED → IN_WORK w.p. 1, every unemployed becomes employed."""
    n = 500
    ds = _population(
        ages=[35] * n,
        incomes=[25_000.0] * (n // 2) + [0.0] * (n // 2),
    )
    ukhls_rates = {
        (band, sex, "UNEMPLOYED"): {"IN_WORK": 1.0}
        for band in ["30-34", "35-39"]
        for sex in ["MALE", "FEMALE"]
    }
    # And keep employed in work to avoid interference.
    ukhls_rates.update(
        {
            (band, sex, "IN_WORK"): {"IN_WORK": 1.0}
            for band in ["30-34", "35-39"]
            for sex in ["MALE", "FEMALE"]
        }
    )
    out = apply_employment_transitions(
        ds,
        ukhls_rates=ukhls_rates,
        wage_drift=0.0,
        rng=np.random.default_rng(0),
    )
    # Every row now has positive income.
    assert (out.person["employment_income"] > 0).all()


def test_ukhls_rates_respect_retirement_at_spa():
    """Even with UKHLS rates, people at SPA still get retired (runs first)."""
    ds = _population(ages=[67], incomes=[40_000.0])
    ukhls_rates = {("65-69", "MALE", "IN_WORK"): {"IN_WORK": 1.0}}
    out = apply_employment_transitions(
        ds,
        ukhls_rates=ukhls_rates,
        state_pension_age=66,
        wage_drift=0.0,
        rng=np.random.default_rng(0),
    )
    assert float(out.person["employment_income"].iloc[0]) == 0.0


def test_ukhls_rates_committed_csv_is_consumable():
    """Loaded transition rates from the committed CSV feed in without shape errors."""
    import pathlib

    path = (
        pathlib.Path(__file__).parents[2]
        / "policyengine_uk_data"
        / "storage"
        / "ukhls_employment_state_transitions.csv"
    )
    if not path.exists():
        import pytest as _pt

        _pt.skip("UKHLS transitions CSV not present in this checkout")
    from policyengine_uk_data.utils.ukhls_transitions import (
        load_employment_transitions,
    )

    ukhls_rates = load_employment_transitions()
    assert ukhls_rates, "committed transition table must not be empty"
    ds = _population(ages=[35, 42, 55], incomes=[25_000.0, 40_000.0, 0.0])
    out = apply_employment_transitions(
        ds,
        ukhls_rates=ukhls_rates,
        wage_drift=0.0,
        rng=np.random.default_rng(0),
    )
    assert len(out.person) == 3
