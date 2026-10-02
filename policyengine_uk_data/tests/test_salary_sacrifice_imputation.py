"""Salary sacrifice imputation keeps every sacrifice within the person's pay.

Invariants of ``limit_salary_sacrifice_to_pay``, for every input:

- the result is between zero and the person's pay, so zero without pay;
- an amount already within those bounds is unchanged;
- applying it twice changes nothing;
- it is non-decreasing in the sacrifice and in the pay.

``impute_salary_sacrifice`` applies it to every record, reported or
imputed, and stage 2 moves employee pension contributions to salary
sacrifice without changing anyone's total of the two.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from policyengine_uk_data.datasets.imputations import (
    salary_sacrifice as salary_sacrifice_module,
)
from policyengine_uk_data.datasets.imputations.salary_sacrifice import (
    limit_salary_sacrifice_to_pay,
)

amounts = st.floats(min_value=-1e5, max_value=1e7, allow_nan=False)
pairs = st.lists(st.tuples(amounts, amounts), min_size=1, max_size=40)


def _split(rows):
    ss, pay = np.array(rows, dtype=float).T
    return ss, pay


@given(pairs)
def test_result_is_between_zero_and_pay(rows):
    ss, pay = _split(rows)
    limited = limit_salary_sacrifice_to_pay(ss, pay)
    assert np.all(limited >= 0)
    assert np.all(limited <= np.maximum(pay, 0))
    assert np.all(limited[pay <= 0] == 0)


@given(pairs)
def test_amounts_within_pay_are_unchanged(rows):
    ss, pay = _split(rows)
    within = (ss >= 0) & (ss <= pay)
    limited = limit_salary_sacrifice_to_pay(ss, pay)
    assert np.array_equal(limited[within], ss[within])


@given(pairs)
def test_limit_is_idempotent(rows):
    ss, pay = _split(rows)
    once = limit_salary_sacrifice_to_pay(ss, pay)
    assert np.array_equal(limit_salary_sacrifice_to_pay(once, pay), once)


@given(pairs, st.floats(min_value=0, max_value=1e6), st.floats(0, 1e6))
def test_limit_is_monotone_in_sacrifice_and_pay(rows, more_ss, more_pay):
    ss, pay = _split(rows)
    limited = limit_salary_sacrifice_to_pay(ss, pay)
    assert np.all(limit_salary_sacrifice_to_pay(ss + more_ss, pay) >= limited)
    assert np.all(limit_salary_sacrifice_to_pay(ss, pay + more_pay) >= limited)


class _StubModel:
    """Stands in for the trained QRF: predicts a fixed amount per row."""

    def __init__(self, predictions):
        self.predictions = np.asarray(predictions, dtype=float)

    def predict(self, X):
        assert len(X) == len(self.predictions)
        return pd.DataFrame(
            {"pension_contributions_via_salary_sacrifice": self.predictions}
        )


def _dataset(pay, asked, reported, employee_pension):
    from policyengine_uk.data import UKSingleYearDataset

    n = len(pay)
    person = pd.DataFrame(
        {
            "person_id": np.arange(n),
            "person_benunit_id": np.arange(n),
            "person_household_id": np.arange(n),
            "age": 40,
            "employment_income": np.asarray(pay, dtype=float),
            "salary_sacrifice_asked": np.asarray(asked, dtype=int),
            "pension_contributions_via_salary_sacrifice": np.asarray(
                reported, dtype=float
            ),
            "employee_pension_contributions": np.asarray(employee_pension, dtype=float),
        }
    )
    benunit = pd.DataFrame({"benunit_id": np.arange(n)})
    household = pd.DataFrame(
        {
            "household_id": np.arange(n),
            "household_weight": 1.0,
            "region": "LONDON",
            "tenure_type": "OWNED_OUTRIGHT",
            "council_tax": 0.0,
            "rent": 0.0,
        }
    )
    return UKSingleYearDataset(
        person=person, benunit=benunit, household=household, fiscal_year=2025
    )


def _impute(monkeypatch, predictions, **columns):
    monkeypatch.setattr(
        salary_sacrifice_module,
        "create_salary_sacrifice_model",
        lambda: _StubModel(predictions),
    )
    dataset = _dataset(**columns)
    before = dataset.person.copy()
    after = salary_sacrifice_module.impute_salary_sacrifice(dataset).person
    return before, after


def _assert_invariants(before, after):
    pay = before.employment_income.values
    ss = after.pension_contributions_via_salary_sacrifice.values
    assert np.all(ss >= 0)
    assert np.all(ss <= np.maximum(pay, 0))
    assert np.all(ss[pay <= 0] == 0)
    # Stage 2 moves employee pension to salary sacrifice: whoever lost
    # employee pension gained exactly that much salary sacrifice.
    pension_before = before.employee_pension_contributions.values
    pension_after = after.employee_pension_contributions.values
    moved = pension_after != pension_before
    assert np.all(pension_after <= pension_before)
    assert np.allclose(pension_after[moved] + ss[moved], pension_before[moved])


def test_nobody_without_pay_keeps_a_sacrifice(monkeypatch):
    pay = [0, 0, -500, 30_000, 30_000, 8_000, 0]
    before, after = _impute(
        monkeypatch,
        # The model predicts a large sacrifice for everyone.
        predictions=[25_000] * 7,
        pay=pay,
        asked=[0, 1, 0, 0, 1, 0, 1],
        reported=[0, 6_000, 0, 0, 1_200, 0, 0],
        employee_pension=[0] * 7,
    )
    ss = after.pension_contributions_via_salary_sacrifice.values
    # Unpaid: imputed (0, 2) and reported (1) amounts are all dropped.
    assert ss[[0, 1, 2, 6]].tolist() == [0, 0, 0, 0]
    # Paid and not asked: the prediction, limited to pay.
    assert ss[3] == 25_000
    assert ss[5] == 8_000
    # Paid and asked: the reported amount is kept.
    assert ss[4] == 1_200
    _assert_invariants(before, after)


def test_stage_two_moves_employee_pension_up_to_pay(monkeypatch):
    n = 40
    # Every donor's employee pension exceeds their pay, so each one moved
    # keeps the part above their pay as employee pension.
    before, after = _impute(
        monkeypatch,
        predictions=[0] * n,
        pay=[1_000] * n,
        asked=[0] * n,
        reported=[0] * n,
        employee_pension=[1_500] * n,
    )
    ss = after.pension_contributions_via_salary_sacrifice.values
    moved = ss > 0
    assert moved.any()
    assert np.all(ss[moved] == 1_000)
    assert np.all(after.employee_pension_contributions.values[moved] == 500)
    _assert_invariants(before, after)


row = st.tuples(
    st.sampled_from([-1_000.0, 0.0, 500.0, 15_000.0, 60_000.0]),  # pay
    st.integers(0, 1),  # asked SALSAC
    st.sampled_from([0.0, 800.0, 6_000.0, 70_000.0]),  # reported amount
    st.sampled_from([0.0, 300.0, 3_000.0, 90_000.0]),  # employee pension
    st.sampled_from([-100.0, 0.0, 1_500.0, 27_000.0]),  # model prediction
)


@settings(max_examples=10, deadline=None)
@given(st.lists(row, min_size=1, max_size=12))
def test_imputation_keeps_every_sacrifice_within_pay(rows):
    pay, asked, reported, employee_pension, predictions = map(list, zip(*rows))
    with pytest.MonkeyPatch.context() as monkeypatch:
        before, after = _impute(
            monkeypatch,
            predictions=predictions,
            pay=pay,
            asked=asked,
            reported=reported,
            employee_pension=employee_pension,
        )
    _assert_invariants(before, after)
