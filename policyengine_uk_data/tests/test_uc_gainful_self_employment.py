import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from policyengine_uk.variables.household.income.employment_status import (
    EmploymentStatus,
)

from policyengine_uk_data.datasets.frs import (
    SELF_EMPLOYED_STATUSES,
    derive_uc_is_in_gainful_self_employment,
)
from policyengine_uk_data.tests.test_imputation_source_flags import (
    _FakeDataset,
    _stack_without_remapping,
)

STATUSES = [status.name for status in EmploymentStatus]
GAINFUL = "uc_is_in_gainful_self_employment"
incomes = st.floats(0, 1e6, allow_nan=False)


def derive(status, profit, pay):
    return derive_uc_is_in_gainful_self_employment([status], [profit], [pay])[0]


def test_self_employed_statuses_are_model_enum_members():
    assert set(SELF_EMPLOYED_STATUSES) <= set(STATUSES)


@pytest.mark.parametrize("status", STATUSES)
@pytest.mark.parametrize("profit", [0.0, 5_000.0])
def test_every_status_without_a_side_trade_that_out_earns_pay(status, profit):
    # Pay at or above profit: only the main-job status decides.
    expected = status in ("FT_SELF_EMPLOYED", "PT_SELF_EMPLOYED")
    assert derive(status, profit, 10_000.0) == expected


@pytest.mark.parametrize("profit", [0.0, 1.0, 20_000.0])
def test_self_employed_main_job_is_gainful_whatever_the_profit(profit):
    # A loss reaches this build floored at zero, so zero covers losses too.
    for status in SELF_EMPLOYED_STATUSES:
        assert derive(status, profit, 0.0)
        assert derive(status, profit, 50_000.0)


def test_side_trade_counts_only_when_it_out_earns_pay():
    # ADM H4034 example 1 (Jos): more employed hours, more self-employed pay.
    assert derive("PT_EMPLOYED", 140 * 52, 80 * 52)
    # ADM H4035 example 2 (Ann): the job earns more.
    assert not derive("PT_EMPLOYED", 40 * 52, 49.6 * 52)
    assert not derive("FT_EMPLOYED", 10_000.0, 10_000.0)
    assert not derive("FT_EMPLOYED", 0.0, 0.0)
    assert derive("UNEMPLOYED", 1.0, 0.0)


@settings(max_examples=500, deadline=None)
@given(st.sampled_from(STATUSES), incomes, incomes)
def test_invariants(status, profit, pay):
    gainful = derive(status, profit, pay)
    # Every self-employed main job is gainful.
    if status in SELF_EMPLOYED_STATUSES:
        assert gainful
    # Nobody else is gainful without a profit above their pay.
    if gainful:
        assert status in SELF_EMPLOYED_STATUSES or profit > max(pay, 0)
    # More profit never removes the flag; more pay never adds it.
    assert derive(status, profit * 2 + 1, pay) >= gainful
    assert derive(status, profit, pay * 2 + 1) <= gainful


@settings(max_examples=100, deadline=None)
@given(st.lists(st.tuples(st.sampled_from(STATUSES), incomes, incomes), max_size=40))
def test_vectorised_matches_elementwise(rows):
    statuses = [r[0] for r in rows]
    profits = [r[1] for r in rows]
    pays = [r[2] for r in rows]
    result = derive_uc_is_in_gainful_self_employment(statuses, profits, pays)
    assert result.dtype == bool
    assert result.tolist() == [derive(*r) for r in rows]
    np.testing.assert_array_equal(
        derive_uc_is_in_gainful_self_employment(
            pd.Series(statuses, dtype="category"),
            pd.Series(profits, dtype=float),
            pd.Series(pays, dtype=float),
        ),
        result,
    )


def _frs_like_dataset(statuses, profits, pays):
    n = len(statuses)
    person = pd.DataFrame(
        {
            "person_id": np.arange(1, n + 1),
            "person_household_id": np.arange(1, n + 1),
            "person_benunit_id": np.arange(1, n + 1),
            "employment_status": statuses,
            "employment_income": pays,
            "self_employment_income": profits,
            "savings_interest_income": 0.0,
            "dividend_income": 0.0,
            "private_pension_income": 0.0,
            "property_income": 0.0,
        }
    )
    person[GAINFUL] = derive_uc_is_in_gainful_self_employment(
        person.employment_status,
        person.self_employment_income,
        person.employment_income,
    )
    household = pd.DataFrame(
        {
            "household_id": np.arange(1, n + 1),
            "household_weight": 1.0,
            "region": "LONDON",
        }
    )
    return _FakeDataset(person=person, household=household)


@settings(max_examples=50, deadline=None)
@given(
    st.lists(
        st.tuples(st.sampled_from(STATUSES), incomes, incomes, incomes, incomes),
        min_size=1,
        max_size=10,
    )
)
def test_spi_copy_flag_follows_its_own_imputed_incomes(rows):
    from policyengine_uk_data.datasets import disability_benefits
    from policyengine_uk_data.datasets.imputations import frs_only
    from policyengine_uk_data.datasets.imputations import income as income_module

    imputed_profit = [r[3] for r in rows]
    imputed_pay = [r[4] for r in rows]

    def impute_over_incomes(dataset, _model, output_variables):
        dataset = dataset.copy()
        if "self_employment_income" in output_variables:
            dataset.person["self_employment_income"] = imputed_profit
            dataset.person["employment_income"] = imputed_pay
        return dataset

    with pytest.MonkeyPatch.context() as m:
        m.setattr(income_module, "create_income_model", lambda: object())
        m.setattr(income_module, "subsample_dataset", lambda d, _n: d.copy())
        m.setattr(income_module, "impute_over_incomes", impute_over_incomes)
        m.setattr(
            frs_only,
            "impute_frs_only_variables",
            lambda train_dataset, target_dataset: target_dataset,
        )
        m.setattr(
            disability_benefits,
            "strip_internal_disability_reported_amounts",
            lambda dataset: dataset,
        )
        m.setattr(income_module, "stack_datasets", _stack_without_remapping)
        result = income_module.impute_income(
            _frs_like_dataset(
                [r[0] for r in rows], [r[1] for r in rows], [r[2] for r in rows]
            )
        )

    person = result.person
    n = len(rows)
    assert len(person) == 2 * n
    # The SPI copy kept the donors' statuses and took the imputed incomes.
    assert person.self_employment_income.iloc[n:].tolist() == imputed_profit
    np.testing.assert_array_equal(
        person[GAINFUL].to_numpy(dtype=bool),
        derive_uc_is_in_gainful_self_employment(
            person.employment_status,
            person.self_employment_income,
            person.employment_income,
        ),
    )


@pytest.mark.parametrize("fixture", ["frs", "enhanced_frs"])
def test_built_dataset_flag(fixture, request):
    dataset = request.getfixturevalue(fixture)
    person = dataset.person
    if GAINFUL not in person.columns:
        pytest.skip(f"{fixture} was built before this input existed")
    gainful = person[GAINFUL].to_numpy(dtype=bool)
    self_employed = np.isin(person.employment_status, SELF_EMPLOYED_STATUSES)
    assert gainful[self_employed].all()
    profit = person.self_employment_income.to_numpy()
    assert (profit[gainful & ~self_employed] > 0).all()
    if fixture == "frs":
        # Later enhanced-FRS stages reprice incomes, so the exact rule holds on
        # the base build only.
        np.testing.assert_array_equal(
            gainful,
            derive_uc_is_in_gainful_self_employment(
                person.employment_status, profit, person.employment_income
            ),
        )
