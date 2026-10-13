from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings, strategies as st
from policyengine_uk.data import UKSingleYearDataset


class _FakeDataset:
    def __init__(
        self,
        person: pd.DataFrame,
        household: pd.DataFrame,
        benunit: pd.DataFrame | None = None,
        fiscal_year: int = 2023,
    ):
        self.person = person
        self.household = household
        self.benunit = (
            benunit
            if benunit is not None
            else pd.DataFrame({"benunit_id": person["person_benunit_id"].unique()})
        )
        self.time_period = fiscal_year

    def copy(self):
        return _FakeDataset(
            person=self.person.copy(),
            household=self.household.copy(),
            benunit=self.benunit.copy(),
            fiscal_year=self.time_period,
        )

    def validate(self):
        return None


def _stack_without_remapping(left: _FakeDataset, right: _FakeDataset) -> _FakeDataset:
    return _FakeDataset(
        person=pd.concat([left.person, right.person], ignore_index=True),
        household=pd.concat([left.household, right.household], ignore_index=True),
        benunit=pd.concat([left.benunit, right.benunit], ignore_index=True),
        fiscal_year=left.time_period,
    )


def _fake_dataset() -> _FakeDataset:
    person = pd.DataFrame(
        {
            "person_id": [1, 2],
            "person_household_id": [1, 2],
            "person_benunit_id": [1, 2],
            "employment_income": [20_000.0, 80_000.0],
            "self_employment_income": [0.0, 0.0],
            "savings_interest_income": [0.0, 0.0],
            "dividend_income": [0.0, 0.0],
            "private_pension_income": [0.0, 0.0],
            "property_income": [0.0, 0.0],
        }
    )
    household = pd.DataFrame(
        {
            "household_id": [1, 2],
            "household_weight": [1.0, 2.0],
            "region": ["LONDON", "WALES"],
        }
    )
    return _FakeDataset(person=person, household=household)


def test_impute_income_marks_spi_synthetic_households(monkeypatch):
    from policyengine_uk_data.datasets.imputations import income as income_module
    from policyengine_uk_data.datasets import disability_benefits
    from policyengine_uk_data.datasets.imputations import frs_only

    monkeypatch.setattr(income_module, "create_income_model", lambda: object())
    monkeypatch.setattr(
        income_module,
        "subsample_dataset",
        lambda dataset, _sample_size: dataset.copy(),
    )
    monkeypatch.setattr(
        income_module,
        "impute_over_incomes",
        lambda dataset, _model, _output_variables: dataset,
    )
    monkeypatch.setattr(
        frs_only,
        "impute_frs_only_variables",
        lambda train_dataset, target_dataset: target_dataset,
    )
    monkeypatch.setattr(
        disability_benefits,
        "strip_internal_disability_reported_amounts",
        lambda dataset: dataset,
    )
    monkeypatch.setattr(income_module, "stack_datasets", _stack_without_remapping)

    result = income_module.impute_income(_fake_dataset())

    assert result.household["household_is_spi_synthetic"].tolist() == [
        False,
        False,
        True,
        True,
    ]
    assert result.household.loc[2:, "household_weight"].eq(0).all()


def test_impute_capital_gains_marks_capital_gains_clone_households(monkeypatch):
    cg_module = importlib.import_module(
        "policyengine_uk_data.datasets.imputations.capital_gains"
    )

    monkeypatch.setattr(cg_module, "stack_datasets", _stack_without_remapping)
    monkeypatch.setattr(
        cg_module,
        "impute_cg_to_doubled_dataset",
        lambda dataset: (
            np.zeros(len(dataset.person), dtype=float),
            dataset.household["household_weight"].to_numpy(dtype=float),
        ),
    )

    result = cg_module.impute_capital_gains(_fake_dataset())

    assert result.household["household_is_capital_gains_clone"].tolist() == [
        False,
        False,
        True,
        True,
    ]
    assert result.household.loc[2:, "household_weight"].eq(1).all()


def test_impute_income_keeps_reported_dividends_on_the_frs_half(monkeypatch):
    """SPI draws replace incomes only on the SPI-donor copy (policyengine-uk#1948).

    The SPI income model predicts from age, gender and region within earnings
    groups, so using it to overwrite the FRS half's dividends gave Universal
    Credit claimants dividends unrelated to any investments they reported.
    The stage-2 model then trains on the FRS half's reported dividends.
    """
    from policyengine_uk_data.datasets.imputations import income as income_module
    from policyengine_uk_data.datasets import disability_benefits
    from policyengine_uk_data.datasets.imputations import frs_only

    imputed_halves = []
    stage_2_training = []

    def fake_impute_over_incomes(dataset, _model, output_variables):
        imputed_halves.append(
            (
                bool(dataset.household["household_is_spi_synthetic"].all()),
                tuple(output_variables),
            )
        )
        dataset = dataset.copy()
        for column in output_variables:
            dataset.person[column] = 123_456.0
        return dataset

    monkeypatch.setattr(income_module, "create_income_model", lambda: object())
    monkeypatch.setattr(
        income_module,
        "subsample_dataset",
        lambda dataset, _sample_size: dataset.copy(),
    )
    monkeypatch.setattr(income_module, "impute_over_incomes", fake_impute_over_incomes)

    def fake_impute_frs_only_variables(train_dataset, target_dataset):
        stage_2_training.append(train_dataset.person["dividend_income"].tolist())
        return target_dataset

    monkeypatch.setattr(
        frs_only, "impute_frs_only_variables", fake_impute_frs_only_variables
    )
    monkeypatch.setattr(
        disability_benefits,
        "strip_internal_disability_reported_amounts",
        lambda dataset: dataset,
    )
    monkeypatch.setattr(income_module, "stack_datasets", _stack_without_remapping)

    dataset = _fake_dataset()
    dataset.person["dividend_income"] = [150.0, 0.0]
    result = income_module.impute_income(dataset)

    # Only the SPI-donor copy is imputed, with the full income set.
    assert imputed_halves == [(True, tuple(income_module.IMPUTATIONS))]
    frs_half = result.person.iloc[:2]
    assert frs_half["dividend_income"].tolist() == [150.0, 0.0]
    assert frs_half["employment_income"].tolist() == [20_000.0, 80_000.0]
    assert result.person.iloc[2:]["dividend_income"].eq(123_456.0).all()
    # Stage 2 learns from the FRS half's reported dividends, not SPI draws.
    assert stage_2_training == [[150.0, 0.0]]


def _reported_dividend_dataset(dividends_by_person) -> UKSingleYearDataset:
    person_ids = list(dividends_by_person)
    household_ids = [person_id // 1_000 for person_id in person_ids]
    person = pd.DataFrame(
        {
            "person_id": person_ids,
            "person_household_id": household_ids,
            "person_benunit_id": [
                household_id * 100 + 1 for household_id in household_ids
            ],
            "age": 40,
            "gender": "FEMALE",
            "employment_status": "FT_EMPLOYED",
            "employment_income": 20_000.0,
            "self_employment_income": 0.0,
            "savings_interest_income": 0.0,
            "dividend_income": list(dividends_by_person.values()),
            "private_pension_income": 0.0,
            "property_income": 0.0,
        },
        index=np.arange(len(person_ids)) * 7 + 10,
    )
    household_ids = list(dict.fromkeys(household_ids))[::-1]
    return UKSingleYearDataset(
        person=person,
        benunit=pd.DataFrame(
            {"benunit_id": [household_id * 100 + 1 for household_id in household_ids]},
            index=np.arange(len(household_ids)) * 3 + 50,
        ),
        household=pd.DataFrame(
            {
                "household_id": household_ids,
                "household_weight": 2.0,
                "region": "LONDON",
            },
            index=np.arange(len(household_ids)) * 5 + 20,
        ),
        fiscal_year=2025,
    )


def _stub_income_training(monkeypatch):
    """Avoid model training and simulation; retain income draws and real stacking."""
    from policyengine_uk_data.datasets.imputations import income as income_module
    from policyengine_uk_data.datasets.imputations import frs_only

    class FakeSimulation:
        def __init__(self, dataset):
            self.dataset = dataset

        def calculate_dataframe(self, variables):
            person = self.dataset.person
            regions = self.dataset.household.set_index("household_id").region
            return pd.DataFrame(
                {
                    "age": person.age.to_numpy(),
                    "gender": person.gender.to_numpy(),
                    "region": person.person_household_id.map(regions).to_numpy(),
                }
            )[variables]

    class FakeIncomeModel:
        def __init__(self):
            self.predictions = []

        def predict(self, inputs):
            self.predictions.append(inputs.copy())
            return pd.DataFrame(
                123_456.0, index=inputs.index, columns=income_module.IMPUTATIONS
            )

    def reordered_donor_copy(dataset, _sample_size):
        dataset = dataset.copy()
        dataset.person = dataset.person.iloc[::-1].copy()
        dataset.benunit = dataset.benunit.iloc[::-1].copy()
        dataset.household = dataset.household.iloc[::-1].copy()
        return dataset

    model = FakeIncomeModel()
    monkeypatch.setattr(income_module, "Microsimulation", FakeSimulation)
    monkeypatch.setattr(income_module, "create_income_model", lambda: model)
    monkeypatch.setattr(income_module, "subsample_dataset", reordered_donor_copy)
    monkeypatch.setattr(
        frs_only,
        "impute_frs_only_variables",
        lambda train_dataset, target_dataset: target_dataset,
    )
    return income_module, model


def _income_halves(dataset):
    source = dataset.household.set_index("household_id").household_is_spi_synthetic
    is_spi = dataset.person.person_household_id.map(source)
    assert is_spi.notna().all()
    return (
        dataset.person.loc[~is_spi].set_index("person_id"),
        dataset.person.loc[is_spi].set_index("person_id"),
    )


def test_account_dividends_survive_imputation_and_real_stacking(monkeypatch):
    from policyengine_uk_data.datasets.frs import WEEKS_IN_YEAR, frs_dividend_income

    income_module, model = _stub_income_training(monkeypatch)
    dataset = _reported_dividend_dataset(
        {2_001: 0.0, 1_002: 0.0, 1_001: 0.0, 2_002: 0.0}
    )
    account = pd.DataFrame(
        {
            "person_id": [1_002, 2_001, 2_001, 1_001, 2_002],
            "account": [8, 7, 6, 1, 6],
            "accint": [10.0, 4.0, 2.0, 50.0, 3.0],
            "invtax": [2, 1, 1, 2, 2],
        },
        index=[43, 19, 71, 7, 28],
    )
    dataset.person["dividend_income"] = frs_dividend_income(
        account, dataset.person.person_id
    )
    original_person = dataset.person.copy()

    result = income_module.impute_income(dataset)
    frs_half, spi_half = _income_halves(result)

    assert frs_half.dividend_income.to_dict() == pytest.approx(
        {2_001: 7.5 * WEEKS_IN_YEAR, 1_002: 10 * WEEKS_IN_YEAR, 1_001: 0, 2_002: 0}
    )
    assert frs_half.employment_income.eq(20_000.0).all()
    person_offset = original_person.person_id.max() + 1
    assert (
        spi_half.index.tolist()
        == (original_person.person_id.iloc[::-1] + person_offset).tolist()
    )
    assert spi_half.dividend_income.eq(123_456.0).all()
    assert len(model.predictions) == 1
    assert len(model.predictions[0]) == len(original_person)
    assert result.person.person_id.is_unique
    assert result.benunit.benunit_id.is_unique
    assert result.household.household_id.is_unique
    assert result.person.person_benunit_id.isin(result.benunit.benunit_id).all()
    pd.testing.assert_frame_equal(dataset.person, original_person)
    result.validate()


@given(
    st.dictionaries(
        st.integers(1, 50).map(lambda household_id: household_id * 1_000 + 1),
        st.integers(0, 1_000_000).map(lambda pence: pence / 100),
        min_size=1,
        max_size=8,
    )
)
@settings(max_examples=50, deadline=None)
def test_frs_dividend_totals_are_conserved_through_income_imputation(
    reported_dividends,
):
    # A fresh patch context per example avoids state leaking between property cases.
    with pytest.MonkeyPatch.context() as monkeypatch:
        income_module, _model = _stub_income_training(monkeypatch)
        dataset = _reported_dividend_dataset(reported_dividends)
        result = income_module.impute_income(dataset)

    frs_half, spi_half = _income_halves(result)
    assert frs_half.dividend_income.sum() == pytest.approx(
        sum(reported_dividends.values())
    )
    assert frs_half.dividend_income.to_dict() == pytest.approx(reported_dividends)
    assert spi_half.dividend_income.sum() == pytest.approx(
        len(reported_dividends) * 123_456
    )
