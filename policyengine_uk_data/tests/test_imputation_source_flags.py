from __future__ import annotations

import importlib

import numpy as np
import pandas as pd


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
