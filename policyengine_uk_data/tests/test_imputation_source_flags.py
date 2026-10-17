from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import pytest


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


def _fake_households(dataset: _FakeDataset, household_ids) -> _FakeDataset:
    person = dataset.person[dataset.person.person_household_id.isin(household_ids)]
    return _FakeDataset(
        person=person.reset_index(drop=True).copy(),
        household=dataset.household[dataset.household.household_id.isin(household_ids)]
        .reset_index(drop=True)
        .copy(),
        fiscal_year=dataset.time_period,
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
    result = income_module.impute_income(_fake_dataset())

    assert result.household["household_is_spi_synthetic"].tolist() == [
        False,
        False,
        True,
        True,
    ]
    assert result.household.loc[2:, "household_weight"].eq(0).all()
    assert result.household["source_household_id"].tolist() == [1, 2, 1, 2]


def test_impute_capital_gains_marks_capital_gains_clone_households(monkeypatch):
    cg_module = importlib.import_module(
        "policyengine_uk_data.datasets.imputations.capital_gains"
    )

    # Both people are first adults at the top of the income distribution;
    # only the second draws a gain HMRC's statistics describe.
    top_band = len(cg_module.capital_gains) - 1
    monkeypatch.setattr(
        cg_module,
        "_first_adult_incomes",
        lambda dataset: (
            np.full(len(dataset.person), 1e6),
            np.ones(len(dataset.person), dtype=bool),
        ),
    )
    monkeypatch.setattr(
        cg_module, "draw_gains", lambda income_band, gainer: np.array([0.0, 50_000.0])
    )

    result = cg_module.impute_capital_gains(_fake_dataset())

    # Each household is kept, and the one with a gain is stacked again as
    # its has-gains twin.
    assert result.household["household_is_capital_gains_clone"].tolist() == [
        False,
        False,
        True,
    ]
    share = cg_module.capital_gains.percent_with_gains.iloc[top_band]
    assert result.household["household_weight"].tolist() == pytest.approx(
        [1.0, 2.0 * (1 - share), 2.0 * share]
    )
    assert result.person["capital_gains"].tolist() == [0.0, 0.0, 50_000.0]
    assert result.household["source_household_id"].tolist() == [1, 2, 2]


def test_built_dataset_sources_are_original_survey_households(enhanced_frs):
    """Issue #591: disclosure support must count each survey household once."""
    household = enhanced_frs.household
    original = ~(
        household["household_is_spi_synthetic"]
        | household["household_is_capital_gains_clone"]
        | household["household_is_cgt_band_donor"]
    ) & (household["clone_index"] == 0)
    original_ids = household.loc[original, "household_id"]

    # Report aggregate counts on failure, never private survey identifiers.
    n_unknown_sources = int(
        (~household["source_household_id"].isin(original_ids)).sum()
    )
    n_misidentified_originals = int(
        (household.loc[original, "source_household_id"] != original_ids).sum()
    )
    assert n_unknown_sources == 0
    assert n_misidentified_originals == 0
