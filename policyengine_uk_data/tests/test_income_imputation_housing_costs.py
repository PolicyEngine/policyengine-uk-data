"""Tests for preserving housing costs during SPI income imputation."""

from __future__ import annotations

import numpy as np
import pandas as pd


class _FixedIncomeModel:
    """Small stand-in for the QRF model used by income imputation."""

    def predict(self, input_df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "employment_income": [50_000.0, 80_000.0],
                "self_employment_income": [2_000.0, 0.0],
                "savings_interest_income": [200.0, 500.0],
                "dividend_income": [1_000.0, 2_500.0],
                "private_pension_income": [0.0, 5_000.0],
                "property_income": [0.0, 3_000.0],
            },
            index=input_df.index,
        )


def _tiny_dataset():
    from policyengine_uk.data import UKSingleYearDataset

    person = pd.DataFrame(
        {
            "person_id": [0, 1],
            "person_benunit_id": [0, 1],
            "person_household_id": [0, 1],
            "age": [35, 70],
            "gender": ["FEMALE", "MALE"],
            "employment_income": [10_000.0, 20_000.0],
            "self_employment_income": [0.0, 0.0],
            "savings_interest_income": [0.0, 0.0],
            "dividend_income": [0.0, 0.0],
            "private_pension_income": [0.0, 0.0],
            "property_income": [0.0, 0.0],
        }
    )
    benunit = pd.DataFrame({"benunit_id": [0, 1]})
    household = pd.DataFrame(
        {
            "household_id": [0, 1],
            "household_weight": [1.0, 1.0],
            "region": ["LONDON", "NORTH_EAST"],
            "tenure_type": ["RENT_PRIVATELY", "OWNED_WITH_MORTGAGE"],
            "council_tax": [1_500.0, 2_000.0],
            "rent": [12_000.0, 0.0],
            "mortgage_interest_repayment": [0.0, 4_000.0],
            "mortgage_capital_repayment": [0.0, 6_000.0],
        }
    )
    return UKSingleYearDataset(
        person=person,
        benunit=benunit,
        household=household,
        fiscal_year=2025,
    )


def test_impute_over_incomes_preserves_housing_costs():
    from policyengine_uk_data.datasets.imputations.income import (
        INCOME_COMPONENTS,
        impute_over_incomes,
    )

    dataset = _tiny_dataset()
    housing_columns = [
        "rent",
        "mortgage_interest_repayment",
        "mortgage_capital_repayment",
    ]
    before_housing = dataset.household[housing_columns].copy()

    result = impute_over_incomes(
        dataset,
        _FixedIncomeModel(),
        INCOME_COMPONENTS,
    )

    for column in housing_columns:
        np.testing.assert_array_equal(
            result.household[column].values,
            before_housing[column].values,
        )
    assert result.person["employment_income"].tolist() == [50_000.0, 80_000.0]
    assert dataset.person["employment_income"].tolist() == [10_000.0, 20_000.0]
