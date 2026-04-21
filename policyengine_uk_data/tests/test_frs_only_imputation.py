"""Unit tests for the second-stage FRS-only QRF imputation.

Guards the design decisions for ``impute_frs_only_variables``:

- Predictors are demographics + first-stage income outputs
- Outputs are a curated list of FRS-only variables
- Non-negative clamp is applied to predictions
- Missing variables in train or target are skipped silently, not raised
- Only the listed output variables are touched; other columns pass
  through unchanged
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _fake_dataset(person_rows: int, seed: int = 0):
    """Build a tiny UKSingleYearDataset-shaped stub with realistic columns.

    We don't need every FRS column; just enough for the stage-2
    imputation's predictor + output list to resolve. This is strictly a
    shape-and-wiring test, not an end-to-end retrain; the real QRF
    training path is exercised in the integration build.
    """
    from policyengine_uk.data import UKSingleYearDataset

    rng = np.random.default_rng(seed)
    person = pd.DataFrame(
        {
            "person_id": np.arange(person_rows),
            "person_benunit_id": np.arange(person_rows),
            "person_household_id": np.arange(person_rows),
            "age": rng.integers(18, 80, size=person_rows),
            "gender": np.where(rng.random(person_rows) < 0.5, "MALE", "FEMALE"),
            "region": np.where(rng.random(person_rows) < 0.5, "LONDON", "NORTH_EAST"),
            "employment_income": rng.gamma(2, 15_000, size=person_rows),
            "self_employment_income": np.where(
                rng.random(person_rows) < 0.1,
                rng.gamma(2, 20_000, size=person_rows),
                0.0,
            ),
            "savings_interest_income": rng.exponential(200, size=person_rows),
            "dividend_income": rng.exponential(500, size=person_rows),
            "private_pension_income": np.where(
                rng.random(person_rows) < 0.2,
                rng.gamma(2, 10_000, size=person_rows),
                0.0,
            ),
            "property_income": rng.exponential(300, size=person_rows),
            # FRS-only variables
            "employee_pension_contributions": rng.exponential(500, size=person_rows),
            "employer_pension_contributions": rng.exponential(1_500, size=person_rows),
            "personal_pension_contributions": rng.exponential(100, size=person_rows),
            "pension_contributions_via_salary_sacrifice": rng.exponential(
                50, size=person_rows
            ),
            "tax_free_savings_income": rng.exponential(50, size=person_rows),
            "universal_credit_reported": np.where(
                rng.random(person_rows) < 0.2,
                rng.gamma(2, 4_000, size=person_rows),
                0.0,
            ),
            "pension_credit_reported": 0.0,
            "child_benefit_reported": 0.0,
            "housing_benefit_reported": 0.0,
            "income_support_reported": 0.0,
            "working_tax_credit_reported": 0.0,
            "child_tax_credit_reported": 0.0,
            "attendance_allowance_reported": 0.0,
            "state_pension_reported": 0.0,
            "dla_sc_reported": 0.0,
            "dla_m_reported": 0.0,
            "pip_m_reported": 0.0,
            "pip_dl_reported": 0.0,
            "sda_reported": 0.0,
            "carers_allowance_reported": 0.0,
            "iidb_reported": 0.0,
            "afcs_reported": 0.0,
            "bsp_reported": 0.0,
            "incapacity_benefit_reported": 0.0,
            "maternity_allowance_reported": 0.0,
            "winter_fuel_allowance_reported": 0.0,
            "council_tax_benefit_reported": 0.0,
            "jsa_contrib_reported": 0.0,
            "jsa_income_reported": 0.0,
            "esa_contrib_reported": 0.0,
            "esa_income_reported": 0.0,
        }
    )
    benunit = pd.DataFrame({"benunit_id": np.arange(person_rows)})
    household = pd.DataFrame(
        {
            "household_id": np.arange(person_rows),
            "household_weight": 1.0,
            "region": person.region.values,
        }
    )
    return UKSingleYearDataset(
        person=person,
        benunit=benunit,
        household=household,
        fiscal_year=2023,
    )


def test_frs_only_outputs_are_non_negative():
    from policyengine_uk_data.datasets.imputations.frs_only import (
        FRS_ONLY_PERSON_VARIABLES,
        impute_frs_only_variables,
    )

    train = _fake_dataset(person_rows=400, seed=0)
    target = _fake_dataset(person_rows=60, seed=1)

    result = impute_frs_only_variables(
        train_dataset=train,
        target_dataset=target,
    )

    for column in FRS_ONLY_PERSON_VARIABLES:
        if column not in result.person.columns:
            continue
        values = result.person[column].values
        assert np.all(values >= 0), f"{column} has negative predictions"
        assert np.isfinite(values).all(), f"{column} has NaN / inf predictions"


def test_frs_only_does_not_touch_non_output_columns():
    """Stage-2 must only rewrite the curated output list, nothing else."""
    from policyengine_uk_data.datasets.imputations.frs_only import (
        impute_frs_only_variables,
    )

    train = _fake_dataset(person_rows=400, seed=0)
    target = _fake_dataset(person_rows=60, seed=1)

    pre_age = target.person["age"].copy()
    pre_income = target.person["employment_income"].copy()

    result = impute_frs_only_variables(
        train_dataset=train,
        target_dataset=target,
    )

    # Predictors (age, employment_income) must survive untouched.
    pd.testing.assert_series_equal(result.person["age"], pre_age, check_names=False)
    pd.testing.assert_series_equal(
        result.person["employment_income"], pre_income, check_names=False
    )


def test_frs_only_skips_missing_output_columns():
    """Variables absent from train or target are silently skipped."""
    from policyengine_uk_data.datasets.imputations.frs_only import (
        impute_frs_only_variables,
    )

    train = _fake_dataset(person_rows=200, seed=0)
    target = _fake_dataset(person_rows=40, seed=1)

    # Drop one output from each side to simulate variables that haven't
    # been wired into the FRS build yet. Must not raise.
    train.person = train.person.drop(columns=["sda_reported"])
    target.person = target.person.drop(columns=["iidb_reported"])

    result = impute_frs_only_variables(
        train_dataset=train,
        target_dataset=target,
    )

    # Target still lacks the column it entered without.
    assert "iidb_reported" not in result.person.columns
    # Other curated columns remain available and non-negative.
    assert "universal_credit_reported" in result.person.columns
    assert np.all(result.person["universal_credit_reported"].values >= 0)


def test_frs_only_reported_values_correlate_with_training_pattern():
    """UC ``_reported`` predictions should respect the training-data pattern.

    The stage-1 QRF-imputed income on the SPI-donor side gets fed back
    as a stage-2 predictor. If the training data only has non-zero UC
    for low-income respondents, the QRF should preferentially draw
    near-zero values when predicting for high-income target rows,
    compared with low-income target rows.
    """
    from policyengine_uk_data.datasets.imputations.frs_only import (
        impute_frs_only_variables,
    )

    # Train set with a clean employment-income → UC relationship:
    # low-income respondents sometimes claim UC, high-income never do.
    rng = np.random.default_rng(42)
    train = _fake_dataset(person_rows=2_000, seed=0)
    low_income_mask = train.person["employment_income"] < 20_000
    train.person["universal_credit_reported"] = 0.0
    train.person.loc[low_income_mask, "universal_credit_reported"] = rng.gamma(
        2, 4_000, size=int(low_income_mask.sum())
    )

    # Two target populations: one with high incomes, one with low incomes.
    high_target = _fake_dataset(person_rows=60, seed=1)
    high_target.person["employment_income"] = 500_000.0
    low_target = _fake_dataset(person_rows=60, seed=2)
    low_target.person["employment_income"] = 10_000.0

    high_result = impute_frs_only_variables(
        train_dataset=train,
        target_dataset=high_target,
    )
    low_result = impute_frs_only_variables(
        train_dataset=train,
        target_dataset=low_target,
    )

    high_mean = high_result.person["universal_credit_reported"].mean()
    low_mean = low_result.person["universal_credit_reported"].mean()
    assert high_mean < low_mean, (
        "Stage-2 QRF should produce lower UC-receipt predictions for high-"
        f"income target rows (got high={high_mean:.2f} vs low={low_mean:.2f})."
    )
