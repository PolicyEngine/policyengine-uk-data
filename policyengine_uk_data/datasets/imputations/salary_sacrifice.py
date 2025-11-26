"""
Salary sacrifice imputation for pension contributions.

This module imputes salary sacrifice participation using machine learning
models trained on FRS respondents who were asked the SALSAC question.

Training data (FRS 2023-24):
- SALSAC='1' (Yes, uses salary sacrifice): ~224 jobs
- SALSAC='2' (No, doesn't use): ~3,803 jobs

Imputation candidates:
- SALSAC=' ' (skip/not asked): ~13,265 jobs

The imputation predicts participation based on the observed relationship
in the training data. Targeting to HMRC totals (~24bn SS contributions,
~30% participation rate) happens via weight calibration, not here.
"""

import pandas as pd
import numpy as np
from policyengine_uk_data.storage import STORAGE_FOLDER
from policyengine_uk.data import UKSingleYearDataset
from policyengine_uk import Microsimulation


PREDICTORS = [
    "age",
    "employment_income",
]

IMPUTATIONS = [
    "uses_salary_sacrifice",
]


def save_salary_sacrifice_model():
    """
    Train and save salary sacrifice imputation model using FRS SALSAC data.

    Uses FRS respondents who were asked about salary sacrifice (SALSAC field)
    as training data to predict participation for non-respondents.

    Returns:
        Trained QRF model for salary sacrifice imputation.
    """
    from policyengine_uk_data.utils import QRF

    # Load the base FRS dataset
    frs_path = STORAGE_FOLDER / "frs_2023_24.h5"
    if not frs_path.exists():
        raise FileNotFoundError(
            f"FRS dataset not found at {frs_path}. "
            "Run create_frs() first to generate the base dataset."
        )

    dataset = UKSingleYearDataset(frs_path)
    sim = Microsimulation(dataset=dataset)

    # Get predictor variables
    age = sim.calculate("age").values
    employment_income = sim.calculate("employment_income").values

    # Get salary sacrifice indicators from dataset
    if "salary_sacrifice_asked" not in dataset.person.columns:
        raise ValueError(
            "Dataset missing salary_sacrifice_asked field. "
            "Ensure frs.py extracts SALSAC before numeric conversion."
        )

    ss_reported = dataset.person.salary_sacrifice_reported.values
    ss_asked = dataset.person.salary_sacrifice_asked.values

    # Build training DataFrame with only those who were asked
    training_mask = ss_asked == 1

    if training_mask.sum() == 0:
        raise ValueError(
            "No training data found - no respondents were asked SALSAC."
        )

    train_df = pd.DataFrame(
        {
            "age": age[training_mask],
            "employment_income": employment_income[training_mask],
            "uses_salary_sacrifice": ss_reported[training_mask].astype(bool),
        }
    )

    print(f"Training salary sacrifice model on {len(train_df)} observations")
    print(
        f"  SS users: {train_df['uses_salary_sacrifice'].sum()} "
        f"({train_df['uses_salary_sacrifice'].mean():.1%})"
    )

    # Train QRF model
    model = QRF()
    model.fit(train_df[PREDICTORS], train_df[IMPUTATIONS])
    model.save(STORAGE_FOLDER / "salary_sacrifice.pkl")

    return model


def create_salary_sacrifice_model(overwrite_existing: bool = False):
    """
    Create or load salary sacrifice participation model.

    Args:
        overwrite_existing: Whether to retrain model if it exists.

    Returns:
        Trained QRF model for salary sacrifice imputation.
    """
    from policyengine_uk_data.utils.qrf import QRF

    model_path = STORAGE_FOLDER / "salary_sacrifice.pkl"
    if model_path.exists() and not overwrite_existing:
        return QRF(file_path=model_path)
    return save_salary_sacrifice_model()


def impute_salary_sacrifice(
    dataset: UKSingleYearDataset,
) -> UKSingleYearDataset:
    """
    Impute salary sacrifice participation for FRS non-respondents.

    For respondents not asked about salary sacrifice (SALSAC=' '), uses
    a QRF model trained on those who were asked to predict participation.
    For participants, assigns SS contributions equal to their employee
    pension contributions.

    Note: This imputation does NOT target any specific participation rate
    or contribution total. Targeting to HMRC figures happens via weight
    calibration in a subsequent step.

    Args:
        dataset: PolicyEngine UK dataset with salary_sacrifice_reported
            and salary_sacrifice_asked fields from FRS processing.

    Returns:
        Dataset with imputed salary sacrifice participation and amounts.
    """
    dataset = dataset.copy()
    sim = Microsimulation(dataset=dataset)

    # Get variables needed for imputation
    age = sim.calculate("age").values
    employment_income = sim.calculate("employment_income").values
    employee_pension = sim.calculate("employee_pension_contributions").values
    current_ss = dataset.person.pension_contributions_via_salary_sacrifice

    # Get indicators
    if "salary_sacrifice_reported" in dataset.person.columns:
        ss_reported = dataset.person.salary_sacrifice_reported.values
        ss_asked = dataset.person.salary_sacrifice_asked.values
    else:
        # If indicators not available, skip imputation
        print(
            "Warning: salary_sacrifice_asked not in dataset, "
            "skipping imputation"
        )
        return dataset

    # Identify imputation candidates: those not asked about SS
    not_asked = ss_asked == 0

    # Create prediction DataFrame for all records
    pred_df = pd.DataFrame(
        {
            "age": age,
            "employment_income": employment_income,
        }
    )

    # Get or train model and predict
    model = create_salary_sacrifice_model()
    predictions = model.predict(pred_df)

    # microimpute returns boolean for bool target variables
    imputed_uses_ss = predictions["uses_salary_sacrifice"].values

    # For those who were asked, use their actual response
    # For those not asked, use the imputed value
    final_uses_ss = np.where(
        ss_asked == 1,
        ss_reported.astype(bool),
        imputed_uses_ss,
    )

    # For SS participants, set their SS contributions equal to
    # employee pension contributions (typical SS arrangement)
    # Only impute amounts for those not asked - keep reported amounts
    new_ss_amounts = np.where(
        final_uses_ss & (not_asked),
        employee_pension,
        0,
    )

    # Combine with existing SS amounts
    final_ss = np.where(
        ss_asked == 1,
        current_ss.values,  # Keep reported values exactly
        np.maximum(current_ss.values, new_ss_amounts),
    )

    # Update dataset
    dataset.person["pension_contributions_via_salary_sacrifice"] = final_ss

    # Report results (no targeting - just descriptive)
    weights = sim.calculate("person_weight").values
    is_employee = employment_income > 0
    total_ss = (final_ss * weights).sum()
    participation_rate = (final_uses_ss * weights * is_employee).sum() / (
        weights * is_employee
    ).sum()

    print("Salary sacrifice imputation results (pre-calibration):")
    print(f"  Total SS contributions: {total_ss / 1e9:.1f}bn")
    print(f"  Employee participation rate: {participation_rate:.1%}")
    print("  (Final totals depend on subsequent weight calibration)")

    return dataset
