"""
Salary sacrifice imputation for pension contributions.

This module imputes salary sacrifice pension amounts using QRF trained on
FRS respondents who were asked the SALSAC question. The model predicts
the continuous amount (pension_contributions_via_salary_sacrifice), with
non-participants naturally having 0.

Training data (FRS 2023-24):
- SALSAC='1' (Yes): ~224 jobs with reported SPNAMT amounts
- SALSAC='2' (No): ~3,803 jobs with SPNAMT=0

Imputation candidates:
- SALSAC=' ' (skip/not asked): ~13,265 jobs

Targeting to HMRC totals (~24bn SS contributions) happens via weight
calibration, not in this imputation step.
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
    "pension_contributions_via_salary_sacrifice",
]


def save_salary_sacrifice_model():
    """
    Train and save salary sacrifice imputation model using FRS data.

    Uses FRS respondents who were asked about salary sacrifice (SALSAC field)
    as training data. The model learns to predict the SS pension amount
    directly - non-participants have 0, participants have their reported
    SPNAMT value.

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

    # Get SS amounts and indicator for who was asked
    ss_amount = (
        dataset.person.pension_contributions_via_salary_sacrifice.values
    )
    if "salary_sacrifice_asked" not in dataset.person.columns:
        raise ValueError(
            "Dataset missing salary_sacrifice_asked field. "
            "Ensure frs.py extracts SALSAC before numeric conversion."
        )
    ss_asked = dataset.person.salary_sacrifice_asked.values

    # Build training DataFrame with only those who were asked
    # This includes both participants (with amounts) and non-participants (0)
    training_mask = ss_asked == 1

    if training_mask.sum() == 0:
        raise ValueError(
            "No training data found - no respondents were asked SALSAC."
        )

    train_df = pd.DataFrame(
        {
            "age": age[training_mask],
            "employment_income": employment_income[training_mask],
            "pension_contributions_via_salary_sacrifice": ss_amount[
                training_mask
            ],
        }
    )

    # Train QRF model
    model = QRF()
    model.fit(train_df[PREDICTORS], train_df[IMPUTATIONS])
    model.save(STORAGE_FOLDER / "salary_sacrifice.pkl")

    return model


def create_salary_sacrifice_model(overwrite_existing: bool = False):
    """
    Create or load salary sacrifice imputation model.

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
    Impute salary sacrifice pension amounts for FRS non-respondents.

    For respondents not asked about salary sacrifice (SALSAC=' '), uses
    a QRF model trained on those who were asked to predict the SS pension
    contribution amount directly. The model naturally predicts 0 for
    non-participants and positive amounts for likely participants.

    Note: This imputation does NOT target any specific total. Targeting
    to HMRC figures happens via weight calibration in a subsequent step.

    Args:
        dataset: PolicyEngine UK dataset with salary_sacrifice_asked
            field from FRS processing.

    Returns:
        Dataset with imputed salary sacrifice amounts.
    """
    dataset = dataset.copy()
    sim = Microsimulation(dataset=dataset)

    # Get variables needed for imputation
    age = sim.calculate("age").values
    employment_income = sim.calculate("employment_income").values
    current_ss = (
        dataset.person.pension_contributions_via_salary_sacrifice.values
    )

    # Get indicator for who was asked
    if "salary_sacrifice_asked" not in dataset.person.columns:
        return dataset

    ss_asked = dataset.person.salary_sacrifice_asked.values

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

    # Get imputed amounts (QRF predicts continuous values)
    imputed_ss = predictions[
        "pension_contributions_via_salary_sacrifice"
    ].values

    # Ensure non-negative
    imputed_ss = np.maximum(0, imputed_ss)

    # For those who were asked, keep their reported values
    # For those not asked, use the imputed values
    final_ss = np.where(
        ss_asked == 1,
        current_ss,  # Keep reported values exactly
        imputed_ss,  # Use imputed for non-respondents
    )

    # Update dataset
    dataset.person["pension_contributions_via_salary_sacrifice"] = final_ss

    return dataset
