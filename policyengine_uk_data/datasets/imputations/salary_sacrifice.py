"""
Salary sacrifice imputation for pension contributions.

This module imputes salary sacrifice participation and amounts using machine
learning models trained on FRS respondents who were asked the SALSAC question.
The FRS under-reports salary sacrifice usage (only ~1% of jobs have non-zero
SPNAMT), while HMRC data indicates ~30% of private sector employees use it.

Training data:
- SALSAC='1' (Yes, uses salary sacrifice): 224 jobs
- SALSAC='2' (No, doesn't use): 3,803 jobs

Imputation candidates:
- SALSAC=' ' (skip/not asked): 13,265 jobs

External validation targets (HMRC Table 6.2, 2023-24):
- Total SS pension contributions: ~24bn
- IT relief from SS: ~7.2bn
- Participation rate: ~30% of private sector employees
"""

import pandas as pd
import numpy as np
from policyengine_uk_data.storage import STORAGE_FOLDER
from policyengine_uk.data import UKSingleYearDataset
from policyengine_uk import Microsimulation


# Target participation rate from HMRC surveys
TARGET_SS_PARTICIPATION_RATE = 0.30  # 30% of private sector employees

# HMRC Table 6.2 targets for 2023-24 (uprated to 2025 using growth factors)
SS_CONTRIBUTIONS_TARGET_2025 = 24e9  # 24bn in SS pension contributions


def create_salary_sacrifice_model(overwrite_existing: bool = False):
    """
    Create or load salary sacrifice participation model.

    Uses a logistic-style probability model based on employee characteristics
    to predict probability of salary sacrifice participation.

    Args:
        overwrite_existing: Whether to retrain model if it exists.

    Returns:
        Trained model for salary sacrifice imputation.
    """
    from policyengine_uk_data.utils.qrf import QRF

    model_path = STORAGE_FOLDER / "salary_sacrifice.pkl"
    if model_path.exists() and not overwrite_existing:
        return QRF(file_path=model_path)
    return save_salary_sacrifice_model()


def save_salary_sacrifice_model():
    """
    Train and save salary sacrifice imputation model.

    Uses FRS data where SALSAC was answered to train a QRF model
    predicting salary sacrifice participation probability.

    Returns:
        Trained QRF model.
    """
    from policyengine_uk_data.utils.qrf import QRF

    # Load the raw FRS dataset (before imputation)
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
    gender = sim.calculate("gender").values
    employment_income = sim.calculate("employment_income").values
    region = sim.calculate("region").values

    # Get salary sacrifice indicators from dataset
    ss_reported = dataset.person.salary_sacrifice_reported.values
    ss_asked = dataset.person.salary_sacrifice_asked.values

    # Build training DataFrame with only those who were asked
    training_mask = ss_asked == 1

    train_df = pd.DataFrame(
        {
            "age": age[training_mask],
            "gender": gender[training_mask],
            "employment_income": employment_income[training_mask],
            "region": region[training_mask],
            "uses_salary_sacrifice": ss_reported[training_mask],
        }
    )

    # Train QRF model
    model = QRF()
    predictors = ["age", "gender", "employment_income", "region"]
    targets = ["uses_salary_sacrifice"]

    model.fit(train_df[predictors], train_df[targets])
    model.save(STORAGE_FOLDER / "salary_sacrifice.pkl")

    return model


def impute_salary_sacrifice(
    dataset: UKSingleYearDataset,
) -> UKSingleYearDataset:
    """
    Impute salary sacrifice participation and amounts.

    For respondents not asked about salary sacrifice, uses ML model to
    predict participation probability, then assigns SS contributions
    proportional to their pension contributions.

    The imputation targets ~30% participation among employees with
    pension contributions, per HMRC survey data.

    Args:
        dataset: PolicyEngine UK dataset with salary_sacrifice_reported
            and salary_sacrifice_asked fields from FRS processing.

    Returns:
        Dataset with imputed salary sacrifice contributions.
    """
    dataset = dataset.copy()
    sim = Microsimulation(dataset=dataset)

    # Get current values
    age = sim.calculate("age").values
    gender = sim.calculate("gender").values
    employment_income = sim.calculate("employment_income").values
    region = sim.calculate("region").values
    employee_pension = sim.calculate("employee_pension_contributions").values
    current_ss = dataset.person.pension_contributions_via_salary_sacrifice

    # Get indicators
    if "salary_sacrifice_reported" in dataset.person.columns:
        ss_reported = dataset.person.salary_sacrifice_reported.values
        ss_asked = dataset.person.salary_sacrifice_asked.values
    else:
        # If not available, treat all as not asked (impute for everyone)
        ss_reported = np.zeros(len(age))
        ss_asked = np.zeros(len(age))

    # Identify imputation candidates: employed adults not asked about SS
    # who have pension contributions
    is_employee = employment_income > 0
    has_pension = employee_pension > 0
    not_asked = ss_asked == 0
    imputation_candidates = is_employee & has_pension & not_asked

    # Create prediction DataFrame
    pred_df = pd.DataFrame(
        {
            "age": age,
            "gender": gender,
            "employment_income": employment_income,
            "region": region,
        }
    )

    # Get or train model
    try:
        model = create_salary_sacrifice_model()
        # Predict participation probability
        predictions = model.predict(pred_df)
        ss_probability = predictions["uses_salary_sacrifice"].values
    except (FileNotFoundError, KeyError):
        # If model not available, use simple heuristic based on income
        # Higher earners more likely to use SS
        income_percentile = np.clip(
            employment_income / employment_income.max(), 0, 1
        )
        ss_probability = income_percentile * TARGET_SS_PARTICIPATION_RATE * 2

    # Apply participation probability to candidates
    np.random.seed(42)  # Reproducibility
    random_draw = np.random.random(len(age))

    # Assign SS participation based on probability
    new_ss_participant = imputation_candidates & (random_draw < ss_probability)

    # For new participants, convert portion of employee pension to SS
    # Typical SS arrangements convert 100% of employee contributions
    # plus some additional employer contribution
    ss_contribution_rate = 1.0  # Convert 100% of employee pension

    new_ss_amounts = np.where(
        new_ss_participant,
        employee_pension * ss_contribution_rate,
        0,
    )

    # Combine with existing SS amounts (don't override reported values)
    final_ss = np.where(
        ss_reported == 1,
        current_ss.values,  # Keep reported values
        np.maximum(current_ss.values, new_ss_amounts),  # Add imputed
    )

    # Update dataset
    dataset.person["pension_contributions_via_salary_sacrifice"] = final_ss

    # Validate against targets
    weights = sim.calculate("person_weight").values
    total_ss = (final_ss * weights).sum()
    participation_rate = (
        (final_ss > 0) * weights * (employment_income > 0)
    ).sum() / (weights * (employment_income > 0)).sum()

    print(f"Salary sacrifice imputation results:")
    print(f"  Total SS contributions: {total_ss / 1e9:.1f}bn")
    print(f"  Target: ~{SS_CONTRIBUTIONS_TARGET_2025 / 1e9:.0f}bn")
    print(f"  Employee participation rate: {participation_rate:.1%}")
    print(f"  Target rate: ~{TARGET_SS_PARTICIPATION_RATE:.0%}")

    return dataset
