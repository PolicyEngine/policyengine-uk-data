"""
Salary sacrifice imputation for pension contributions.

Two-stage imputation:

1. QRF trained on FRS respondents who were asked SALSAC (~224 yes,
   ~3,803 no). Predicts SS amounts for ~13,265 jobs where SALSAC was
   not asked.

2. Headcount-targeted imputation: converts a fraction of pension
   contributors without SS into below-cap (≤£2,000) SS users, moving
   employee pension contributions to salary sacrifice. Targets the
   OBR/ASHE estimate of ~4.3mn below-cap SS users.

Exact monetary totals (~£24bn SS contributions) and final headcount
calibration happen via weight optimisation in a subsequent step.
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

    Stage 1: QRF predicts SS amounts for respondents not asked SALSAC.
    Stage 2: Converts a fraction of pension contributors to below-cap
    SS users, targeting ~4.3mn (OBR/ASHE). Moves employee pension
    contributions to salary sacrifice to keep total pension consistent.

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

    # Stage 2: Headcount-targeted imputation for below-cap SS users.
    # ASHE data shows many more SS users than the FRS captures due to
    # self-reporting bias in auto-enrolment. Impute additional SS users
    # from pension contributors to create enough records for calibration
    # to hit OBR headcount targets (7.7mn total, 4.3mn below £2,000).
    person_weight = sim.calculate("person_weight").values
    employee_pension = dataset.person[
        "employee_pension_contributions"
    ].values.copy()
    has_ss = final_ss > 0
    below_cap_ss = has_ss & (final_ss <= 2000)

    # Donor pool: employed pension contributors not already SS users
    is_donor = (employee_pension > 0) & ~has_ss & (employment_income > 0)

    # Target ~4.3mn below-cap SS users (HMRC/ASHE estimate)
    TARGET_BELOW_CAP = 4_300_000
    current_below_cap = (person_weight * below_cap_ss).sum()
    shortfall = max(0, TARGET_BELOW_CAP - current_below_cap)

    if shortfall > 0:
        donor_weighted = (person_weight * is_donor).sum()
        if donor_weighted > 0:
            imputation_rate = min(0.8, shortfall / donor_weighted)
            rng = np.random.default_rng(seed=2024)
            newly_imputed = is_donor & (
                rng.random(len(final_ss)) < imputation_rate
            )

            # Move up to £2,000 of employee pension to SS
            ss_new = np.minimum(employee_pension, 2000.0)
            final_ss = np.where(newly_imputed, ss_new, final_ss)

            # Reduce employee pension correspondingly
            dataset.person["employee_pension_contributions"] = np.where(
                newly_imputed,
                employee_pension - ss_new,
                employee_pension,
            )

    dataset.person["pension_contributions_via_salary_sacrifice"] = final_ss

    return dataset
