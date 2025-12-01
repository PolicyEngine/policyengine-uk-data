"""
Student loan imputation.

This module imputes student loan variables:

1. student_loan_plan: Based on reported repayments and estimated university start year
   - NONE: No reported repayments
   - PLAN_1: Started university before September 2012
   - PLAN_2: Started September 2012 - August 2023
   - PLAN_5: Started September 2023 onwards

2. student_loan_balance: Outstanding loan balance imputed from WAS data
   - Uses household-level SLC debt from WAS Round 7
   - Trained QRF model predicts balance based on household characteristics
   - Allocated to individuals based on who has student loan repayments
   - Calibration to admin totals happens in the main calibration step

This enables policyengine-uk's student_loan_repayment variable to calculate
repayments using official threshold parameters, and to cap repayments at
the outstanding balance.
"""

import numpy as np
import pandas as pd
from policyengine_uk.data import UKSingleYearDataset
from policyengine_uk import Microsimulation
from policyengine_uk_data.storage import STORAGE_FOLDER

# WAS Round 7 data location
WAS_TAB_FOLDER = STORAGE_FOLDER / "was_2006_20"

# Predictor variables available in both WAS and FRS (household level)
STUDENT_LOAN_PREDICTORS = [
    "household_net_income",
    "num_adults",
    "num_children",
]

# Region mapping for WAS
REGIONS = {
    1: "NORTH_EAST",
    2: "NORTH_WEST",
    4: "YORKSHIRE",
    5: "EAST_MIDLANDS",
    6: "WEST_MIDLANDS",
    7: "EAST_OF_ENGLAND",
    8: "LONDON",
    9: "SOUTH_EAST",
    10: "SOUTH_WEST",
    11: "WALES",
    12: "SCOTLAND",
}


def impute_student_loan_plan(
    dataset: UKSingleYearDataset,
    year: int = 2025,
) -> UKSingleYearDataset:
    """
    Impute student loan plan type based on age and reported repayments.

    The plan type determines which repayment threshold applies:
    - PLAN_1: £26,065 (2025), pre-Sept 2012 England/Wales
    - PLAN_2: £29,385 (2026-2029 frozen), Sept 2012 - Aug 2023
    - PLAN_4: Scottish loans (not imputed here - requires explicit flag)
    - PLAN_5: £25,000 (2025), Sept 2023 onwards

    Args:
        dataset: PolicyEngine UK dataset with student_loan_repayments.
        year: The simulation year, used to estimate university attendance.

    Returns:
        Dataset with imputed student_loan_plan values.
    """
    dataset = dataset.copy()
    sim = Microsimulation(dataset=dataset)

    # Get required variables
    age = sim.calculate("age").values
    student_loan_repayments = sim.calculate("student_loan_repayments").values

    # Determine if person has a student loan based on reported repayments
    has_student_loan = student_loan_repayments > 0

    # Estimate when they started university (assume age 18)
    # For simulation year Y and age A, university start year = Y - A + 18
    estimated_uni_start_year = year - age + 18

    # Assign plan types based on when loan system changed
    # StudentLoanPlan is a string enum: "NONE", "PLAN_1", "PLAN_2", "PLAN_4", "PLAN_5"
    plan = np.full(len(age), "NONE", dtype=object)

    # Plan 1: Started before September 2012
    plan_1_mask = has_student_loan & (estimated_uni_start_year < 2012)
    plan[plan_1_mask] = "PLAN_1"

    # Plan 2: Started September 2012 - August 2023
    plan_2_mask = has_student_loan & (
        (estimated_uni_start_year >= 2012) & (estimated_uni_start_year < 2023)
    )
    plan[plan_2_mask] = "PLAN_2"

    # Plan 5: Started September 2023 onwards
    plan_5_mask = has_student_loan & (estimated_uni_start_year >= 2023)
    plan[plan_5_mask] = "PLAN_5"

    # Store as the plan type
    dataset.person["student_loan_plan"] = plan

    # Report imputation results
    weights = sim.calculate("person_weight").values
    total_with_loan = (has_student_loan * weights).sum()
    plan_1_count = (plan_1_mask * weights).sum()
    plan_2_count = (plan_2_mask * weights).sum()
    plan_5_count = (plan_5_mask * weights).sum()

    print("Student loan plan imputation results:")
    print(f"  Total with student loan: {total_with_loan / 1e6:.2f}m")
    print(f"  Plan 1 (pre-2012): {plan_1_count / 1e6:.2f}m")
    print(f"  Plan 2 (2012-2023): {plan_2_count / 1e6:.2f}m")
    print(f"  Plan 5 (2023+): {plan_5_count / 1e6:.2f}m")

    return dataset


def generate_was_student_loan_table() -> pd.DataFrame:
    """
    Load and process WAS data for student loan balance imputation.

    WAS doesn't have a direct SLC debt variable, but we can derive it from:
    - Tot_LosR7_aggr: Total loans (all types)
    - Tot_los_exc_SLCR7_aggr: Total loans excluding SLC

    Returns:
        DataFrame with household characteristics and SLC debt for training.
    """
    was = pd.read_csv(
        WAS_TAB_FOLDER / "was_round_7_hhold_eul_march_2022.tab",
        sep="\t",
        low_memory=False,
    )

    # Lowercase all column names for consistency
    was.columns = was.columns.str.lower()

    # Calculate SLC debt as difference between total loans and non-SLC loans
    was["slc_debt"] = was["tot_losr7_aggr"] - was["tot_los_exc_slcr7_aggr"]
    was["slc_debt"] = was["slc_debt"].clip(lower=0)  # Ensure non-negative

    # Get household weight
    was["household_weight"] = was["r7xshhwgt"]

    # Get predictors that match FRS variables
    was["num_adults"] = was.get("numadultw7", was.get("numadultr7", 0))
    was["num_children"] = was.get("numch18w7", was.get("numch18r7", 0))
    was["household_net_income"] = was.get(
        "dvtotinc_bhcr7", was.get("dvtotinc_bhcw7", 0)
    )

    # Fill missing values
    was = was.fillna(0)

    return was[
        ["slc_debt", "household_weight"]
        + STUDENT_LOAN_PREDICTORS
    ]


def save_student_loan_model():
    """
    Train and save the student loan balance imputation model.

    Returns:
        Trained QRF model.
    """
    from policyengine_uk_data.utils.qrf import QRF

    was = generate_was_student_loan_table()

    model = QRF()
    model.fit(
        was[STUDENT_LOAN_PREDICTORS],
        was[["slc_debt"]],
    )
    model.save(STORAGE_FOLDER / "student_loan_balance.pkl")
    return model


def create_student_loan_model(overwrite_existing: bool = False):
    """
    Create or load student loan balance imputation model.

    Args:
        overwrite_existing: Whether to retrain model if it exists.

    Returns:
        QRF model for student loan balance imputation.
    """
    from policyengine_uk_data.utils.qrf import QRF

    model_path = STORAGE_FOLDER / "student_loan_balance.pkl"
    if model_path.exists() and not overwrite_existing:
        return QRF(file_path=model_path)
    return save_student_loan_model()


def impute_student_loan_balance(
    dataset: UKSingleYearDataset,
    year: int = 2025,
) -> UKSingleYearDataset:
    """
    Impute student loan balance for individuals with student loans.

    The imputation uses a QRF model trained on WAS household-level SLC debt:
    1. Predict household-level SLC debt using household characteristics
    2. Allocate to individuals within households who have student loans
    3. Calibration to admin totals happens in the main calibration step

    Args:
        dataset: PolicyEngine UK dataset with student_loan_plan imputed.
        year: Simulation year (currently unused, for future time adjustment).

    Returns:
        Dataset with student_loan_balance variable added.
    """
    dataset = dataset.copy()
    sim = Microsimulation(dataset=dataset)

    # Get the trained model
    model = create_student_loan_model()

    # Get household-level predictors
    input_df = sim.calculate_dataframe(
        STUDENT_LOAN_PREDICTORS, map_to="household"
    )

    # Predict household-level SLC debt
    household_slc_debt = model.predict(input_df)["slc_debt"].values

    # Get person-level data for allocation
    plan = dataset.person.get(
        "student_loan_plan", np.full(len(dataset.person.person_id), "NONE")
    )
    has_student_loan = plan != "NONE"

    # Get household membership
    person_household_id = sim.calculate("household_id").values
    household_ids = dataset.household.household_id.values

    # Create person-to-household index mapping
    household_id_to_idx = {hh_id: idx for idx, hh_id in enumerate(household_ids)}
    person_household_idx = np.array(
        [household_id_to_idx[hh_id] for hh_id in person_household_id]
    )

    # Allocate household debt to individuals with student loans
    # First, count how many people with loans are in each household
    loans_per_household = np.zeros(len(household_ids))
    for person_idx, hh_idx in enumerate(person_household_idx):
        if has_student_loan[person_idx]:
            loans_per_household[hh_idx] += 1

    # Allocate household debt equally among loan holders in that household
    person_balance = np.zeros(len(plan))
    for person_idx, hh_idx in enumerate(person_household_idx):
        if has_student_loan[person_idx] and loans_per_household[hh_idx] > 0:
            person_balance[person_idx] = (
                household_slc_debt[hh_idx] / loans_per_household[hh_idx]
            )

    # Store the balance
    dataset.person["student_loan_balance"] = person_balance

    # Report results
    weights = sim.calculate("person_weight").values
    has_balance = person_balance > 0
    total_balance = (person_balance * weights).sum()

    if weights[has_balance].sum() > 0:
        mean_balance = (
            (person_balance[has_balance] * weights[has_balance]).sum()
            / weights[has_balance].sum()
        )
    else:
        mean_balance = 0

    print("Student loan balance imputation results:")
    print(f"  People with balance > 0: {weights[has_balance].sum() / 1e6:.2f}m")
    print(f"  Total balance: £{total_balance / 1e9:.1f}bn")
    print(f"  Mean balance (those with loans): £{mean_balance:,.0f}")
    print("  Note: Calibration to admin totals happens in main calibration step")

    return dataset
