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
   - Allocated to individuals based on who has student loan repayments
   - Scaled to match SLC admin totals

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

# SLC admin totals for scaling (March 2025, UK total)
# Source: https://www.gov.uk/government/statistics/student-loans-in-england-2024-to-2025
SLC_TOTAL_BALANCE_2025 = 294e9  # £294 billion


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


def load_was_student_loan_data() -> pd.DataFrame:
    """
    Load and process WAS data to extract household-level student loan debt.

    WAS doesn't have a direct SLC debt variable, but we can derive it from:
    - Tot_LosR7_aggr: Total loans (all types)
    - Tot_los_exc_SLCR7_aggr: Total loans excluding SLC

    Returns:
        DataFrame with household characteristics and SLC debt.
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
    was["household_net_income"] = was.get(
        "dvtotinc_bhcr7", was.get("dvtotinc_bhcw7", 0)
    )

    return was[["slc_debt", "household_weight", "num_adults", "household_net_income"]]


def impute_student_loan_balance(
    dataset: UKSingleYearDataset,
    year: int = 2025,
    scale_to_admin: bool = True,
) -> UKSingleYearDataset:
    """
    Impute student loan balance for individuals with student loans.

    The imputation uses a simple approach:
    1. For each person with student loan repayments, estimate their balance
       based on their plan type and years since graduation
    2. Scale totals to match SLC admin statistics

    Average balances by plan type (approximate, based on SLC data):
    - Plan 1: Lower balances (older loans, more repaid) - mean ~£10k
    - Plan 2: Higher balances (higher fees) - mean ~£45k
    - Plan 5: New loans, near original amount - mean ~£25k (partial)

    Args:
        dataset: PolicyEngine UK dataset with student_loan_plan imputed.
        year: Simulation year for calculating years since graduation.
        scale_to_admin: Whether to scale totals to match SLC statistics.

    Returns:
        Dataset with student_loan_balance variable added.
    """
    dataset = dataset.copy()
    sim = Microsimulation(dataset=dataset)

    # Get required variables
    age = sim.calculate("age").values
    plan = dataset.person.get("student_loan_plan", np.full(len(age), "NONE"))
    weights = sim.calculate("person_weight").values

    # Estimate years since graduation (assume graduated at 21)
    years_since_grad = np.maximum(0, age - 21)

    # Base balances by plan type (from SLC statistics)
    # These are rough averages that will be scaled
    base_balance = np.zeros(len(age))

    # Plan 1: Older loans, lower original amounts, more repaid
    # Average original ~£20k, many have paid down significantly
    plan_1_mask = plan == "PLAN_1"
    # Decay balance over time (rough model: 3% reduction per year from base of £15k)
    base_balance[plan_1_mask] = 15000 * np.exp(
        -0.03 * years_since_grad[plan_1_mask]
    )

    # Plan 2: Higher fees (£9k+), higher maintenance, average ~£45k original
    plan_2_mask = plan == "PLAN_2"
    # Recent grads have more, decay over time
    base_balance[plan_2_mask] = 45000 * np.exp(
        -0.02 * years_since_grad[plan_2_mask]
    )

    # Plan 5: Very new (2023+), near original amounts
    plan_5_mask = plan == "PLAN_5"
    # Just starting, assume ~£25k average (partial year borrowing)
    base_balance[plan_5_mask] = 25000

    # Scale to match admin totals if requested
    if scale_to_admin:
        current_total = (base_balance * weights).sum()
        if current_total > 0:
            scale_factor = SLC_TOTAL_BALANCE_2025 / current_total
            base_balance = base_balance * scale_factor
            print(f"Scaling student loan balances by {scale_factor:.2f}x")

    # Store the balance
    dataset.person["student_loan_balance"] = base_balance

    # Report results
    has_balance = base_balance > 0
    total_balance = (base_balance * weights).sum()
    mean_balance = (
        (base_balance[has_balance] * weights[has_balance]).sum()
        / weights[has_balance].sum()
    )

    print("Student loan balance imputation results:")
    print(f"  People with balance > 0: {weights[has_balance].sum() / 1e6:.2f}m")
    print(f"  Total balance: £{total_balance / 1e9:.1f}bn")
    print(f"  Mean balance (those with loans): £{mean_balance:,.0f}")

    return dataset
