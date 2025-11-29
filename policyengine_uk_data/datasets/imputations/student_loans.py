"""
Student loan plan imputation.

This module imputes the student_loan_plan variable based on:
- Whether the person has reported student loan repayments
- Their estimated university attendance year (inferred from age)

The imputation assigns plan types according to when the loan system changed:
- NONE: No reported repayments
- PLAN_1: Started university before September 2012
- PLAN_2: Started September 2012 - August 2023
- PLAN_5: Started September 2023 onwards

This enables policyengine-uk's student_loan_repayment variable to calculate
repayments using official threshold parameters.
"""

import numpy as np
from policyengine_uk.data import UKSingleYearDataset
from policyengine_uk import Microsimulation


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
