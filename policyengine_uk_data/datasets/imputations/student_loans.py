"""
Student loan plan imputation.

This module imputes the student_loan_plan variable based on:
- Whether the person has reported student loan repayments (above threshold)
- Their estimated university attendance year (inferred from age)
- Probabilistic assignment for below-threshold borrowers

The imputation assigns plan types according to when the loan system changed:
- NONE: No loan
- PLAN_1: Started university before September 2012
- PLAN_2: Started September 2012 - August 2023
- PLAN_5: Started September 2023 onwards

The FRS only records active repayers (via PAYE). SLC data shows many borrowers
earn below repayment thresholds (~55% of Plan 2 holders). This imputation
fills that gap by probabilistically assigning plans to people in the relevant
age cohort without reported repayments, based on SLC "liable to repay" minus
"above threshold" counts.
"""

import numpy as np
from policyengine_uk.data import UKSingleYearDataset
from policyengine_uk import Microsimulation

# England regions for filtering (SLC data covers England only)
_ENGLAND_REGIONS = {
    "NORTH_EAST",
    "NORTH_WEST",
    "YORKSHIRE",
    "EAST_MIDLANDS",
    "WEST_MIDLANDS",
    "EAST_OF_ENGLAND",
    "LONDON",
    "SOUTH_EAST",
    "SOUTH_WEST",
}

# SLC liable-to-repay counts (Higher Education total, England)
# Source: https://explore-education-statistics.service.gov.uk/data-tables/permalink/6ff75517-7124-487c-cb4e-08de6eccf22d
_PLAN_2_LIABLE = {
    2025: 8_940_000,
    2026: 9_710_000,
    2027: 10_360_000,
    2028: 10_615_000,
    2029: 10_600_000,
    2030: 10_525_000,
}

_PLAN_5_LIABLE = {
    2025: 10_000,
    2026: 230_000,
    2027: 630_000,
    2028: 1_380_000,
    2029: 2_360_000,
    2030: 3_400_000,
}

# SLC above-threshold counts (borrowers making repayments)
_PLAN_2_ABOVE_THRESHOLD = {
    2025: 3_985_000,
    2026: 4_460_000,
    2027: 4_825_000,
    2028: 5_045_000,
    2029: 5_160_000,
    2030: 5_205_000,
}

_PLAN_5_ABOVE_THRESHOLD = {
    2026: 35_000,
    2027: 145_000,
    2028: 390_000,
    2029: 770_000,
    2030: 1_235_000,
}


def impute_student_loan_plan(
    dataset: UKSingleYearDataset,
    year: int = 2025,
    seed: int = 42,
) -> UKSingleYearDataset:
    """
    Impute student loan plan type based on age, repayments, and education.

    The plan type determines which repayment threshold applies:
    - PLAN_1: £26,065 (2025), pre-Sept 2012 England/Wales
    - PLAN_2: £29,385 (2026-2029 frozen), Sept 2012 - Aug 2023
    - PLAN_4: Scottish loans (not imputed here - requires explicit flag)
    - PLAN_5: £25,000 (2025), Sept 2023 onwards

    This function:
    1. Assigns plans to people with reported repayments (above threshold)
    2. Probabilistically assigns plans to tertiary-educated people without
       repayments (below threshold) to match SLC liable-to-repay totals

    Args:
        dataset: PolicyEngine UK dataset with student_loan_repayments.
        year: The simulation year, used to estimate university attendance.
        seed: Random seed for reproducibility.

    Returns:
        Dataset with imputed student_loan_plan values.
    """
    dataset = dataset.copy()
    sim = Microsimulation(dataset=dataset)
    rng = np.random.default_rng(seed)

    age = sim.calculate("age").values
    repayments = sim.calculate("student_loan_repayments").values
    region = sim.calculate("region", map_to="person").values
    weights = sim.calculate("person_weight").values

    is_england = np.isin(region, list(_ENGLAND_REGIONS))
    has_repayments = repayments > 0

    # Estimate university start year (assume started at 18)
    uni_start_year = year - age + 18

    # Age bands for plausible loan holders (graduates typically 21+)
    # Plan 1: 32+ (started before 2012, graduated 21+ by 2015)
    # Plan 2: 21+ and cohort 2012-2022
    # Plan 5: 21+ and cohort 2023+ (but in early years, recent grads are 18-22)
    plan_1_age_mask = age >= 32
    plan_2_age_mask = age >= 21
    # Plan 5: use cohort constraint only since graduates are very young in early years
    plan_5_age_mask = age >= 18  # Anyone 18+ who started 2023+ could have a loan

    # Cohort masks based on university start year
    plan_1_cohort = uni_start_year < 2012
    plan_2_cohort = (uni_start_year >= 2012) & (uni_start_year < 2023)
    plan_5_cohort = uni_start_year >= 2023

    plan = np.full(len(age), "NONE", dtype=object)

    # Step 1: Assign plans to people with reported repayments
    plan[has_repayments & plan_1_cohort] = "PLAN_1"
    plan[has_repayments & plan_2_cohort] = "PLAN_2"
    plan[has_repayments & plan_5_cohort] = "PLAN_5"

    # Step 2: Probabilistically assign below-threshold borrowers
    # Only for tertiary-educated people in England without repayments
    no_repayments = ~has_repayments

    # Calculate target below-threshold counts
    plan_2_below = _PLAN_2_LIABLE.get(year, 0) - _PLAN_2_ABOVE_THRESHOLD.get(
        year, 0
    )
    plan_5_below = _PLAN_5_LIABLE.get(year, 0) - _PLAN_5_ABOVE_THRESHOLD.get(
        year, 0
    )

    # Plan 2 below-threshold assignment
    # No tertiary filter - SLC data shows ~94% of cohort has loans
    plan_2_eligible = (
        no_repayments
        & is_england
        & plan_2_age_mask
        & plan_2_cohort
    )
    if plan_2_below > 0 and plan_2_eligible.sum() > 0:
        eligible_weight = (weights * plan_2_eligible).sum()
        if eligible_weight > 0:
            prob = min(1.0, plan_2_below / eligible_weight)
            draws = rng.random(len(age))
            plan[plan_2_eligible & (draws < prob)] = "PLAN_2"

    # Plan 5 below-threshold assignment
    plan_5_eligible = (
        no_repayments
        & is_england
        & plan_5_age_mask
        & plan_5_cohort
    )
    if plan_5_below > 0 and plan_5_eligible.sum() > 0:
        eligible_weight = (weights * plan_5_eligible).sum()
        if eligible_weight > 0:
            prob = min(1.0, plan_5_below / eligible_weight)
            draws = rng.random(len(age))
            plan[plan_5_eligible & (draws < prob)] = "PLAN_5"

    dataset.person["student_loan_plan"] = plan

    return dataset
