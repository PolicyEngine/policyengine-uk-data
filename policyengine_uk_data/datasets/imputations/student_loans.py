"""Student loan plan imputation.

This module imputes `student_loan_plan` in two steps:
- assign plans to people with reported PAYE student loan repayments
- assign missing below-threshold holders to match SLC liable-to-repay totals

The FRS only observes active repayment through PAYE, so many England borrowers
who hold a loan but earn below the repayment threshold are missing from the
base dataset. We fill that stock using the checked-in SLC snapshot, restricting
the new assignments to plausible England tertiary-education cohorts.
"""

import numpy as np
from policyengine_uk import Microsimulation
from policyengine_uk.data import UKSingleYearDataset

from policyengine_uk_data.targets.sources.slc import get_snapshot_data

_ENGLAND = "ENGLAND"
_PLAN_2_MIN_AGE = 21
_PLAN_2_MAX_AGE = 55
_PLAN_5_MAX_AGE = 25


def _weighted_count(mask: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sum(weights[mask]))


def _assign_probabilistically(
    plan: np.ndarray,
    eligible: np.ndarray,
    weights: np.ndarray,
    target_count: float,
    plan_name: str,
    rng: np.random.Generator,
) -> None:
    """Assign a plan to a weighted eligible pool up to a target count."""
    eligible_weight = _weighted_count(eligible, weights)
    if target_count <= 0 or eligible_weight <= 0:
        return
    assignment_probability = min(1.0, target_count / eligible_weight)
    draws = rng.random(len(plan))
    plan[eligible & (draws < assignment_probability)] = plan_name


def _impute_student_loan_plan_values(
    age: np.ndarray,
    student_loan_repayments: np.ndarray,
    country: np.ndarray,
    highest_education: np.ndarray,
    person_weight: np.ndarray,
    *,
    year: int,
    seed: int = 42,
    slc_data: dict | None = None,
) -> np.ndarray:
    """Impute plan values from person-level arrays."""
    age = np.asarray(age)
    repayments = np.asarray(student_loan_repayments)
    country = np.asarray(country)
    highest_education = np.asarray(highest_education)
    person_weight = np.asarray(person_weight, dtype=float)
    slc_data = get_snapshot_data() if slc_data is None else slc_data

    rng = np.random.default_rng(seed)
    plan = np.full(len(age), "NONE", dtype=object)

    has_repayments = repayments > 0
    is_england = country == _ENGLAND
    is_tertiary = highest_education == "TERTIARY"
    estimated_uni_start_year = year - age + 18

    plan_1_cohort = estimated_uni_start_year < 2012
    plan_5_cohort = estimated_uni_start_year >= 2023
    plan_2_age_band = (age >= _PLAN_2_MIN_AGE) & (age <= _PLAN_2_MAX_AGE)
    plan_5_age_band = (age >= 18) & (age <= _PLAN_5_MAX_AGE)

    # Reported PAYE repayers identify the active stock directly.
    plan[has_repayments & plan_1_cohort] = "PLAN_1"
    plan[has_repayments & plan_5_cohort] = "PLAN_5"
    plan[has_repayments & (plan == "NONE")] = "PLAN_2"

    # Impute missing below-threshold holders so the total England stock matches
    # the SLC liable-to-repay series, using the observed repayer stock as the
    # starting point rather than the official above-threshold count.
    plan_5_target = slc_data["plan_5"]["liable"].get(year, 0)
    plan_5_shortfall = max(
        0.0,
        plan_5_target - _weighted_count((plan == "PLAN_5") & is_england, person_weight),
    )
    plan_5_eligible = (
        (plan == "NONE") & is_england & is_tertiary & plan_5_age_band & plan_5_cohort
    )
    _assign_probabilistically(
        plan,
        plan_5_eligible,
        person_weight,
        plan_5_shortfall,
        "PLAN_5",
        rng,
    )

    plan_2_target = slc_data["plan_2"]["liable"].get(year, 0)
    plan_2_shortfall = max(
        0.0,
        plan_2_target - _weighted_count((plan == "PLAN_2") & is_england, person_weight),
    )
    plan_2_eligible = (plan == "NONE") & is_england & is_tertiary & plan_2_age_band
    _assign_probabilistically(
        plan,
        plan_2_eligible,
        person_weight,
        plan_2_shortfall,
        "PLAN_2",
        rng,
    )

    return plan


def impute_student_loan_plan(
    dataset: UKSingleYearDataset,
    year: int = 2025,
    seed: int = 42,
    slc_data: dict | None = None,
) -> UKSingleYearDataset:
    """
    Impute student loan plan type based on age and reported repayments.

    The plan type determines which repayment threshold applies:
    - PLAN_1: £26,065 (2025), pre-Sept 2012 England/Wales
    - PLAN_2: £29,385 (2026-2029 frozen), Sept 2012 - Aug 2023
    - PLAN_4: Scottish loans (not imputed here - requires explicit flag)
    - PLAN_5: £25,000 (2025), Sept 2023 onwards

    Args:
        dataset: PolicyEngine UK dataset with student loan inputs.
        year: Simulation year, used to estimate university start cohorts.
        seed: Random seed for reproducible below-threshold assignment.
        slc_data: Optional override for the SLC borrower snapshot.
    """
    dataset = dataset.copy()
    sim = Microsimulation(dataset=dataset)
    dataset.person["student_loan_plan"] = _impute_student_loan_plan_values(
        age=sim.calculate("age").values,
        student_loan_repayments=sim.calculate("student_loan_repayments").values,
        country=sim.calculate("country", map_to="person").values,
        highest_education=sim.calculate("highest_education").values,
        person_weight=sim.calculate("person_weight").values,
        year=year,
        seed=seed,
        slc_data=slc_data,
    )

    return dataset
