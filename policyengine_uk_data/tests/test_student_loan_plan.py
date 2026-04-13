"""Tests for student loan plan imputation."""

import numpy as np


def test_repaying_borrowers_are_assigned_expected_plans():
    """Repayers should map to the expected plan cohorts."""
    from policyengine_uk_data.datasets.imputations.student_loans import (
        _impute_student_loan_plan_values,
    )

    plans = _impute_student_loan_plan_values(
        age=np.array([40, 30, 20]),
        student_loan_repayments=np.array([100.0, 100.0, 100.0]),
        country=np.array(["ENGLAND", "ENGLAND", "ENGLAND"]),
        highest_education=np.array(["TERTIARY", "TERTIARY", "TERTIARY"]),
        person_weight=np.ones(3),
        year=2025,
        slc_data={"plan_2": {"liable": {2025: 1}}, "plan_5": {"liable": {2025: 1}}},
    )

    assert plans.tolist() == ["PLAN_1", "PLAN_2", "PLAN_5"]


def test_below_threshold_imputation_uses_liable_shortfall():
    """Missing holders should be imputed from the liable target shortfall."""
    from policyengine_uk_data.datasets.imputations.student_loans import (
        _impute_student_loan_plan_values,
    )

    plans = _impute_student_loan_plan_values(
        age=np.array([40, 30, 20, 30, 30, 30]),
        student_loan_repayments=np.array([100.0, 100.0, 0.0, 0.0, 0.0, 0.0]),
        country=np.array(
            ["ENGLAND", "ENGLAND", "ENGLAND", "ENGLAND", "WALES", "ENGLAND"]
        ),
        highest_education=np.array(
            ["POST_SECONDARY", "TERTIARY", "TERTIARY", "TERTIARY", "TERTIARY", "GCSE"]
        ),
        person_weight=np.ones(6),
        year=2025,
        slc_data={
            "plan_2": {"liable": {2025: 2}},
            "plan_5": {"liable": {2025: 1}},
        },
    )

    assert plans.tolist() == [
        "PLAN_1",
        "PLAN_2",
        "PLAN_5",
        "PLAN_2",
        "NONE",
        "NONE",
    ]


def test_plan5_assignment_has_priority_over_plan2_for_recent_cohort():
    """Recent cohorts should stay on Plan 5 rather than being swallowed by Plan 2."""
    from policyengine_uk_data.datasets.imputations.student_loans import (
        _impute_student_loan_plan_values,
    )

    plans = _impute_student_loan_plan_values(
        age=np.array([21]),
        student_loan_repayments=np.array([0.0]),
        country=np.array(["ENGLAND"]),
        highest_education=np.array(["TERTIARY"]),
        person_weight=np.ones(1),
        year=2026,
        slc_data={
            "plan_2": {"liable": {2026: 1}},
            "plan_5": {"liable": {2026: 1}},
        },
    )

    assert plans.tolist() == ["PLAN_5"]


def test_plan2_below_threshold_imputation_respects_estimated_cohort():
    """Pre-2012 cohorts should not be assigned Plan 2 just because they fit the age band."""
    from policyengine_uk_data.datasets.imputations.student_loans import (
        _impute_student_loan_plan_values,
    )

    plans = _impute_student_loan_plan_values(
        age=np.array([40]),
        student_loan_repayments=np.array([0.0]),
        country=np.array(["ENGLAND"]),
        highest_education=np.array(["TERTIARY"]),
        person_weight=np.ones(1),
        year=2025,
        slc_data={
            "plan_2": {"liable": {2025: 1}},
            "plan_5": {"liable": {2025: 0}},
        },
    )

    assert plans.tolist() == ["NONE"]


def test_student_loan_plan_enum_values():
    """Student-loan plan strings should still match policyengine-uk's enum."""
    from policyengine_uk.variables.gov.hmrc.student_loans.student_loan_plan import (
        StudentLoanPlan,
    )

    assert StudentLoanPlan.NONE.value == "NONE"
    assert StudentLoanPlan.PLAN_1.value == "PLAN_1"
    assert StudentLoanPlan.PLAN_2.value == "PLAN_2"
    assert StudentLoanPlan.PLAN_4.value == "PLAN_4"
    assert StudentLoanPlan.PLAN_5.value == "PLAN_5"
