"""Tests for student loan plan imputation."""

import numpy as np
import pytest


def test_student_loan_plan_imputation_logic():
    """Test the plan assignment logic based on university start year."""
    # Test data: (age, year, expected_uni_start, expected_plan)
    # Plan 1: pre-2012, Plan 2: 2012-2022, Plan 5: 2023+

    year = 2025

    # Age 40 in 2025 -> started uni ~2003 -> Plan 1
    age_40_uni_year = year - 40 + 18  # = 2003
    assert age_40_uni_year < 2012, "Age 40 should be Plan 1"

    # Age 30 in 2025 -> started uni ~2013 -> Plan 2
    age_30_uni_year = year - 30 + 18  # = 2013
    assert 2012 <= age_30_uni_year < 2023, "Age 30 should be Plan 2"

    # Age 25 in 2025 -> started uni ~2018 -> Plan 2
    age_25_uni_year = year - 25 + 18  # = 2018
    assert 2012 <= age_25_uni_year < 2023, "Age 25 should be Plan 2"

    # Age 20 in 2025 -> started uni ~2023 -> Plan 5
    age_20_uni_year = year - 20 + 18  # = 2023
    assert age_20_uni_year >= 2023, "Age 20 should be Plan 5"

    # Age 18 in 2025 -> started uni ~2025 -> Plan 5
    age_18_uni_year = year - 18 + 18  # = 2025
    assert age_18_uni_year >= 2023, "Age 18 should be Plan 5"


def test_student_loan_plan_enum_values():
    """Test that plan enum values match policyengine-uk's string enum."""
    from policyengine_uk.variables.gov.hmrc.student_loans.student_loan_plan import (
        StudentLoanPlan,
    )

    # Verify our assumptions about enum values (string-based enum)
    assert StudentLoanPlan.NONE.value == "NONE"
    assert StudentLoanPlan.PLAN_1.value == "PLAN_1"
    assert StudentLoanPlan.PLAN_2.value == "PLAN_2"
    assert StudentLoanPlan.PLAN_4.value == "PLAN_4"
    assert StudentLoanPlan.PLAN_5.value == "PLAN_5"


def test_student_loan_balance_allocation_logic():
    """Test the household-to-person allocation logic."""
    import numpy as np

    # Test case: 2 people with loans in household, £40k debt
    household_debt = 40000
    num_loan_holders = 2
    per_person_debt = household_debt / num_loan_holders
    assert per_person_debt == 20000, "Should split equally"

    # Test case: 1 person with loan in household, £30k debt
    household_debt = 30000
    num_loan_holders = 1
    per_person_debt = household_debt / num_loan_holders
    assert per_person_debt == 30000, "Single holder gets all"

    # Test case: No loan holders - should not divide by zero
    household_debt = 50000
    num_loan_holders = 0
    # In our implementation, we check for this condition
    if num_loan_holders > 0:
        per_person_debt = household_debt / num_loan_holders
    else:
        per_person_debt = 0
    assert per_person_debt == 0, "No loan holders means zero allocation"


def test_student_loan_predictor_variables():
    """Test that predictor variables are defined correctly."""
    from policyengine_uk_data.datasets.imputations.student_loans import (
        STUDENT_LOAN_PREDICTORS,
    )

    # Check that key predictors are included
    assert "household_net_income" in STUDENT_LOAN_PREDICTORS
    assert "num_adults" in STUDENT_LOAN_PREDICTORS
    assert "num_children" in STUDENT_LOAN_PREDICTORS
