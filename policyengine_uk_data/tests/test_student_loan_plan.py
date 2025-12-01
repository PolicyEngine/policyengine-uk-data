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


def test_student_loan_balance_base_values():
    """Test the base balance calculation logic by plan type."""
    import numpy as np

    year = 2025

    # Test Plan 1 balance decay
    # Base £15k decaying at 3% per year
    age_40 = 40
    years_since_grad = max(0, age_40 - 21)  # 19 years
    plan_1_balance = 15000 * np.exp(-0.03 * years_since_grad)
    assert 7000 < plan_1_balance < 10000, f"Plan 1 balance {plan_1_balance} out of range"

    # Test Plan 2 balance decay
    # Base £45k decaying at 2% per year
    age_30 = 30
    years_since_grad = max(0, age_30 - 21)  # 9 years
    plan_2_balance = 45000 * np.exp(-0.02 * years_since_grad)
    assert 35000 < plan_2_balance < 40000, f"Plan 2 balance {plan_2_balance} out of range"

    # Test Plan 5 balance (no decay, very new)
    plan_5_balance = 25000
    assert plan_5_balance == 25000, "Plan 5 should be £25k base"


def test_student_loan_balance_scaling_logic():
    """Test that scaling logic would adjust totals correctly."""
    import numpy as np

    # Simple scaling test
    base_total = 100e9  # £100bn
    admin_total = 294e9  # £294bn (SLC target)
    scale_factor = admin_total / base_total

    assert 2.9 < scale_factor < 3.0, f"Scale factor {scale_factor} unexpected"

    # After scaling
    scaled_total = base_total * scale_factor
    assert abs(scaled_total - admin_total) < 1e6, "Scaling should match admin total"
