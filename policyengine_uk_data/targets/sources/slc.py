"""Student Loans Company (SLC) calibration targets.

Borrower counts for England only: Plan 2 and Plan 5, restricted to
borrowers liable to repay and earning above the repayment threshold.
This matches the FRS coverage (PAYE deductions only).

Source: SLC 'Student loans: borrower liability and repayment' statistical
release, Table 6a — Forecast number of student borrowers liable to repay
and number earning above repayment threshold.
https://www.gov.uk/government/collections/student-loans-in-england-statistics
"""

from policyengine_uk_data.targets.schema import Target, Unit


def get_targets() -> list[Target]:
    targets = []

    _REFERENCE = (
        "https://www.gov.uk/government/collections/"
        "student-loans-in-england-statistics"
    )

    # Plan 2 — England, earning above threshold
    # Academic year 20XX-YY maps to calendar year 20XX.
    targets.append(
        Target(
            name="slc/plan_2_borrowers_above_threshold",
            variable="student_loan_plan",
            source="slc",
            unit=Unit.COUNT,
            is_count=True,
            values={
                2025: 3_985_000,
                2026: 4_460_000,
                2027: 4_825_000,
                2028: 5_045_000,
                2029: 5_160_000,
                2030: 5_205_000,
            },
            reference_url=_REFERENCE,
        )
    )

    # Plan 5 — England, earning above threshold
    targets.append(
        Target(
            name="slc/plan_5_borrowers_above_threshold",
            variable="student_loan_plan",
            source="slc",
            unit=Unit.COUNT,
            is_count=True,
            values={
                2026: 35_000,
                2027: 145_000,
                2028: 390_000,
                2029: 770_000,
                2030: 1_235_000,
            },
            reference_url=_REFERENCE,
        )
    )

    return targets
