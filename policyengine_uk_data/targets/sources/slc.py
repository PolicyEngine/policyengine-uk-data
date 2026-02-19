"""Student Loans Company (SLC) calibration targets.

Borrower counts for England only: Plan 2 and Plan 5, restricted to
borrowers liable to repay and earning above the repayment threshold.
This matches the FRS coverage (PAYE deductions only).

Source: Explore Education Statistics — Student loan forecasts for England,
Table 6a: Forecast number of student borrowers liable to repay and number
earning above repayment threshold, by product. Figures are the sum of
higher education full-time, higher education part-time, and advanced
learner loan borrowers (Master's and Doctoral loans use Plan 3 and are
excluded). Academic year 20XX-YY maps to calendar year 20XX.

Data permalink:
https://explore-education-statistics.service.gov.uk/data-tables/permalink/6ff75517-7124-487c-cb4e-08de6eccf22d
"""

from policyengine_uk_data.targets.schema import Target, Unit

_REFERENCE = (
    "https://explore-education-statistics.service.gov.uk/data-tables"
    "/permalink/6ff75517-7124-487c-cb4e-08de6eccf22d"
)

# Plan 2, earning above threshold — sum of HE full-time + part-time + AL
# 2024-25: 3,670k + 225k + 90k = 3,985k
# 2025-26: 4,130k + 245k + 85k = 4,460k
# 2026-27: 4,480k + 260k + 85k = 4,825k
# 2027-28: 4,700k + 265k + 80k = 5,045k
# 2028-29: 4,820k + 265k + 70k = 5,155k
# 2029-30: 4,870k + 270k + 65k = 5,205k
_PLAN2_ABOVE_THRESHOLD = {
    2025: 3_985_000,
    2026: 4_460_000,
    2027: 4_825_000,
    2028: 5_045_000,
    2029: 5_155_000,
    2030: 5_205_000,
}

# Plan 5, earning above threshold — sum of HE full-time + part-time + AL
# 2024-25: 0 + 0 + 0 = 0
# 2025-26: 25k + 5k + 5k = 35k
# 2026-27: 115k + 20k + 10k = 145k
# 2027-28: 340k + 35k + 15k = 390k
# 2028-29: 700k + 50k + 15k = 765k
# 2029-30: 1,140k + 75k + 20k = 1,235k
_PLAN5_ABOVE_THRESHOLD = {
    2026: 35_000,
    2027: 145_000,
    2028: 390_000,
    2029: 765_000,
    2030: 1_235_000,
}


def get_targets() -> list[Target]:
    targets = []

    targets.append(
        Target(
            name="slc/plan_2_borrowers_above_threshold",
            variable="student_loan_plan",
            source="slc",
            unit=Unit.COUNT,
            is_count=True,
            values=_PLAN2_ABOVE_THRESHOLD,
            reference_url=_REFERENCE,
        )
    )

    targets.append(
        Target(
            name="slc/plan_5_borrowers_above_threshold",
            variable="student_loan_plan",
            source="slc",
            unit=Unit.COUNT,
            is_count=True,
            values=_PLAN5_ABOVE_THRESHOLD,
            reference_url=_REFERENCE,
        )
    )

    return targets
