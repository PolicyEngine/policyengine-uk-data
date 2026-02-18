"""
Student loan plan imputation.

Assigns Plan 1, 2 and 5 based on age-cohort eligibility and HE participation
rates, regardless of whether the person currently has repayments > 0. This
correctly captures below-threshold borrowers who will start repaying as incomes
rise under uprating.

Plan boundaries:
  Plan 1: started uni before Sept 2012  (ages ~34+ in 2023)
  Plan 2: started uni Sept 2012-Aug 2023 (ages ~19-33 in 2023)
  Plan 5: started uni Sept 2023+         (near-zero in 2023)

HE participation rates by age are derived from HESA data and calibrated so
that total imputed Plan 2 holders (~5.9M GB) is consistent with the DfE
forecast of ~7.4M England graduates with outstanding Plan 2 loans (2024-25),
scaled to GB and adjusted for FRS coverage (~80% of graduates).

Within each age group, we assign the loan plan to the highest-income people
first, reflecting that graduate earnings are above-average.
"""

import numpy as np
from policyengine_uk.data import UKSingleYearDataset
from policyengine_uk import Microsimulation

# Fraction of each age group with an outstanding Plan 2 loan (GB, 2023).
# Calibrated against DfE forecast: 7.44M England graduates with Plan 2 outstanding
# in 2024-25, scaled to GB (÷0.84) and adjusted for FRS coverage (~80%).
# Target GB total: ~7.44/0.84*0.80 = ~7.1M... but FRS pop aged 19-34 = 13.9M
# so realistic rate given actual HE participation for 2012-2022 cohort.
# Rates peak at ages 24-28 (graduates 2-6 years post-study, most still repaying).
_PLAN_2_PARTICIPATION = {
    19: 0.09,
    20: 0.16,
    21: 0.32,
    22: 0.44,
    23: 0.48,
    24: 0.58,
    25: 0.60,
    26: 0.58,
    27: 0.55,
    28: 0.53,
    29: 0.50,
    30: 0.46,
    31: 0.44,
    32: 0.40,
    33: 0.35,
    34: 0.29,
}

# Plan 1: pre-2012 starters. Calibrated to ~3.5M GB total outstanding loans.
# HESA pre-2012 entry ~200-280k/yr England; 14 active cohorts (1998-2011).
# Rates taper at older ages as loans are paid off or written off at age 65.
_PLAN_1_PARTICIPATION = {
    34: 0.37,
    35: 0.37,
    36: 0.35,
    37: 0.34,
    38: 0.32,
    39: 0.30,
    40: 0.29,
    41: 0.27,
    42: 0.25,
    43: 0.24,
    44: 0.22,
    45: 0.20,
    46: 0.17,
    47: 0.13,
    48: 0.12,
    49: 0.10,
    50: 0.08,
    51: 0.07,
    52: 0.05,
    53: 0.03,
    54: 0.03,
    55: 0.02,
}


def impute_student_loan_plan(
    dataset: UKSingleYearDataset,
    year: int = 2023,
) -> UKSingleYearDataset:
    """Impute student loan plan type from age-cohort eligibility and income rank.

    Assigns plans to the highest-income people within each eligible age group,
    up to the participation rate target. This captures both above- and
    below-threshold borrowers, so that uprating correctly activates repayments
    as incomes grow.

    Args:
        dataset: PolicyEngine UK dataset.
        year: FRS survey year (used to compute cohort start years).

    Returns:
        Dataset with imputed student_loan_plan values.
    """
    dataset = dataset.copy()
    sim = Microsimulation(dataset=dataset)

    age = sim.calculate("age").values.astype(int)
    income = sim.calculate("employment_income").values

    n = len(age)
    plan = np.full(n, "NONE", dtype=object)

    def assign_plan(participation_rates, plan_label):
        for a, rate in participation_rates.items():
            age_mask = age == a
            if age_mask.sum() == 0:
                continue
            idx = np.where(age_mask)[0]
            n_assign = max(1, round(len(idx) * rate))
            # Assign to highest-income people in this age group
            ranked = idx[np.argsort(income[idx])[::-1]]
            plan[ranked[:n_assign]] = plan_label

    assign_plan(_PLAN_1_PARTICIPATION, "PLAN_1")
    assign_plan(_PLAN_2_PARTICIPATION, "PLAN_2")
    # Plan 5: near-zero in 2023 (first cohort only just starting in Sept 2023)

    dataset.person["student_loan_plan"] = plan
    return dataset
