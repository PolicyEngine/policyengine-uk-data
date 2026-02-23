"""Compute subpackage: domain-specific column computation for targets."""

from policyengine_uk_data.targets.compute.benefits import (
    compute_benefit_cap,
    compute_pip_claimants,
    compute_scotland_uc_child,
    compute_two_child_limit,
    compute_uc_by_children,
    compute_uc_by_family_type,
    compute_uc_jobseeker,
    compute_uc_outside_cap,
    compute_uc_payment_dist,
)
from policyengine_uk_data.targets.compute.council_tax import (
    compute_council_tax_band,
    compute_obr_council_tax,
)
from policyengine_uk_data.targets.compute.demographics import (
    compute_gender_age,
    compute_regional_age,
    compute_scotland_demographics,
    compute_uk_population,
)
from policyengine_uk_data.targets.compute.households import (
    compute_household_type,
    compute_tenure,
)
from policyengine_uk_data.targets.compute.income import (
    compute_esa,
    compute_income_band,
    compute_ss_contributions,
    compute_ss_headcount,
    compute_ss_it_relief,
    compute_ss_ni_relief,
)
from policyengine_uk_data.targets.compute.other import (
    compute_housing,
    compute_savings_interest,
    compute_scottish_child_payment,
    compute_student_loan_plan,
    compute_student_loan_plan_liable,
    compute_vehicles,
)

__all__ = [
    "compute_benefit_cap",
    "compute_council_tax_band",
    "compute_esa",
    "compute_gender_age",
    "compute_household_type",
    "compute_housing",
    "compute_income_band",
    "compute_obr_council_tax",
    "compute_pip_claimants",
    "compute_regional_age",
    "compute_savings_interest",
    "compute_scotland_demographics",
    "compute_scotland_uc_child",
    "compute_scottish_child_payment",
    "compute_student_loan_plan",
    "compute_student_loan_plan_liable",
    "compute_ss_contributions",
    "compute_ss_headcount",
    "compute_ss_it_relief",
    "compute_ss_ni_relief",
    "compute_tenure",
    "compute_two_child_limit",
    "compute_uc_by_children",
    "compute_uc_by_family_type",
    "compute_uc_jobseeker",
    "compute_uc_outside_cap",
    "compute_uc_payment_dist",
    "compute_uk_population",
    "compute_vehicles",
]
