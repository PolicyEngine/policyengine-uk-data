"""Miscellaneous compute functions (vehicles, housing, savings, SCP, student loans)."""

import numpy as np


def compute_vehicles(target, ctx) -> np.ndarray:
    """Compute vehicle ownership targets."""
    name = target.name
    if name == "nts/households_no_vehicle":
        return (ctx.pe("num_vehicles") == 0).astype(float)
    if name == "nts/households_one_vehicle":
        return (ctx.pe("num_vehicles") == 1).astype(float)
    return (ctx.pe("num_vehicles") >= 2).astype(float)


def compute_housing(target, ctx) -> np.ndarray:
    """Compute housing targets (mortgage, private rent, social rent)."""
    name = target.name
    if name == "housing/total_mortgage":
        return ctx.pe("mortgage_capital_repayment") + ctx.pe(
            "mortgage_interest_repayment"
        )
    tenure = ctx.sim.calculate("tenure_type", map_to="household").values
    if name == "housing/rent_social":
        is_social = (tenure == "RENT_FROM_COUNCIL") | (tenure == "RENT_FROM_HA")
        return ctx.pe("rent") * is_social
    return ctx.pe("rent") * (tenure == "RENT_PRIVATELY")


def compute_savings_interest(target, ctx) -> np.ndarray:
    """Compute ONS savings interest income."""
    savings = ctx.sim.calculate("savings_interest_income")
    return ctx.household_from_person(savings)


def compute_scottish_child_payment(target, ctx) -> np.ndarray:
    """Compute Scottish child payment spend."""
    scp = ctx.sim.calculate("scottish_child_payment")
    return ctx.household_from_person(scp)


def compute_land_value(target, ctx) -> np.ndarray:
    """Compute land/property wealth targets from household-level variables."""
    return ctx.pe(target.variable)


def compute_regional_land_value(target, ctx) -> np.ndarray:
    """Compute household land value filtered to a single region."""
    region = target.name.split("/")[-1]  # e.g. "ons/household_land_value/LONDON"
    in_region = ctx.sim.calculate("region").values == region
    return ctx.pe("household_land_value") * in_region


def compute_student_loan_plan(target, ctx) -> np.ndarray:
    """Count England borrowers on a given plan with repayments > 0.

    SLC targets cover borrowers liable to repay AND earning above threshold
    in England only — matching exactly what the FRS captures via PAYE.
    """
    plan_name = target.name  # e.g. "slc/plan_2_borrowers_above_threshold"
    if "plan_2" in plan_name:
        plan_value = "PLAN_2"
    elif "plan_5" in plan_name:
        plan_value = "PLAN_5"
    else:
        return None

    plan = ctx.pe_person("student_loan_plan")
    repayments = ctx.pe_person("student_loan_repayments")
    on_plan = (plan == plan_value) & (ctx.country == "ENGLAND") & (repayments > 0)

    return ctx.household_from_person(on_plan.astype(float))


def compute_student_loan_plan_liable(target, ctx) -> np.ndarray:
    """Count all England borrowers on a given plan, including non-repayers."""
    plan_name = target.name  # e.g. "slc/plan_2_borrowers_liable"
    if "plan_2" in plan_name:
        plan_value = "PLAN_2"
    elif "plan_5" in plan_name:
        plan_value = "PLAN_5"
    else:
        return None

    plan = ctx.pe_person("student_loan_plan")
    on_plan = (plan == plan_value) & (ctx.country == "ENGLAND")

    return ctx.household_from_person(on_plan.astype(float))
