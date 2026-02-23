"""Miscellaneous compute functions (vehicles, housing, savings, SCP,
student loans)."""

import numpy as np

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


def compute_vehicles(target, ctx) -> np.ndarray:
    """Compute vehicle ownership targets."""
    name = target.name
    if name == "nts/households_no_vehicle":
        return (ctx.pe("num_vehicles") == 0).astype(float)
    if name == "nts/households_one_vehicle":
        return (ctx.pe("num_vehicles") == 1).astype(float)
    return (ctx.pe("num_vehicles") >= 2).astype(float)


def compute_housing(target, ctx) -> np.ndarray:
    """Compute housing targets (mortgage, private rent)."""
    name = target.name
    if name == "housing/total_mortgage":
        return ctx.pe("mortgage_capital_repayment") + ctx.pe(
            "mortgage_interest_repayment"
        )
    tenure = ctx.sim.calculate("tenure_type", map_to="household").values
    return ctx.pe("rent") * (tenure == "RENT_PRIVATELY")


def compute_savings_interest(target, ctx) -> np.ndarray:
    """Compute ONS savings interest income."""
    savings = ctx.sim.calculate("savings_interest_income")
    return ctx.household_from_person(savings)


def compute_scottish_child_payment(target, ctx) -> np.ndarray:
    """Compute Scottish child payment spend."""
    scp = ctx.sim.calculate("scottish_child_payment")
    return ctx.household_from_person(scp)


def compute_student_loan_plan(target, ctx) -> np.ndarray:
    """Count England borrowers on a given plan with repayments > 0.

    SLC "above_threshold" targets cover borrowers liable to repay AND earning
    above threshold in England only — matching what the FRS captures via PAYE.
    """
    plan_name = target.name  # e.g. "slc/plan_2_borrowers_above_threshold"
    if "plan_2" in plan_name:
        plan_value = "PLAN_2"
    elif "plan_5" in plan_name:
        plan_value = "PLAN_5"
    else:
        return None

    plan = ctx.sim.calculate("student_loan_plan").values
    region = ctx.sim.calculate("region", map_to="person").values
    repayments = ctx.sim.calculate("student_loan_repayments").values

    is_england = np.isin(region, list(_ENGLAND_REGIONS))
    # Only count those with repayments > 0 (above threshold)
    on_plan = (plan == plan_value) & is_england & (repayments > 0)

    return ctx.household_from_person(on_plan.astype(float))


def compute_student_loan_plan_liable(target, ctx) -> np.ndarray:
    """Count ALL England borrowers on a given plan (including below-threshold).

    SLC "liable" targets cover all borrowers liable to repay, regardless of
    whether they earn above the repayment threshold.
    """
    plan_name = target.name  # e.g. "slc/plan_2_borrowers_liable"
    if "plan_2" in plan_name:
        plan_value = "PLAN_2"
    elif "plan_5" in plan_name:
        plan_value = "PLAN_5"
    else:
        return None

    plan = ctx.sim.calculate("student_loan_plan").values
    region = ctx.sim.calculate("region", map_to="person").values

    is_england = np.isin(region, list(_ENGLAND_REGIONS))
    on_plan = (plan == plan_value) & is_england

    return ctx.household_from_person(on_plan.astype(float))
