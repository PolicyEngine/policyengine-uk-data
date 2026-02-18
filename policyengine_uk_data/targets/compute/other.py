"""Miscellaneous compute functions (vehicles, housing, savings, SCP)."""

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
