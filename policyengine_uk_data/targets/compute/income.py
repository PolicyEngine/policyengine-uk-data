"""Income and salary sacrifice compute functions."""

import numpy as np


def compute_income_band(target, ctx) -> np.ndarray:
    """Compute income variable within a total income band."""
    variable = target.variable
    lower = target.lower_bound
    upper = target.upper_bound

    income_df = ctx.sim.calculate_dataframe(["total_income", variable])
    in_band = (income_df.total_income >= lower) & (
        income_df.total_income < upper
    )

    if target.is_count:
        return ctx.household_from_person((income_df[variable] > 0) * in_band)
    else:
        return ctx.household_from_person(income_df[variable] * in_band)


def compute_ss_it_relief(target, ctx) -> np.ndarray:
    """Compute salary sacrifice IT relief by tax band."""
    it_base = ctx.sim.calculate("income_tax")
    it_cf = ctx.counterfactual_sim.calculate("income_tax", ctx.time_period)
    it_relief = it_cf - it_base

    adj_net_income_cf = ctx.counterfactual_sim.calculate(
        "adjusted_net_income", ctx.time_period
    )

    params = ctx.sim.tax_benefit_system.parameters.gov.hmrc.income_tax.rates.uk
    basic_thresh = params[0].threshold(ctx.time_period)
    higher_thresh = params[1].threshold(ctx.time_period)
    additional_thresh = params[2].threshold(ctx.time_period)

    name = target.name
    if "basic" in name:
        mask = (adj_net_income_cf > basic_thresh) & (
            adj_net_income_cf <= higher_thresh
        )
    elif "higher" in name:
        mask = (adj_net_income_cf > higher_thresh) & (
            adj_net_income_cf <= additional_thresh
        )
    elif "additional" in name:
        mask = adj_net_income_cf > additional_thresh
    else:
        mask = np.ones_like(it_relief, dtype=bool)

    return ctx.household_from_person(it_relief * mask)


def compute_ss_contributions(target, ctx) -> np.ndarray:
    """Compute total salary sacrifice pension contributions."""
    ss = ctx.sim.calculate("pension_contributions_via_salary_sacrifice")
    return ctx.household_from_person(ss)


def compute_ss_ni_relief(target, ctx) -> np.ndarray:
    """Compute salary sacrifice NI relief (employee or employer)."""
    name = target.name
    if "employee" in name:
        ni_base = ctx.sim.calculate("ni_employee")
        ni_cf = ctx.counterfactual_sim.calculate(
            "ni_employee", ctx.time_period
        )
    else:
        ni_base = ctx.sim.calculate("ni_employer")
        ni_cf = ctx.counterfactual_sim.calculate(
            "ni_employer", ctx.time_period
        )
    return ctx.household_from_person(ni_cf - ni_base)


def compute_esa(target, ctx) -> np.ndarray:
    """Compute ESA (combined income-related + contributory)."""
    return ctx.household_from_person(
        ctx.sim.calculate("esa_income")
    ) + ctx.household_from_person(ctx.sim.calculate("esa_contrib"))
