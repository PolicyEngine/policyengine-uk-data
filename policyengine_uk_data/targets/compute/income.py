"""Income and salary sacrifice compute functions."""

import numpy as np
import pandas as pd
from policyengine_uk_data.storage import STORAGE_FOLDER


def compute_income_band(target, ctx) -> np.ndarray:
    """Compute income variable within a total income band."""
    variable = target.variable
    lower = target.lower_bound
    upper = target.upper_bound

    income_df = ctx.sim.calculate_dataframe(["total_income", variable])
    in_band = (income_df.total_income >= lower) & (income_df.total_income < upper)

    if target.is_count:
        return ctx.household_from_person((income_df[variable] > 0) * in_band)
    else:
        return ctx.household_from_person(income_df[variable] * in_band)


# Income tax on earned income in each rUK band (rates.uk thresholds on
# earned_taxable_income), keyed by the band word in the target name. Scottish
# taxpayers are split on the same rUK boundaries, an approximation: HMRC
# groups their relief by Scottish marginal rate (starter and intermediate
# with basic), and the Scottish band thresholds differ from the rUK ones.
_BAND_TAX_VARIABLES = {
    "basic": "basic_rate_earned_income_tax",
    "higher": "higher_rate_earned_income_tax",
    "additional": "add_rate_earned_income_tax",
}


def split_relief_by_band(
    relief: np.ndarray, band_tax_cf: dict, band_tax_base: dict
) -> dict:
    """Share each person's income tax relief across the bands it is given in.

    HMRC's Table 6.1 models contributions that straddle bands ("not all be
    relievable at the same rate"; private pension statistics, background and
    methodology), so a person's relief is split in proportion to the fall in
    their earned-income tax within each band. Relief with no fall in any
    earned band (it comes through savings or dividend bands) counts as basic
    rate. The shares sum to one, so the bands add up to ``relief``.
    """
    drops = {
        band: np.maximum(np.asarray(band_tax_cf[band]) - band_tax_base[band], 0)
        for band in _BAND_TAX_VARIABLES
    }
    total = sum(drops.values())
    has_drop = total > 0
    safe_total = np.where(has_drop, total, 1)
    relief = np.asarray(relief)
    return {
        band: relief
        * np.where(has_drop, drop / safe_total, 1.0 if band == "basic" else 0.0)
        for band, drop in drops.items()
    }


def compute_ss_it_relief(target, ctx) -> np.ndarray:
    """Compute salary sacrifice IT relief, in total or for one rate band."""
    period = ctx.time_period
    cf, base = ctx.counterfactual_sim, ctx.sim
    relief = np.asarray(cf.calculate("income_tax", period)) - np.asarray(
        base.calculate("income_tax", period)
    )
    band = next((b for b in _BAND_TAX_VARIABLES if b in target.name), None)
    if band is not None:
        relief = split_relief_by_band(
            relief,
            {
                b: np.asarray(cf.calculate(v, period))
                for b, v in _BAND_TAX_VARIABLES.items()
            },
            {
                b: np.asarray(base.calculate(v, period))
                for b, v in _BAND_TAX_VARIABLES.items()
            },
        )[band]
    return ctx.household_from_person(relief)


def compute_ss_contributions(target, ctx) -> np.ndarray:
    """Compute total salary sacrifice pension contributions."""
    ss = ctx.sim.calculate("pension_contributions_via_salary_sacrifice")
    return ctx.household_from_person(ss)


def compute_ss_ni_relief(target, ctx) -> np.ndarray:
    """Compute salary sacrifice NI relief (employee or employer)."""
    name = target.name
    if "employee" in name:
        ni_base = ctx.sim.calculate("ni_employee")
        ni_cf = ctx.counterfactual_sim.calculate("ni_employee", ctx.time_period)
    else:
        ni_base = ctx.sim.calculate("ni_employer")
        ni_cf = ctx.counterfactual_sim.calculate("ni_employer", ctx.time_period)
    return ctx.household_from_person(ni_cf - ni_base)


def compute_ss_headcount(target, ctx) -> np.ndarray:
    """Compute salary sacrifice user headcounts.

    The 2k cap is defined at 2023-24 FRS base-year prices. The dataset
    is uprated to 2025 for calibration then downrated to 2023 for
    saving, but PE does not uprate SS when loading. To keep the
    above/below classification consistent, deflate SS to base-year
    prices before applying the threshold.
    """
    ss = ctx.sim.calculate("pension_contributions_via_salary_sacrifice")
    uprating = pd.read_csv(STORAGE_FOLDER / "uprating_factors.csv").set_index(
        "Variable"
    )
    row = "pension_contributions_via_salary_sacrifice"
    price_adj = uprating.loc[row, "2023"] / uprating.loc[row, str(ctx.time_period)]
    ss_base = ss * price_adj

    name = target.name
    if "below_cap" in name:
        mask = (ss_base > 0) & (ss_base <= 2000)
    elif "above_cap" in name:
        mask = ss_base > 2000
    else:
        mask = ss_base > 0
    return ctx.household_from_person(mask)


def compute_esa(target, ctx) -> np.ndarray:
    """Compute ESA (combined income-related + contributory)."""
    return ctx.household_from_family(
        ctx.sim.calculate("esa_income")
    ) + ctx.household_from_person(ctx.sim.calculate("esa_contrib"))
