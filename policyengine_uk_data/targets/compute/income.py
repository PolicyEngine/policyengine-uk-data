"""Income and salary sacrifice compute functions."""

import numpy as np


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


def tax_by_band(income: np.ndarray, thresholds, rates) -> dict:
    """Tax on ``income`` within each of HMRC's rate categories.

    HMRC's Table 6.1 groups relief by marginal rate: basic (with the Scottish
    starter and intermediate rates), higher and additional. Brackets taxed
    below 30% are basic, brackets at the scale's top rate are additional (the
    rUK additional rate, the Scottish top rate) and the rest are higher (the
    Scottish higher and advanced rates, which cover the income range of the
    rUK higher rate). The categories sum to the scale's tax on ``income``.
    """
    income = np.asarray(income, dtype=float)
    thresholds = [t for t, r in zip(thresholds, rates) if r is not None]
    rates = [r for r in rates if r is not None]
    uppers = thresholds[1:] + [np.inf]
    bands = {"basic": 0.0, "higher": 0.0, "additional": 0.0}
    for lower, upper, rate in zip(thresholds, uppers, rates):
        band = (
            "basic" if rate < 0.3 else "additional" if rate == rates[-1] else "higher"
        )
        bands[band] = bands[band] + rate * np.clip(income - lower, 0, upper - lower)
    return bands


def ss_it_relief_by_band(ctx) -> dict:
    """Person-level salary sacrifice income tax relief in each rate category.

    HMRC applies income tax rates to employees' pay (ASHE), so relief is the
    rise in tax on earned income when the sacrifice is paid as salary, under
    the person's rUK or Scottish rates. Contributions that straddle a band
    boundary are relieved partly at each rate, as in HMRC's estimates.
    """
    period = ctx.time_period
    rates = ctx.sim.tax_benefit_system.parameters(period).gov.hmrc.income_tax.rates
    scottish = np.asarray(ctx.sim.calculate("pays_scottish_income_tax", period))
    base = np.asarray(ctx.sim.calculate("earned_taxable_income", period))
    cf = np.asarray(ctx.counterfactual_sim.calculate("earned_taxable_income", period))
    relief = {}
    for scale, in_scale in ((rates.uk, ~scottish), (rates.scotland.rates, scottish)):
        band_cf = tax_by_band(cf, scale.thresholds, scale.rates)
        band_base = tax_by_band(base, scale.thresholds, scale.rates)
        for band in band_cf:
            relief[band] = relief.get(band, 0) + in_scale * (
                band_cf[band] - band_base[band]
            )
    return relief


def compute_ss_it_relief(target, ctx) -> np.ndarray:
    """Compute salary sacrifice income tax relief at one rate."""
    band = target.name.removeprefix("hmrc/salary_sacrifice_it_relief_")
    return ctx.household_from_person(
        ss_it_relief_by_band(ctx)[band.removesuffix("_rate")]
    )


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

    The 2k cap is applied to the contributions the calibration-year
    simulation holds. uprating_factors.csv follows policyengine-uk's
    load-time uprating, so these are the amounts the model runs on in that
    year (the survey amounts, as policyengine-uk does not uprate this
    variable at load), the values test_salary_sacrifice_headcount checks.
    """
    ss = ctx.sim.calculate("pension_contributions_via_salary_sacrifice")

    name = target.name
    if "below_cap" in name:
        mask = (ss > 0) & (ss <= 2000)
    elif "above_cap" in name:
        mask = ss > 2000
    else:
        mask = ss > 0
    return ctx.household_from_person(mask)


def compute_esa(target, ctx) -> np.ndarray:
    """Compute ESA (combined income-related + contributory)."""
    return ctx.household_from_family(
        ctx.sim.calculate("esa_income")
    ) + ctx.household_from_person(ctx.sim.calculate("esa_contrib"))
