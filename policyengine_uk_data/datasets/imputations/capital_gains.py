import pandas as pd
import numpy as np
from policyengine_uk_data.utils.stack import stack_datasets

# Fit a spline to each income band's percentiles
from scipy.interpolate import UnivariateSpline

from policyengine_uk_data.storage import STORAGE_FOLDER

import torch
from torch.optim import Adam
from policyengine_uk.data import UKSingleYearDataset
import logging

capital_gains = pd.read_csv(
    STORAGE_FOLDER / "capital_gains_distribution_advani_summers.csv.gz"
)
capital_gains["maximum_total_income"] = capital_gains.minimum_total_income.shift(
    -1
).fillna(np.inf)

# --- Upper tail of the capital gains distribution -------------------------
#
# The Advani & Summers percentile table stops at p95, so fitting a linear
# spline through p05..p95 and sampling uniform quantiles linearly extrapolates
# everything above the 95th percentile off the last segment. That removes the
# Pareto tail of capital gains entirely: gains end up spread thinly over many
# people instead of concentrated on a few, which both misstates the shape of
# the distribution and suppresses CGT revenue (small gains sit in lower rate
# bands and near the annual exempt amount).
#
# We keep the spline for the body (below p95) -- that is the Advani & Summers
# evidence and is not in question -- and replace the extrapolated region with a
# Pareto tail anchored at p95.
#
# The tail index is *fitted*, not assumed. The same source table publishes
# `mean_gains_given_gains` for each income band, which the spline alone
# undershoots by 35-50% in every band. That gap is precisely the missing tail
# mass, so for each band we solve for the Pareto shape parameter that makes the
# full (body + tail) distribution reproduce the published band mean. Because
# the fit is done per income band, the income-band conditioning of the original
# imputation is preserved rather than a single global tail being imposed.
#
# The resulting shape parameters cluster around 1.45-1.5, consistent with the
# heavy tail HMRC's own size-of-gain distribution implies (see
# storage/capital_gains_size_distribution_hmrc.csv, which is used as the
# external validation reference for this fit).

TAIL_START_QUANTILE = 0.95

# Shape parameters below ~1.15 give a tail so heavy that the sample mean is
# dominated by a handful of draws and the build becomes unstable. A small
# number of bands have an anomalously low p95 relative to their published mean
# and would otherwise solve to alpha ~ 1.0; we floor them.
MIN_TAIL_ALPHA = 1.15
MAX_TAIL_ALPHA = 50.0

# Largest single gain any individual may be assigned. HMRC's top published band
# is "£5m and over"; this bounds the tail well above that while preventing a
# single draw from dominating national totals.
MAX_SINGLE_GAIN = 500_000_000.0


def _band_spline(row) -> UnivariateSpline:
    """Linear spline through the published percentiles for one income band."""
    return UnivariateSpline(
        [0.05, 0.1, 0.25, 0.5, 0.75, 0.90, 0.95],
        [row.p05, row.p10, row.p25, row.p50, row.p75, row.p90, row.p95],
        k=1,
    )


def _body_mean(spline: UnivariateSpline) -> float:
    """Mean of the spline body over quantiles [0, TAIL_START_QUANTILE)."""
    # Deterministic quadrature rather than Monte Carlo, so the fitted tail does
    # not depend on any random draw.
    grid = np.linspace(0, TAIL_START_QUANTILE, 10001)
    return float(np.trapezoid(spline(grid), grid) / TAIL_START_QUANTILE)


def _capped_pareto_mean(alpha: float, scale: float, cap: float) -> float:
    """E[min(X, cap)] for X ~ Pareto(shape=alpha, scale=scale)."""
    if scale >= cap:
        return cap
    return scale + scale**alpha * (cap ** (1 - alpha) - scale ** (1 - alpha)) / (
        1 - alpha
    )


def _fit_tail_alpha(row) -> float:
    """Fit the Pareto shape parameter for one income band.

    Chosen so that the band's full distribution -- spline body below p95, Pareto
    above -- reproduces the published `mean_gains_given_gains` for that band.
    """
    from scipy.optimize import brentq

    scale = float(row.p95)
    target_mean = float(row.mean_gains_given_gains)
    body_mean = _body_mean(_band_spline(row))

    if scale <= 0:
        # Degenerate band: no positive anchor for a multiplicative tail.
        return MAX_TAIL_ALPHA

    tail_weight = 1 - TAIL_START_QUANTILE
    required_tail_mean = (target_mean - TAIL_START_QUANTILE * body_mean) / tail_weight

    def gap(alpha: float) -> float:
        return _capped_pareto_mean(alpha, scale, MAX_SINGLE_GAIN) - required_tail_mean

    # The capped Pareto mean decreases monotonically in alpha, so the solution
    # is bracketed whenever the endpoints straddle zero.
    if gap(MIN_TAIL_ALPHA) < 0:
        # Even the heaviest tail we allow cannot reach the published mean.
        return MIN_TAIL_ALPHA
    if gap(MAX_TAIL_ALPHA) > 0:
        # Band mean is already met by the body; use the thinnest tail.
        return MAX_TAIL_ALPHA
    return float(brentq(gap, MIN_TAIL_ALPHA, MAX_TAIL_ALPHA))


def sample_band_gains(row, quantiles: np.ndarray) -> np.ndarray:
    """Map uniform quantiles to capital gains for one income band.

    Below TAIL_START_QUANTILE this is exactly the original Advani & Summers
    spline. Above it, quantiles are mapped through a fitted Pareto tail
    anchored at the band's p95.
    """
    spline = _band_spline(row)
    gains = spline(quantiles)

    in_tail = quantiles >= TAIL_START_QUANTILE
    if in_tail.any() and row.p95 > 0:
        alpha = _fit_tail_alpha(row)
        # Rescale the tail quantiles to (0, 1] and invert the Pareto CDF.
        residual = (1 - quantiles[in_tail]) / (1 - TAIL_START_QUANTILE)
        residual = np.clip(residual, 1e-12, 1.0)
        gains[in_tail] = np.minimum(row.p95 * residual ** (-1 / alpha), MAX_SINGLE_GAIN)

    return gains


# Silence verbose logging
logging.getLogger("root").setLevel(logging.WARNING)


def impute_cg_to_doubled_dataset(
    dataset: UKSingleYearDataset,
) -> tuple[np.ndarray, np.ndarray]:
    """Assumes that the capital gains distribution is the same for all years."""

    from policyengine_uk import Microsimulation

    sim = Microsimulation(dataset=dataset)
    ti = sim.calculate("total_income").values
    household_weight = sim.calculate("household_weight").values
    first_half = (
        np.concatenate(
            [
                np.ones(len(household_weight) // 2),
                np.zeros(len(household_weight) // 2),
            ]
        )
        > 0
    )
    # Give capital gains to one adult aged 15+ in each household
    adult_index = sim.calculate("adult_index").values == 1
    in_person_second_half = np.zeros(len(ti)) > 0
    in_person_second_half[len(ti) // 2 :] = True
    has_cg = np.zeros(len(ti)) > 0
    has_cg[adult_index & in_person_second_half] = True
    blend_factor = torch.tensor(
        np.zeros(first_half.sum()), requires_grad=True, dtype=torch.float32
    )
    household_weight = torch.tensor(household_weight, dtype=torch.float32)
    sigmoid = torch.nn.Sigmoid()

    def loss(blend_factor):
        loss = 0

        blended_household_weight = household_weight.clone()
        adjusted_blend_factor = sigmoid(blend_factor)
        blended_household_weight[first_half] = (
            adjusted_blend_factor * blended_household_weight[first_half]
        )
        blended_household_weight[~first_half] = (
            1 - adjusted_blend_factor
        ) * blended_household_weight[first_half]
        for i in range(len(capital_gains)):
            lower = capital_gains.minimum_total_income.iloc[i]
            upper = capital_gains.maximum_total_income.iloc[i]
            true_pct_with_gains = capital_gains.percent_with_gains.iloc[i]

            ti_in_range = (ti >= lower) * (ti < upper)
            cg_in_income_range = has_cg * ti_in_range
            household_ti_in_range_count = torch.tensor(
                sim.map_result(ti_in_range, "person", "household", how="sum")
            )
            household_cg_in_income_range_count = torch.tensor(
                sim.map_result(cg_in_income_range, "person", "household", how="sum")
            )
            pred_ti_in_range = (
                blended_household_weight * household_ti_in_range_count
            ).sum()
            pred_cg_in_income_range = (
                blended_household_weight * household_cg_in_income_range_count
            ).sum()
            pred_pct_with_gains = pred_cg_in_income_range / torch.clip(
                pred_ti_in_range, 1
            )
            loss += (pred_pct_with_gains - true_pct_with_gains) ** 2

        return loss

    optimiser = Adam([blend_factor], lr=1e-1)
    progress = range(100)
    logging.info("Splitting household weights into has-gains and no-gains")
    for i in progress:
        optimiser.zero_grad()
        loss_value = loss(blend_factor)
        loss_value.backward()
        optimiser.step()
        if loss_value.item() < 1e-3:
            break

    new_household_weight = household_weight.detach().numpy()
    original_household_weight = new_household_weight.copy()
    blend_factor = sigmoid(blend_factor).detach().numpy()
    new_household_weight[first_half] = (
        blend_factor * original_household_weight[first_half]
    )
    new_household_weight[~first_half] = (1 - blend_factor) * original_household_weight[
        first_half
    ]

    # Impute actual capital gains amounts given gains
    new_cg = np.zeros(len(ti))

    logging.info("Imputing capital gains among those with gains")

    # Draw imputation quantiles from a seeded generator so the build is
    # reproducible: an unseeded global np.random made capital gains (and hence
    # CGT revenue) differ between otherwise identical builds.
    cg_rng = np.random.default_rng(0)

    for i in range(len(capital_gains)):
        row = capital_gains.iloc[i]
        lower = row.minimum_total_income
        upper = row.maximum_total_income
        ti_in_range = (ti >= lower) * (ti < upper)
        in_target_range = has_cg * ti_in_range > 0
        # One uniform draw per person from the seeded generator, exactly as
        # before, so the sequence of random numbers consumed is unchanged.
        quantiles = cg_rng.random(int(in_target_range.sum()))
        pred_capital_gains = sample_band_gains(row, quantiles)
        new_cg[in_target_range] = pred_capital_gains

    return new_cg, new_household_weight


def impute_capital_gains(dataset: UKSingleYearDataset) -> UKSingleYearDataset:
    dataset = dataset.copy()
    dataset.household["household_is_capital_gains_clone"] = False
    zero_weight_copy = dataset.copy()
    zero_weight_copy.household.household_weight = 1
    zero_weight_copy.household["household_is_capital_gains_clone"] = True
    data = stack_datasets(
        dataset,
        zero_weight_copy,
    )

    pred_cg, household_weight = impute_cg_to_doubled_dataset(data)

    data.person["capital_gains"] = pred_cg
    data.household["household_weight"] = household_weight

    data.validate()
    return data
