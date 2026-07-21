"""HMRC capital gains targets.

The Enhanced FRS imputes capital gains, but nothing in the calibration
constrains how those gains are *distributed* across households. Measured
on the stock weights for 2026, the microdata carries 1.29m CGT taxpayers
holding £112bn of gains (an £87,158 mean) against HMRC's 378k taxpayers,
£65.9bn and a ~£174,000 mean: 3.4x too many taxpayers, each holding a
gain about half the administrative average. That is a distributional
error, not a level shift, so a revenue target cannot fix it.

Two targets are added:

* ``hmrc/capital_gains_total`` — total chargeable gains of CGT taxpayers.
* ``hmrc/cgt_taxpayers`` — the number of CGT taxpayers.

Both use a ``custom_compute``. The default count path in
``build_loss_matrix._compute_simple_count`` counts people with
``capital_gains > 0``, which is not HMRC's concept: HMRC counts CGT
*taxpayers*, i.e. people whose gains exceed the annual exempt amount
(AEA). The amount target is gated on the same AEA condition, for the
same reason — HMRC's £65.9bn is the gains of CGT-liable taxpayers, not
all gains realised in the economy. Gating only the count and leaving the
amount ungated would target a mean gain that is a ratio of two different
populations, and would let the optimiser satisfy the amount with
sub-threshold gains that HMRC never counts.

The AEA is policy-dependent (£12,300 for 2022-23, £6,000 for 2023-24,
£3,000 from 2024-25) and is read from PolicyEngine's own parameter tree
for the simulation year, so the target's meaning tracks the policy in
force. ``_AEA_FALLBACK`` is used only if the parameter lookup fails.

Year handling: HMRC's figures are the 2023-24 outturn, mapped to
calendar 2024 following ``hmrc_spi._SPI_YEAR``. The amount is projected
to 2029 with PolicyEngine's ``capital_gains`` uprating factors so the
target constrains the projection years the calibration actually runs
for; the taxpayer count is held flat, as the repo has no administrative
basis for forecasting CGT taxpayer numbers and count uprating factors
would be a nominal-income index applied to a headcount.

Caveat: the 2023-24 outturn was realised under a £6,000 AEA, while the
gate applied in later years uses the £3,000 AEA then in force. The
targets are therefore approximate for the projection years; the
alternative (a fixed £6,000 gate) would be wrong in a different and less
transparent way, because it would not describe any year's actual policy.

Source: https://www.gov.uk/government/statistics/capital-gains-tax-statistics
"""

import logging

import numpy as np

from policyengine_uk_data.targets.schema import Target, Unit
from policyengine_uk_data.targets.sources._common import load_config

logger = logging.getLogger(__name__)

# HMRC CGT statistics, tax year 2023-24 outturn, mapped to calendar 2024
# following the hmrc_spi._SPI_YEAR convention.
_CGT_BASE_YEAR = 2024
_MAX_YEAR = 2029

# Total chargeable gains of CGT taxpayers, 2023-24 (£65.9bn).
_CGT_TOTAL_GAINS = 65.9e9
# Number of CGT taxpayers, 2023-24.
_CGT_TAXPAYERS = 378_000

# Annual exempt amount by tax year start, used only if the PolicyEngine
# parameter lookup fails.
_AEA_FALLBACK = {
    2022: 12_300,
    2023: 6_000,
}
_AEA_FALLBACK_DEFAULT = 3_000

_PARAMETER_PATH = "gov.hmrc.cgt.annual_exempt_amount"


def _annual_exempt_amount(ctx, year: int) -> float:
    """Annual exempt amount in force in ``year``.

    Read from PolicyEngine's parameter tree so the threshold tracks the
    policy actually simulated, falling back to an explicit mapping.
    """
    try:
        params = ctx.sim.tax_benefit_system.parameters(f"{year}-06-01")
        return float(params.gov.hmrc.cgt.annual_exempt_amount)
    except Exception as e:  # pragma: no cover - defensive
        logger.warning(
            "Could not read %s for %s (%s); using fallback AEA",
            _PARAMETER_PATH,
            year,
            e,
        )
        if year <= 2023:
            return float(_AEA_FALLBACK.get(year, _AEA_FALLBACK_DEFAULT))
        return float(_AEA_FALLBACK_DEFAULT)


def _above_aea_gains(ctx, year: int) -> np.ndarray:
    """Person-level capital gains, zeroed below the annual exempt amount."""
    gains = np.asarray(ctx.pe_person("capital_gains"), dtype=float)
    aea = _annual_exempt_amount(ctx, year)
    return np.where(gains > aea, gains, 0.0)


def compute_cgt_taxpayers(ctx, target: Target, year: int) -> np.ndarray:
    """Household count of people with gains above the annual exempt amount."""
    above = _above_aea_gains(ctx, year) > 0
    return np.asarray(ctx.household_from_person(above.astype(float)), dtype=float)


def compute_capital_gains_total(ctx, target: Target, year: int) -> np.ndarray:
    """Household total of gains held by CGT taxpayers (above-AEA people)."""
    return np.asarray(
        ctx.household_from_person(_above_aea_gains(ctx, year)), dtype=float
    )


def _project_gains(base_value: float) -> dict[int, float]:
    """Uprate the base-year gains total across the calibration years."""
    from policyengine_uk_data.utils.uprating import uprate_values

    values = {_CGT_BASE_YEAR: base_value}
    for year in range(_CGT_BASE_YEAR + 1, _MAX_YEAR + 1):
        try:
            values[year] = float(
                uprate_values(
                    base_value,
                    "capital_gains",
                    start_year=_CGT_BASE_YEAR,
                    end_year=year,
                )
            )
        except Exception as e:
            logger.warning("Could not uprate capital gains to %s: %s", year, e)
            break
    return values


# --- Size-of-gain band targets --------------------------------------------
#
# The aggregate count and total above cannot, on their own, constrain the
# *shape* of the gains distribution: the published enhanced FRS carried
# only ~1.6% of gains in gains of £1m+ against HMRC's ~61%, while still
# being able to satisfy an aggregate total. HMRC Table 2.1a (committed as
# capital_gains_size_distribution_hmrc.csv) publishes taxpayer counts and
# gains by size of gain, so each band becomes a target pair. The
# zero-weight donor households stacked in
# datasets/imputations/capital_gains.py carry each band's mean gain, which
# gives the calibration records it can weight to satisfy these targets.
#
# Bands below £12,300 are skipped: HMRC's sub-AEA rows mix exemption
# regimes and are definitionally incomplete (see the CSV header), and the
# spline-imputed body already covers small gains.

_SIZE_BANDS_FILE = "capital_gains_size_distribution_hmrc.csv"
_MIN_BAND_LOWER = 12_300


def _load_size_bands():
    import pandas as pd

    from policyengine_uk_data.storage import STORAGE_FOLDER

    bands = pd.read_csv(STORAGE_FOLDER / _SIZE_BANDS_FILE, comment="#")
    bands = bands[bands.lower_limit >= _MIN_BAND_LOWER].reset_index(drop=True)
    bands["upper_limit"] = bands.lower_limit.shift(-1).fillna(np.inf)
    return bands


def _band_mask(ctx, year: int, lower: float, upper: float) -> np.ndarray:
    """People whose above-AEA gains fall inside [lower, upper)."""
    gains = np.asarray(ctx.pe_person("capital_gains"), dtype=float)
    aea = _annual_exempt_amount(ctx, year)
    return (gains > aea) & (gains >= lower) & (gains < upper)


def _make_band_count_compute(lower: float, upper: float):
    def compute(ctx, target: Target, year: int) -> np.ndarray:
        mask = _band_mask(ctx, year, lower, upper)
        return np.asarray(ctx.household_from_person(mask.astype(float)), dtype=float)

    return compute


def _make_band_gains_compute(lower: float, upper: float):
    def compute(ctx, target: Target, year: int) -> np.ndarray:
        gains = np.asarray(ctx.pe_person("capital_gains"), dtype=float)
        mask = _band_mask(ctx, year, lower, upper)
        return np.asarray(ctx.household_from_person(gains * mask), dtype=float)

    return compute


def _band_targets(reference_url: str) -> list[Target]:
    targets = []
    for row in _load_size_bands().itertuples():
        lower = float(row.lower_limit)
        upper = float(row.upper_limit)
        count = float(row.taxpayers_thousands) * 1e3
        gains = float(row.gains_gbp_millions) * 1e6
        label = f"{int(lower)}"
        targets.append(
            Target(
                name=f"hmrc/cgt_taxpayers_band_{label}",
                variable="capital_gains",
                source="hmrc",
                unit=Unit.COUNT,
                values={year: count for year in range(_CGT_BASE_YEAR, _MAX_YEAR + 1)},
                is_count=True,
                reference_url=reference_url,
                forecast_vintage="2023-24 outturn",
                custom_compute=_make_band_count_compute(lower, upper),
            )
        )
        targets.append(
            Target(
                name=f"hmrc/capital_gains_band_{label}",
                variable="capital_gains",
                source="hmrc",
                unit=Unit.GBP,
                values=_project_gains(gains),
                reference_url=reference_url,
                forecast_vintage="2023-24 outturn",
                custom_compute=_make_band_gains_compute(lower, upper),
            )
        )
    return targets


def get_targets() -> list[Target]:
    try:
        reference_url = load_config()["hmrc"]["capital_gains_statistics"]
    except Exception as e:
        logger.warning("Could not load HMRC CGT source URL: %s", e)
        reference_url = (
            "https://www.gov.uk/government/statistics/capital-gains-tax-statistics"
        )

    gains_values = _project_gains(_CGT_TOTAL_GAINS)
    count_values = {
        year: float(_CGT_TAXPAYERS) for year in range(_CGT_BASE_YEAR, _MAX_YEAR + 1)
    }

    return [
        Target(
            name="hmrc/capital_gains_total",
            variable="capital_gains",
            source="hmrc",
            unit=Unit.GBP,
            values=gains_values,
            reference_url=reference_url,
            forecast_vintage="2023-24 outturn",
            custom_compute=compute_capital_gains_total,
        ),
        Target(
            name="hmrc/cgt_taxpayers",
            variable="capital_gains",
            source="hmrc",
            unit=Unit.COUNT,
            values=count_values,
            is_count=True,
            reference_url=reference_url,
            forecast_vintage="2023-24 outturn",
            custom_compute=compute_cgt_taxpayers,
        ),
    ] + _band_targets(reference_url)
