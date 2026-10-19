"""HMRC capital gains targets.

The Enhanced FRS imputes capital gains, but nothing in the calibration
constrains how those gains are *distributed* across households unless it
is told HMRC's distribution. HMRC's Capital Gains Tax statistics publish,
for individuals with a CGT liability, the number of taxpayers and their
gains by size of gain (Table 2.1a). Each size band at or above the annual
exempt amount (AEA) becomes a target pair, and their sums become two
aggregate targets:

* ``hmrc/capital_gains_total`` — total chargeable gains of CGT taxpayers.
* ``hmrc/cgt_taxpayers`` — the number of CGT taxpayers.
* ``hmrc/cgt_taxpayers_band_{lower}`` and
  ``hmrc/capital_gains_band_{lower}`` — the same two statistics for each
  size-of-gain band.

Individuals only: the model's gains belong to people, so the targets come
from Table 2.1a (individuals), not from the all-taxpayer headline, which
also counts trusts and personal representatives (Table 1: 584,000
taxpayers and £127.3bn in 2024-25, of which trusts are 32,000 and
£8.1bn). The aggregates are the sums of the targeted bands, so the
aggregate and band targets cannot disagree.

Every target uses a ``custom_compute``. The default count path in
``build_loss_matrix._compute_simple_count`` counts people with
``capital_gains > 0``, which is not HMRC's concept: HMRC counts CGT
*taxpayers*, i.e. people whose gains exceed the AEA. The amounts are
gated on the same AEA condition, for the same reason — HMRC's total is
the gains of CGT-liable taxpayers, not all gains realised in the economy.
Gating only the count and leaving the amount ungated would target a mean
gain that is a ratio of two different populations, and would let the
optimiser satisfy the amount with sub-threshold gains that HMRC never
counts.

The AEA is policy-dependent (£12,300 for 2022-23, £6,000 for 2023-24,
£3,000 from 2024-25) and is read from PolicyEngine's own parameter tree
for the simulation year, so the gate tracks the policy in force.
``_AEA_FALLBACK`` is used only if the parameter lookup fails.

Year handling: HMRC's figures are the 2024-25 outturn. PolicyEngine UK's
year N is tax year N to N+1, so the outturn is the 2024 value: the survey
year of the FRS release the build starts from, and a year with the same
£3,000 AEA as the years the calibration runs for. Amounts are projected
to 2029 with PolicyEngine's ``capital_gains`` uprating factors so the
targets constrain the projection years; taxpayer counts are held flat, as
the repo has no administrative basis for forecasting CGT taxpayer numbers
and count uprating factors would be a nominal-income index applied to a
headcount.

Caveat: 2024-25 is an unusual year. HMRC's commentary attributes part of
the rise in gains (£66.6bn to £119.3bn for individuals) to disposals
brought forward ahead of the 30 October 2024 rate rise. Carrying that
year forward with a nominal-income index assumes gains stay at that
level.

Source: https://www.gov.uk/government/statistics/capital-gains-tax-statistics
"""

import logging

import numpy as np

from policyengine_uk_data.targets.schema import Target, Unit
from policyengine_uk_data.targets.sources._common import load_config

logger = logging.getLogger(__name__)

# HMRC CGT statistics, tax year 2024-25 outturn. PolicyEngine UK's year N
# is tax year N to N+1, so the outturn is the 2024 value.
_CGT_BASE_YEAR = 2024
_MAX_YEAR = 2029
_OUTTURN_LABEL = "2024-25 outturn"

# Annual exempt amount by tax year start, used only if the PolicyEngine
# parameter lookup fails.
_AEA_FALLBACK = {
    2022: 12_300,
    2023: 6_000,
}
_AEA_FALLBACK_DEFAULT = 3_000

_PARAMETER_PATH = "gov.hmrc.cgt.annual_exempt_amount"

# HMRC Table 2.1a (committed as capital_gains_size_distribution_hmrc.csv)
# publishes individuals' taxpayer counts and gains by size of gain.
_SIZE_BANDS_FILE = "capital_gains_size_distribution_hmrc.csv"
# The table's tax year had a £3,000 AEA. Its one band under that limit
# holds the few people liable with smaller gains (3,000 people and £1m in
# 2024-25); the AEA gate cannot represent them, so the targets start at
# the AEA.
_MIN_BAND_LOWER = 3_000


def load_size_bands():
    """HMRC size-of-gain bands at or above the table year's AEA."""
    import pandas as pd

    from policyengine_uk_data.storage import STORAGE_FOLDER

    bands = pd.read_csv(STORAGE_FOLDER / _SIZE_BANDS_FILE, comment="#")
    bands = bands[bands.lower_limit >= _MIN_BAND_LOWER].reset_index(drop=True)
    bands["upper_limit"] = bands.lower_limit.shift(-1).fillna(np.inf)
    return bands


def _outturn_totals() -> tuple[float, float]:
    """Taxpayers and gains summed over the targeted bands."""
    bands = load_size_bands()
    return (
        float(bands.taxpayers_thousands.sum()) * 1e3,
        float(bands.gains_gbp_millions.sum()) * 1e6,
    )


# Number of CGT taxpayers and their total chargeable gains, 2024-25,
# individuals with gains above the AEA.
_CGT_TAXPAYERS, _CGT_TOTAL_GAINS = _outturn_totals()


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
# The aggregate count and total cannot, on their own, constrain the *shape*
# of the gains distribution: an earlier enhanced FRS carried only ~1.6% of
# gains in gains of £1m+ against HMRC's ~61%, while still being able to
# satisfy an aggregate total. Each band of Table 2.1a from the AEA up
# therefore becomes a target pair. The imputation in
# datasets/imputations/capital_gains.py gives every band records to weight,
# and the calibration starts with each band's taxpayer count on its target
# (see band_count_target_names).


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


def _band_count_name(lower: float) -> str:
    return f"hmrc/cgt_taxpayers_band_{int(lower)}"


def band_count_target_names() -> list[str]:
    """Names of the per-band taxpayer-count targets.

    The bands partition CGT taxpayers by size of gain, so the households
    behind one of these targets are behind no other. The calibration
    scales its starting weights so that each is met before the first
    epoch (``calibrate_local_areas(prior_rake_targets=...)``).
    """
    return [_band_count_name(lower) for lower in load_size_bands().lower_limit]


def _band_targets(reference_url: str) -> list[Target]:
    targets = []
    for row in load_size_bands().itertuples():
        lower = float(row.lower_limit)
        upper = float(row.upper_limit)
        count = float(row.taxpayers_thousands) * 1e3
        gains = float(row.gains_gbp_millions) * 1e6
        targets.append(
            Target(
                name=_band_count_name(lower),
                variable="capital_gains",
                source="hmrc",
                unit=Unit.COUNT,
                values={year: count for year in range(_CGT_BASE_YEAR, _MAX_YEAR + 1)},
                is_count=True,
                reference_url=reference_url,
                forecast_vintage=_OUTTURN_LABEL,
                custom_compute=_make_band_count_compute(lower, upper),
            )
        )
        targets.append(
            Target(
                name=f"hmrc/capital_gains_band_{int(lower)}",
                variable="capital_gains",
                source="hmrc",
                unit=Unit.GBP,
                values=_project_gains(gains),
                reference_url=reference_url,
                forecast_vintage=_OUTTURN_LABEL,
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
            forecast_vintage=_OUTTURN_LABEL,
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
            forecast_vintage=_OUTTURN_LABEL,
            custom_compute=compute_cgt_taxpayers,
        ),
    ] + _band_targets(reference_url)
