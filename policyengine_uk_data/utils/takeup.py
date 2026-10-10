"""Shared take-up draw logic with reported-recipient anchoring.

Ported from ``policyengine_us_data/utils/takeup.py``. The core idea: when a
survey respondent reports receiving a benefit, they are by construction a
taker-up; they should be assigned takeup=True with certainty, and the
remaining random fill should hit the target aggregate takeup rate across the
non-reporting eligibles. Pure random draws (the previous UK pattern) ignore
this information and produce noisier calibration.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def assign_takeup_with_reported_anchors(
    draws: np.ndarray,
    rate: float,
    reported_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Apply the SSI/SNAP-style reported-first takeup pattern.

    Reported recipients are always assigned ``takeup=True``. Remaining
    non-reporters are filled probabilistically to reach the target count
    implied by ``rate`` across the full population.

    Args:
        draws: Uniform draws in [0, 1), one per entity.
        rate: Target aggregate takeup rate in [0, 1].
        reported_mask: Boolean array, same length as ``draws``. ``True``
            where the survey reports a positive benefit amount. If ``None``,
            the function falls back to a plain ``draws < rate`` fill.

    Returns:
        Boolean array of the same length as ``draws``, ``True`` for entities
        that take up.
    """
    draws = np.asarray(draws, dtype=np.float64)
    rate = float(rate)

    if reported_mask is None:
        return draws < rate

    reported_mask = np.asarray(reported_mask, dtype=bool)
    if len(reported_mask) != len(draws):
        raise ValueError("reported_mask and draws must align")

    result = reported_mask.copy()
    target_count = int(rate * len(draws))
    remaining_needed = max(0, target_count - int(reported_mask.sum()))
    non_reporters = ~reported_mask
    if not non_reporters.any() or remaining_needed == 0:
        return result

    adjusted_rate = remaining_needed / int(non_reporters.sum())
    result |= non_reporters & (draws < adjusted_rate)
    return result


def solve_fill_probability(
    rate: float,
    weights: np.ndarray,
    eligible: np.ndarray,
    reported: np.ndarray,
) -> float:
    """Claim probability for eligible non-reporters that makes weighted
    take-up among eligible entities equal ``rate`` in expectation.

    Reporters always claim. If they alone exceed ``rate`` the probability is
    0; if every eligible non-reporter claiming still falls short it is 1.
    """
    weights = np.asarray(weights, dtype=np.float64)
    eligible = np.asarray(eligible, dtype=bool)
    reported = np.asarray(reported, dtype=bool)
    eligible_weight = weights[eligible].sum()
    reporting_weight = weights[eligible & reported].sum()
    remaining_weight = weights[eligible & ~reported].sum()
    if remaining_weight <= 0:
        return 0.0
    needed = float(rate) * eligible_weight - reporting_weight
    # A vanishing remaining weight sends the ratio to infinity; clip handles it.
    with np.errstate(over="ignore", divide="ignore"):
        probability = needed / remaining_weight
    return float(np.clip(probability, 0.0, 1.0))


# Legacy means-tested benefits, in the order their names are joined into a
# combination key in parameters/take_up/uc_managed_migration.yaml.
LEGACY_BENEFITS = (
    "child_tax_credit",
    "working_tax_credit",
    "housing_benefit",
    "esa_income",
    "income_support",
    "jsa_income",
)
# Seeds for the Move to Universal Credit draw. Each has its own generator, so
# the draws behind every other take-up flag are unchanged.
UC_MANAGED_MIGRATION_SEED = 492
UC_MANAGED_MIGRATION_SPI_SEED = 493


def reported_benunit_mask(
    person: pd.DataFrame, benunit: pd.DataFrame, column: str
) -> np.ndarray:
    """Benefit units with a member reporting a positive ``column``."""
    reporters = person.loc[person[column] > 0, "person_benunit_id"].unique()
    return benunit["benunit_id"].isin(reporters).values


def legacy_benefit_combination(
    person: pd.DataFrame, benunit: pd.DataFrame
) -> np.ndarray:
    """The legacy benefits each benefit unit reports, joined by "+".

    An empty string marks a benefit unit that reports none of them.
    """
    reported = [
        reported_benunit_mask(person, benunit, f"{benefit}_reported")
        for benefit in LEGACY_BENEFITS
    ]
    return np.array(
        [
            "+".join(b for b, has in zip(LEGACY_BENEFITS, row) if has)
            for row in zip(*reported)
        ],
        dtype=object,
    )


def assign_uc_claim_at_legacy_closure(
    person: pd.DataFrame,
    benunit: pd.DataFrame,
    rates: dict[str, float],
    seed: int,
) -> np.ndarray:
    """Draw ``would_claim_uc_at_legacy_closure`` for each benefit unit.

    policyengine-uk reads it once a legacy benefit the unit reports has
    closed: the unit then claims Universal Credit if this is True and loses
    its legacy awards either way. A unit reporting legacy benefits but not
    Universal Credit claims with DWP's Move to Universal Credit claim rate for
    its combination of benefits (``rates``; "all" where DWP reports none).
    Units reporting Universal Credit, and units reporting no legacy benefit,
    are True, the model's default.

    Args:
        person: Person table with ``person_benunit_id`` and the
            ``<benefit>_reported`` columns.
        benunit: Benefit unit table with ``benunit_id``.
        rates: Claim rate by combination, from
            ``load_uc_managed_migration_claim_rates``.
        seed: Seed for this draw's own generator.

    Returns:
        Boolean array aligned with ``benunit``.
    """
    combination = legacy_benefit_combination(person, benunit)
    on_uc = reported_benunit_mask(person, benunit, "universal_credit_reported")
    rate = np.array([rates.get(c, rates["all"]) for c in combination])
    draws = np.random.default_rng(seed).random(len(benunit))
    return on_uc | (combination == "") | (draws < rate)
