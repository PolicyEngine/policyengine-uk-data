"""Population fidelity regression tests for the calibrated dataset.

Guards against the April 2026 calibration drift (issue #217) where the
weighted UK population inflated ~6.5% above the ONS target. The drift
was pulled back to ~1.6% by the data-pipeline improvements that landed
in #362 (stage-2 QRF), #363 (TFC target refresh), and #359 (reported-
anchor takeup). These tests lock in that gain so future calibration
changes can't regress past current fidelity without a test failure.

Extracted from PolicyEngine/policyengine-uk-data#310 (Vahid Ahmadi).
"""

from __future__ import annotations

import warnings

import numpy as np
from policyengine_uk_data.datasets.frs_release import CURRENT_FRS_RELEASE

POPULATION_TARGET = 69.5  # ONS 2024-based projection for 2025, millions
TOLERANCE = 0.04  # 4% — covers ~1.6%-3.3% stochastic calibration variance
MIN_HOUSEHOLDS_M = 25
MAX_HOUSEHOLDS_M = 34
PERIOD = CURRENT_FRS_RELEASE.calibration_year


def _raw(micro_series):
    """Extract the raw numpy array from a MicroSeries without triggering
    the `.values` deprecation warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return np.array(micro_series.values)


def test_weighted_population_matches_ons_target(baseline):
    """Weighted UK population is within 3 % of the ONS projection."""
    population = baseline.calculate("people", PERIOD).sum() / 1e6
    assert abs(population / POPULATION_TARGET - 1) < TOLERANCE, (
        f"Weighted population {population:.1f}M is >{TOLERANCE:.0%} "
        f"from ONS target {POPULATION_TARGET:.1f}M."
    )


def test_household_count_reasonable(baseline):
    """Total weighted households fall inside a broad CI smoke-test range."""
    hw = _raw(baseline.calculate("household_weight", PERIOD))
    total_hh = hw.sum() / 1e6
    assert MIN_HOUSEHOLDS_M < total_hh < MAX_HOUSEHOLDS_M, (
        f"Total weighted households {total_hh:.1f}M outside "
        f"{MIN_HOUSEHOLDS_M}-{MAX_HOUSEHOLDS_M}M range."
    )


def test_population_not_inflated(baseline):
    """Population stays below the pre-April-2026 inflated level (72 M)."""
    population = baseline.calculate("people", PERIOD).sum() / 1e6
    assert population < 72, (
        f"Population {population:.1f}M exceeds 72M — calibration has "
        "regressed toward the pre-#217 overshoot."
    )


def test_country_populations_sum_to_uk(baseline):
    """England + Scotland + Wales + NI populations sum to the UK total."""
    people = baseline.calculate("people", PERIOD)
    country = baseline.calculate("country", map_to="person")

    uk_pop = people.sum()
    country_sum = sum(people[country == c].sum() for c in country.unique())

    assert abs(country_sum / uk_pop - 1) < 0.001, (
        f"Country populations sum to {country_sum / 1e6:.1f}M "
        f"but UK total is {uk_pop / 1e6:.1f}M."
    )
