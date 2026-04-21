"""Tests for the UKHLS transition-rate estimator.

All tests are hermetic: they use an in-memory synthetic long-format
frame shaped like the output of ``load_all_waves``. No real UKHLS
microdata is read here — that stays behind the UKDS licence boundary
in ``policyengine_uk_data/storage/ukhls/``.

The committed aggregated CSVs are also exercised where they exist so
downstream consumers can't silently regress on column naming or
disclosure-control suppression.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from policyengine_uk_data.utils.ukhls_transitions import (
    MIN_CELL_COUNT,
    _age_band,
    _panel_pairs,
    estimate_employment_transitions,
    estimate_income_decile_transitions,
    load_employment_transitions,
    load_income_decile_transitions,
    save_transition_tables,
)


def _synthetic_panel(n_per_wave: int = 2_000, seed: int = 0) -> pd.DataFrame:
    """Build a multi-wave frame with deterministic transitions.

    We plant a known structure so tests can assert transition
    probabilities recover what we put in.
    """
    rng = np.random.default_rng(seed)
    rows = []
    # One cohort followed across three waves. Each wave increments age
    # by 1 and draws a new employment state with probabilities that
    # depend on the previous state.
    pidps = np.arange(1_000_000, 1_000_000 + n_per_wave)
    ages = rng.integers(25, 55, size=n_per_wave)
    sexes = rng.choice([1, 2], size=n_per_wave)
    state_prev = np.where(rng.random(n_per_wave) < 0.7, "IN_WORK", "UNEMPLOYED")

    def _draw_next(prev: np.ndarray) -> np.ndarray:
        r = rng.random(size=prev.shape[0])
        out = np.where(
            prev == "IN_WORK",
            np.where(r < 0.9, "IN_WORK", "UNEMPLOYED"),
            np.where(r < 0.4, "IN_WORK", "UNEMPLOYED"),
        )
        return out

    for wave in (1, 2, 3):
        rows.append(
            pd.DataFrame(
                {
                    "pidp": pidps,
                    "wave": wave,
                    "year": 2009 + wave - 1,
                    "age_dv": ages + (wave - 1),
                    "sex": sexes,
                    "state": state_prev,
                    "fimngrs_dv": rng.lognormal(mean=7, sigma=1, size=n_per_wave),
                }
            )
        )
        state_prev = _draw_next(state_prev)
    return pd.concat(rows, ignore_index=True)


def test_age_band_bins():
    assert _age_band(15) is None  # below lower edge
    assert _age_band(16) == "16-19"
    assert _age_band(19) == "16-19"
    assert _age_band(20) == "20-24"
    # Final open-ended band runs 75 to 120 inclusive.
    assert _age_band(120) == "75-120"
    assert _age_band(121) is None  # above upper edge


def test_panel_pairs_only_consecutive_waves():
    df = _synthetic_panel(n_per_wave=500)
    pairs = _panel_pairs(df)
    # Waves span 1-3, so we expect pairs (1→2) and (2→3).
    assert set(pairs["wave_t"].unique()) == {1, 2}
    assert set(pairs["wave_t1"].unique()) == {2, 3}
    # Every pair has wave_t1 == wave_t + 1.
    assert (pairs["wave_t1"] - pairs["wave_t"] == 1).all()


def test_employment_transitions_recover_planted_probabilities():
    df = _synthetic_panel(n_per_wave=5_000, seed=42)
    emp = estimate_employment_transitions(df)

    # Aggregate across age/sex since the synthetic data doesn't vary by those.
    totals = emp.groupby("state_from")["count"].sum()
    in_work_to_in_work = (
        emp[(emp.state_from == "IN_WORK") & (emp.state_to == "IN_WORK")]["count"].sum()
        / totals["IN_WORK"]
    )
    # Planted probability was 0.9; recovery within ±3 percentage points.
    assert 0.87 <= in_work_to_in_work <= 0.93


def test_probabilities_sum_to_one_within_state_from():
    df = _synthetic_panel(n_per_wave=4_000, seed=1)
    emp = estimate_employment_transitions(df)
    sums = emp.groupby(["age_band", "sex", "state_from"])["probability"].sum()
    # After the post-suppression renormalisation, every row group must sum to 1
    # (or exactly zero if every cell was suppressed, but that shouldn't happen here).
    assert np.allclose(sums.values, 1.0, atol=1e-9)


def test_low_count_cells_are_suppressed():
    # Build a tiny cohort so almost every cell falls below MIN_CELL_COUNT.
    df = _synthetic_panel(n_per_wave=30, seed=0)
    emp = estimate_employment_transitions(df)
    assert (emp["count"] >= MIN_CELL_COUNT).all()


def test_income_decile_transitions_are_balanced():
    df = _synthetic_panel(n_per_wave=10_000, seed=7)
    dec = estimate_income_decile_transitions(df)
    # Probabilities must sum to 1 within (age, sex, decile_from).
    sums = dec.groupby(["age_band", "sex", "decile_from"])["probability"].sum()
    assert np.allclose(sums.values, 1.0, atol=1e-9)
    # Decile values are 1-10 inclusive.
    assert set(dec["decile_from"]) <= set(range(1, 11))
    assert set(dec["decile_to"]) <= set(range(1, 11))


def test_save_and_load_roundtrip(tmp_path: Path):
    df = _synthetic_panel(n_per_wave=5_000, seed=3)
    paths = save_transition_tables(tmp_path, df=df)
    assert paths["employment_state_transitions"].exists()
    assert paths["income_decile_transitions"].exists()

    nested_emp = load_employment_transitions(paths["employment_state_transitions"])
    # Keys are (age_band, sex, state_from) tuples with stringified components.
    sample_key = next(iter(nested_emp))
    assert isinstance(sample_key, tuple) and len(sample_key) == 3
    # Each inner dict's values sum to 1.
    for _, probs in nested_emp.items():
        assert abs(sum(probs.values()) - 1.0) < 1e-9

    nested_dec = load_income_decile_transitions(paths["income_decile_transitions"])
    for (_, _, decile_from), probs in nested_dec.items():
        assert 1 <= int(decile_from) <= 10
        assert abs(sum(probs.values()) - 1.0) < 1e-9


@pytest.mark.skipif(
    not (
        Path(__file__).parents[2]
        / "policyengine_uk_data"
        / "storage"
        / "ukhls_employment_state_transitions.csv"
    ).exists(),
    reason="Committed aggregate transition CSV not present in this checkout",
)
def test_committed_aggregate_employment_csv_is_usable():
    nested = load_employment_transitions()
    assert nested, "Committed transition table must not be empty"
    # Pick any row group and verify it sums to 1.
    probs = next(iter(nested.values()))
    assert abs(sum(probs.values()) - 1.0) < 1e-9


@pytest.mark.skipif(
    not (
        Path(__file__).parents[2]
        / "policyengine_uk_data"
        / "storage"
        / "ukhls_income_decile_transitions.csv"
    ).exists(),
    reason="Committed aggregate decile transition CSV not present in this checkout",
)
def test_committed_aggregate_decile_csv_is_usable():
    nested = load_income_decile_transitions()
    assert nested
    # Decile keys round-trip as ints.
    for (_, _, decile_from), _ in list(nested.items())[:3]:
        assert isinstance(decile_from, int)
