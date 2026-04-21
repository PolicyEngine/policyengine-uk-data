"""Estimate year-on-year transition rates from UKHLS panel data.

Given the UKHLS indresp files loaded via :mod:`datasets.ukhls`, this
module pairs each respondent in consecutive waves to compute the
within-person transition probabilities that ``age_dataset`` and
``apply_employment_transitions`` consume.

Only aggregated transition tables are returned / saved — individual
rows never leave the estimator. A disclosure-control cell-suppression
rule ensures that any ``(age_band × sex × state_from × state_to)``
cell observed fewer than ``MIN_CELL_COUNT`` times is dropped from the
saved output, matching the ONS Safe Setting convention for published
microdata derivatives.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from policyengine_uk_data.datasets.ukhls import (
    WAVE_LETTERS,
    annotate_four_state,
    load_all_waves,
)

logger = logging.getLogger(__name__)


# Disclosure control: suppress any cell observed fewer than this many
# times before saving. ONS/UKDS safeguarded microdata convention is 10.
MIN_CELL_COUNT = 10

# 5-year age bands keep cell counts comfortably above MIN_CELL_COUNT
# even for sparse states (e.g. retired 30-year-olds).
AGE_BAND_EDGES = [16, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 121]


def _age_band(age: float) -> str | None:
    try:
        a = int(age)
    except (TypeError, ValueError):
        return None
    if a < AGE_BAND_EDGES[0] or a >= AGE_BAND_EDGES[-1]:
        return None
    for lo, hi in zip(AGE_BAND_EDGES, AGE_BAND_EDGES[1:]):
        if lo <= a < hi:
            return f"{lo}-{hi - 1}"
    return None


def _panel_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """Return consecutive-wave pairs keyed on ``pidp``.

    Each row in the output is a (pidp, wave_t → wave_{t+1}) transition,
    with ``_t`` and ``_t1`` suffixes on every numeric column so the
    caller can compare before and after.
    """
    df = df.sort_values(["pidp", "wave"])
    left = df.rename(columns={c: f"{c}_t" for c in df.columns if c != "pidp"})
    right = df.rename(columns={c: f"{c}_t1" for c in df.columns if c != "pidp"})
    right["_join_wave"] = right["wave_t1"] - 1
    merged = left.merge(
        right,
        left_on=["pidp", "wave_t"],
        right_on=["pidp", "_join_wave"],
        how="inner",
    )
    merged = merged.drop(columns=["_join_wave"])
    return merged


def estimate_employment_transitions(
    df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Estimate P(state_{t+1} | state_t, age_band, sex).

    Args:
        df: optional pre-loaded UKHLS frame with ``state`` annotated.
            ``None`` loads all waves from disk.

    Returns:
        Long-format DataFrame with columns
        ``age_band, sex, state_from, state_to, count, probability``.
        Cells with ``count < MIN_CELL_COUNT`` are dropped, and
        probabilities are re-normalised within the surviving rows.
    """
    if df is None:
        df = load_all_waves()
    if "state" not in df.columns:
        df = annotate_four_state(df)

    pairs = _panel_pairs(df)
    pairs["age_band"] = pairs["age_dv_t"].map(_age_band)
    pairs = pairs.dropna(subset=["age_band"])
    pairs = pairs[pairs["state_t"] != "MISSING"]
    pairs = pairs[pairs["state_t1"] != "MISSING"]
    pairs = pairs[pairs["sex_t"].isin([1, 2])]

    grouped = (
        pairs.groupby(["age_band", "sex_t", "state_t", "state_t1"], observed=True)
        .size()
        .rename("count")
        .reset_index()
        .rename(
            columns={"sex_t": "sex", "state_t": "state_from", "state_t1": "state_to"}
        )
    )
    grouped["sex"] = grouped["sex"].map({1: "MALE", 2: "FEMALE"})

    # Disclosure control: drop low-count cells.
    grouped = grouped[grouped["count"] >= MIN_CELL_COUNT].reset_index(drop=True)

    # Re-normalise so probabilities sum to 1 within each (age_band, sex,
    # state_from) row group after suppression.
    totals = grouped.groupby(["age_band", "sex", "state_from"])["count"].transform(
        "sum"
    )
    grouped["probability"] = grouped["count"] / totals
    return grouped


def estimate_income_decile_transitions(
    df: pd.DataFrame | None = None,
    *,
    income_col: str = "fimngrs_dv",
) -> pd.DataFrame:
    """Estimate P(decile_{t+1} | decile_t, age_band, sex).

    Income is assigned a within-wave decile rank so that the estimator
    is comparable across years (absolute income grows with inflation,
    but the decile structure is scale-invariant).

    Args:
        df: optional pre-loaded UKHLS frame. ``None`` loads from disk.
        income_col: which monthly income column to rank. Defaults to
            total gross personal income.

    Returns:
        Long-format DataFrame with columns
        ``age_band, sex, decile_from, decile_to, count, probability``.
        Same suppression rules as :func:`estimate_employment_transitions`.
    """
    if df is None:
        df = load_all_waves()
    df = df.copy()
    df[income_col] = pd.to_numeric(df[income_col], errors="coerce")
    # Negative or missing → drop (those are imputation flags / refusals).
    df = df[df[income_col].notna()]
    df = df[df[income_col] >= 0]

    # Within-wave decile rank so inflation doesn't bias transitions.
    df["decile"] = df.groupby("wave")[income_col].transform(
        lambda s: pd.qcut(s.rank(method="first"), q=10, labels=False) + 1
    )

    pairs = _panel_pairs(df[["pidp", "wave", "age_dv", "sex", "decile"]])
    pairs["age_band"] = pairs["age_dv_t"].map(_age_band)
    pairs = pairs.dropna(subset=["age_band", "decile_t", "decile_t1"])
    pairs = pairs[pairs["sex_t"].isin([1, 2])]

    grouped = (
        pairs.groupby(["age_band", "sex_t", "decile_t", "decile_t1"], observed=True)
        .size()
        .rename("count")
        .reset_index()
        .rename(
            columns={
                "sex_t": "sex",
                "decile_t": "decile_from",
                "decile_t1": "decile_to",
            }
        )
    )
    grouped["sex"] = grouped["sex"].map({1: "MALE", 2: "FEMALE"})
    grouped = grouped[grouped["count"] >= MIN_CELL_COUNT].reset_index(drop=True)
    totals = grouped.groupby(["age_band", "sex", "decile_from"])["count"].transform(
        "sum"
    )
    grouped["probability"] = grouped["count"] / totals
    return grouped


def save_transition_tables(
    output_dir: Path | str,
    df: pd.DataFrame | None = None,
) -> dict[str, Path]:
    """Compute all transition tables and write them to ``output_dir``.

    Produces CSVs small enough to commit:
        - ukhls_employment_state_transitions.csv
        - ukhls_income_decile_transitions.csv

    The ``ukhls_`` prefix matches the
    ``!policyengine_uk_data/storage/*.csv`` gitignore negation rule so
    that the aggregated outputs are tracked while the raw .dta files
    in the ``ukhls/`` subfolder stay out of git.

    Returns a mapping from table name to written path.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if df is None:
        df = load_all_waves()

    emp = estimate_employment_transitions(df)
    dec = estimate_income_decile_transitions(df)

    emp_path = output_dir / "ukhls_employment_state_transitions.csv"
    dec_path = output_dir / "ukhls_income_decile_transitions.csv"
    emp.to_csv(emp_path, index=False)
    dec.to_csv(dec_path, index=False)

    return {
        "employment_state_transitions": emp_path,
        "income_decile_transitions": dec_path,
    }


def load_employment_transitions(
    path: Path | str | None = None,
) -> dict[tuple[str, str, str], dict[str, float]]:
    """Read the committed employment-transition CSV into a nested dict.

    Returns ``{(age_band, sex, state_from): {state_to: probability, ...}}``
    ready to be consumed by a future
    ``apply_employment_transitions(ukhls_rates=...)`` override.
    """
    from policyengine_uk_data.storage import STORAGE_FOLDER

    path = (
        Path(path)
        if path
        else STORAGE_FOLDER / "ukhls_employment_state_transitions.csv"
    )
    if not path.exists():
        raise FileNotFoundError(
            f"Transition table not found at {path}. Run "
            "save_transition_tables() after placing UKHLS data in storage/ukhls/."
        )
    df = pd.read_csv(path)
    nested: dict[tuple[str, str, str], dict[str, float]] = {}
    for (age_band, sex, state_from), group in df.groupby(
        ["age_band", "sex", "state_from"], observed=True
    ):
        nested[(str(age_band), str(sex), str(state_from))] = dict(
            zip(group["state_to"].astype(str), group["probability"].astype(float))
        )
    return nested


def load_income_decile_transitions(
    path: Path | str | None = None,
) -> dict[tuple[str, str, int], dict[int, float]]:
    """Read the committed income-decile-transition CSV into a nested dict."""
    from policyengine_uk_data.storage import STORAGE_FOLDER

    path = (
        Path(path) if path else STORAGE_FOLDER / "ukhls_income_decile_transitions.csv"
    )
    if not path.exists():
        raise FileNotFoundError(
            f"Transition table not found at {path}. Run "
            "save_transition_tables() after placing UKHLS data in storage/ukhls/."
        )
    df = pd.read_csv(path)
    nested: dict[tuple[str, str, int], dict[int, float]] = {}
    for (age_band, sex, decile_from), group in df.groupby(
        ["age_band", "sex", "decile_from"], observed=True
    ):
        nested[(str(age_band), str(sex), int(decile_from))] = dict(
            zip(group["decile_to"].astype(int), group["probability"].astype(float))
        )
    return nested
