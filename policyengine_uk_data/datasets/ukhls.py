"""Loader for Understanding Society (UKHLS) main-survey microdata.

The microdata lives under UKDS End User Licence and is never committed
to git. Raw files are expected at
``policyengine_uk_data/storage/ukhls/`` (usually a symlink into the
user's UKDS download folder).

This module only emits **aggregated** outputs. Individual-level
information never leaves the loader: caller APIs either return full
DataFrames (for in-process use by the transition-rate estimator) or
summary tables that have been grouped by non-disclosive cells.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from policyengine_uk_data.storage import STORAGE_FOLDER

logger = logging.getLogger(__name__)


UKHLS_DIR = STORAGE_FOLDER / "ukhls"

# Wave-letter → wave number mapping. UKHLS uses single-letter prefixes:
# a=Wave 1 (2009-10), b=Wave 2 (2010-11), ..., o=Wave 15 (2023-24).
_LETTERS = "abcdefghijklmno"
WAVE_LETTERS: dict[str, int] = {ch: i + 1 for i, ch in enumerate(_LETTERS)}
WAVE_NUMBERS: dict[int, str] = {v: k for k, v in WAVE_LETTERS.items()}

# Each UKHLS wave spans two calendar years; we adopt the year of first
# interview (same convention the DWP Income Dynamics publication uses).
WAVE_YEAR_START: dict[int, int] = {i + 1: 2009 + i for i in range(len(_LETTERS))}

# Minimal column set for income / employment transition analysis. Using
# a curated list (not `read_stata()` with no filter) because the full
# file has ~1,400 columns and loading even one wave in full is ~10 s.
BASE_COLUMNS_UNPREFIXED = [
    "age_dv",
    "sex",
    "jbstat",
    "fimnlabgrs_dv",  # total monthly labour income (gross)
    "fimngrs_dv",  # total monthly personal income (gross)
    "fimnsben_dv",  # social benefit income (monthly)
    "gor_dv",  # Government Office Region
    "hidp",  # within-wave household identifier
]

# JBSTAT is an ordinal enum — keep the raw codes but expose a compact
# harmonised four-state label that's easier to estimate transitions on.
JBSTAT_LABELS: dict[int, str] = {
    1: "SELF_EMPLOYED",
    2: "EMPLOYED",
    3: "UNEMPLOYED",
    4: "RETIRED",
    5: "OTHER_INACTIVE",  # maternity leave
    6: "OTHER_INACTIVE",  # family care
    7: "STUDENT",
    8: "OTHER_INACTIVE",  # LT sick / disabled
    9: "OTHER_INACTIVE",  # govt training
    10: "SELF_EMPLOYED",  # unpaid family business
    11: "OTHER_INACTIVE",
    12: "OTHER_INACTIVE",
    13: "OTHER_INACTIVE",
    97: "OTHER_INACTIVE",
}

# Collapsed four-state labour market state used for transition matrices.
FOUR_STATE_MAP: dict[str, str] = {
    "SELF_EMPLOYED": "IN_WORK",
    "EMPLOYED": "IN_WORK",
    "UNEMPLOYED": "UNEMPLOYED",
    "RETIRED": "RETIRED",
    "STUDENT": "INACTIVE",
    "OTHER_INACTIVE": "INACTIVE",
}


def _wave_path(wave: int | str) -> Path:
    """Return the path to an indresp file for a given wave (number or letter)."""
    letter = wave if isinstance(wave, str) else WAVE_NUMBERS[int(wave)]
    path = UKHLS_DIR / f"{letter}_indresp.dta"
    if not path.exists():
        raise FileNotFoundError(
            f"UKHLS indresp for wave {wave!r} not found at {path}. "
            "Place the UKDS download at policyengine_uk_data/storage/ukhls."
        )
    return path


def _prefixed(letter: str, base_columns: list[str]) -> list[str]:
    return [f"{letter}_{col}" for col in base_columns]


def load_wave(
    wave: int | str,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Load a single UKHLS wave's individual-response file.

    Returns a DataFrame keyed on ``pidp`` (cross-wave panel identifier).
    All returned column names are stripped of their wave-letter prefix
    so that caller code can iterate over waves without string-munging.

    Args:
        wave: wave number (1-15) or wave letter (``"a"`` … ``"o"``).
        columns: list of unprefixed column names to load. ``None`` uses
            :data:`BASE_COLUMNS_UNPREFIXED`.

    Returns:
        DataFrame with columns ``pidp, wave, year, <requested columns>``.
        Wave-specific prefixes have been removed from the requested
        columns so that ``a_age_dv`` appears as ``age_dv``.
    """
    letter = wave if isinstance(wave, str) else WAVE_NUMBERS[int(wave)]
    wave_num = WAVE_LETTERS[letter]
    path = _wave_path(letter)
    if columns is None:
        columns = BASE_COLUMNS_UNPREFIXED

    prefixed = _prefixed(letter, columns)
    read_cols = ["pidp"] + prefixed
    df = pd.read_stata(path, convert_categoricals=False, columns=read_cols)
    rename = {pfx: base for pfx, base in zip(prefixed, columns)}
    df = df.rename(columns=rename)
    df["wave"] = wave_num
    df["year"] = WAVE_YEAR_START[wave_num]
    return df


def load_all_waves(
    columns: list[str] | None = None,
    waves: list[int] | None = None,
) -> pd.DataFrame:
    """Stack multiple UKHLS waves into one long-format frame.

    The returned frame is keyed on ``(pidp, wave)`` so the caller can
    pair consecutive waves for transition analysis.
    """
    waves = waves or list(WAVE_LETTERS.values())
    frames = []
    for w in waves:
        try:
            frames.append(load_wave(w, columns=columns))
        except FileNotFoundError as exc:
            logger.warning("Skipping wave %s: %s", w, exc)
    if not frames:
        raise FileNotFoundError(
            "No UKHLS waves could be loaded. Check that "
            f"{UKHLS_DIR} contains *_indresp.dta files."
        )
    return pd.concat(frames, ignore_index=True)


def four_state_label(jbstat_code: float) -> str:
    """Map an integer jbstat code to the compact four-state label.

    Missing / refused / inapplicable codes (negative values) return
    ``"MISSING"`` so callers can filter them explicitly.
    """
    try:
        code = int(jbstat_code)
    except (TypeError, ValueError):
        return "MISSING"
    if code < 0:
        return "MISSING"
    detail = JBSTAT_LABELS.get(code, "OTHER_INACTIVE")
    return FOUR_STATE_MAP.get(detail, "INACTIVE")


def annotate_four_state(df: pd.DataFrame) -> pd.DataFrame:
    """Attach a ``state`` column derived from ``jbstat`` to ``df``."""
    if "jbstat" not in df.columns:
        raise KeyError("jbstat column required to compute four-state label")
    df = df.copy()
    df["state"] = df["jbstat"].map(four_state_label)
    return df
