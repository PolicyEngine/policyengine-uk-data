"""ONS National Life Tables loader — age × sex mortality rates.

Loads the single-year ``qx`` (probability of death between exact age x
and x+1) column from the ONS UK National Life Tables. The ONS file is
organised as 3-year rolling periods (e.g. ``"2022-2024"``) with one
sheet per period; each sheet has two side-by-side blocks for Males and
Females.

Used by ``utils.demographic_ageing`` to replace the placeholder
mortality rates shipped in #346 with real ONS data.

Source:
- https://www.ons.gov.uk/peoplepopulationandcommunity/birthsdeathsandmarriages/lifeexpectancies/datasets/nationallifetablesunitedkingdomreferencetables
"""

from __future__ import annotations

import io
import logging
from functools import lru_cache
from pathlib import Path
from typing import Mapping

import pandas as pd
import requests

from policyengine_uk_data.targets.sources._common import HEADERS, STORAGE

logger = logging.getLogger(__name__)

# Direct file URL for the current UK National Life Tables release.
# Kept here rather than derived so the build is reproducible against a
# known dataset revision. Updating the URL is an intentional, reviewable
# step.
_ONS_NLT_URL = (
    "https://www.ons.gov.uk/file?uri="
    "/peoplepopulationandcommunity/birthsdeathsandmarriages/"
    "lifeexpectancies/datasets/nationallifetablesunitedkingdomreferencetables/"
    "current/nltuk198020223.xlsx"
)

_ONS_NLT_REF = (
    "https://www.ons.gov.uk/peoplepopulationandcommunity/"
    "birthsdeathsandmarriages/lifeexpectancies/datasets/"
    "nationallifetablesunitedkingdomreferencetables"
)

_LOCAL_FILENAME = "ons_national_life_tables.xlsx"

# ONS rolling-period sheets are named like "2022-2024". The loader accepts
# either the canonical string or just the end year (2024 → "2022-2024").
# A period label is not a stable key across releases, so we derive it.
MALE = "MALE"
FEMALE = "FEMALE"


def _local_path() -> Path:
    return Path(STORAGE) / _LOCAL_FILENAME


def _download_to(local_path: Path) -> None:
    """Fetch the ONS NLT workbook to ``local_path``."""
    logger.info("Downloading ONS National Life Tables from %s", _ONS_NLT_URL)
    r = requests.get(_ONS_NLT_URL, headers=HEADERS, allow_redirects=True, timeout=120)
    r.raise_for_status()
    local_path.parent.mkdir(parents=True, exist_ok=True)
    local_path.write_bytes(r.content)


def _parse_period_sheet(raw: pd.DataFrame, period: str) -> pd.DataFrame:
    """Parse one rolling-period sheet into long-format (age, sex, qx).

    Each period sheet has, after a header block, two side-by-side tables:
    columns 0-5 are Males (age, mx, qx, lx, dx, ex), col 6 is a blank
    separator, columns 7-12 are Females with the same six fields.
    """
    ages = pd.to_numeric(raw.iloc[:, 0], errors="coerce")
    mask = ages.notna() & ages.between(0, 120)
    data = raw[mask].reset_index(drop=True)
    if data.empty:
        raise ValueError(f"Sheet {period!r} contained no numeric age rows.")

    male = pd.DataFrame(
        {
            "period": period,
            "sex": MALE,
            "age": pd.to_numeric(data.iloc[:, 0]).astype(int),
            "qx": pd.to_numeric(data.iloc[:, 2], errors="coerce"),
        }
    )
    female = pd.DataFrame(
        {
            "period": period,
            "sex": FEMALE,
            "age": pd.to_numeric(data.iloc[:, 7]).astype(int),
            "qx": pd.to_numeric(data.iloc[:, 9], errors="coerce"),
        }
    )
    long = pd.concat([male, female], ignore_index=True)
    long = long.dropna(subset=["qx"]).reset_index(drop=True)
    return long


def _iter_period_sheets(xls: pd.ExcelFile):
    """Yield the sheet names that look like rolling period labels, e.g. 2022-2024."""
    for name in xls.sheet_names:
        parts = name.strip().split("-")
        if len(parts) != 2:
            continue
        try:
            start, end = int(parts[0]), int(parts[1])
        except ValueError:
            continue
        if 1900 < start <= end < 2100:
            yield name, start, end


@lru_cache(maxsize=1)
def load_ons_life_tables(
    *, force_download: bool = False, path: str | None = None
) -> pd.DataFrame:
    """Return every ONS NLT rolling period in one long-format frame.

    Columns: ``period`` (str, ``"YYYY-YYYY"``), ``period_start``,
    ``period_end``, ``sex`` (``"MALE"``/``"FEMALE"``), ``age`` (int,
    0-100), ``qx`` (float).
    """
    xlsx_path = Path(path) if path is not None else _local_path()
    if force_download or not xlsx_path.exists():
        _download_to(xlsx_path)

    xls = pd.ExcelFile(xlsx_path, engine="openpyxl")
    frames = []
    for name, start, end in _iter_period_sheets(xls):
        raw = pd.read_excel(xls, sheet_name=name, header=None, engine="openpyxl")
        long = _parse_period_sheet(raw, name)
        long["period_start"] = start
        long["period_end"] = end
        frames.append(long)

    if not frames:
        raise ValueError("ONS NLT workbook contained no recognisable period sheets.")

    return pd.concat(frames, ignore_index=True)


def _resolve_period(tables: pd.DataFrame, year: int | str | None) -> str:
    """Return the period string covering ``year``, preferring an exact
    match, falling back to the most recent period whose end year is <=
    ``year``. ``None`` picks the most recent period overall.
    """
    if year is None:
        latest = tables.sort_values("period_end").iloc[-1]
        return str(latest["period"])

    if isinstance(year, str) and "-" in year:
        if (tables["period"] == year).any():
            return year
        raise KeyError(f"Period {year!r} not present in ONS NLT.")

    y = int(year)
    covers = tables[(tables["period_start"] <= y) & (tables["period_end"] >= y)]
    if not covers.empty:
        return str(covers.sort_values("period_end").iloc[-1]["period"])

    earlier = tables[tables["period_end"] < y]
    if not earlier.empty:
        return str(earlier.sort_values("period_end").iloc[-1]["period"])

    raise KeyError(
        f"No ONS NLT period covers {y}; earliest is {int(tables['period_start'].min())}."
    )


def get_mortality_rates(
    year: int | str | None = None,
    *,
    tables: pd.DataFrame | None = None,
) -> dict[str, dict[int, float]]:
    """Return ``{sex: {age: qx}}`` for the period covering ``year``.

    Args:
        year: Calendar year (e.g. ``2024``) or an explicit period label
            (``"2022-2024"``). ``None`` picks the most recent period.
        tables: Optional pre-loaded output of ``load_ons_life_tables``.
            Useful for tests that monkey-patch the data.

    Returns:
        Nested dict ``{"MALE": {0: 0.0046, 1: 0.0002, ...},
        "FEMALE": {...}}``. Age keys run 0-100 inclusive.
    """
    if tables is None:
        tables = load_ons_life_tables()
    period = _resolve_period(tables, year)
    sub = tables[tables["period"] == period]
    out: dict[str, dict[int, float]] = {MALE: {}, FEMALE: {}}
    for sex, group in sub.groupby("sex"):
        out[str(sex)] = {int(a): float(q) for a, q in zip(group["age"], group["qx"])}
    return out


def get_mortality_rates_unisex(
    year: int | str | None = None,
    *,
    male_share: float = 0.5,
    tables: pd.DataFrame | None = None,
) -> dict[int, float]:
    """Return ``{age: qx}`` averaged across sexes.

    For use with callers that can only accept an age-indexed mapping.
    Default is a flat 0.5 weighting; pass a real sex ratio if you have
    one.
    """
    rates = get_mortality_rates(year, tables=tables)
    ages = sorted(set(rates[MALE]) | set(rates[FEMALE]))
    return {
        a: male_share * rates[MALE].get(a, 0.0)
        + (1 - male_share) * rates[FEMALE].get(a, 0.0)
        for a in ages
    }
