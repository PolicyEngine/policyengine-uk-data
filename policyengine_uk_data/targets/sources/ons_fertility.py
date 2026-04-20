"""ONS age-specific fertility rates (ASFR) loader.

Reads ``Table 10`` of the ONS *Births in England and Wales: registrations*
workbook — ASFR per 1,000 women by mother's 5-year age band, country, and
year, 1938 onwards. Expands broad bands to single-year ages (uniformly
within each band) and converts to per-year birth probabilities so that
``utils.demographic_ageing.age_dataset`` can consume the output directly.

Notes:

- The ONS file covers England, Wales and Elsewhere (the registration
  universe). Scottish and Northern Irish rates are very close and are
  not blended in here to keep the loader single-source.
- Age bands are stored long-form; within a band the rate is applied to
  every single-year age (uniform-within-band is the standard demographic
  convention when single-year ASFR is not available).
- The input column says "Age-specific fertility rate" which ONS defines
  as births per 1,000 women in that age group in mid-year; we divide by
  1,000 to get the per-woman-per-year probability that
  ``age_dataset`` expects.

Source:
- https://www.ons.gov.uk/peoplepopulationandcommunity/birthsdeathsandmarriages/livebirths/datasets/birthsinenglandandwalesbirthregistrations
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import Mapping

import pandas as pd
import requests

from policyengine_uk_data.targets.sources._common import HEADERS, STORAGE

logger = logging.getLogger(__name__)

# Direct file URL for the 2024 ONS Births in England and Wales: registrations
# workbook. This is pinned so the build is reproducible against a known
# dataset revision; updating the URL is an intentional, reviewable step.
_ONS_ASFR_URL = (
    "https://www.ons.gov.uk/file?uri="
    "/peoplepopulationandcommunity/birthsdeathsandmarriages/livebirths/"
    "datasets/birthsinenglandandwalesbirthregistrations/2024/"
    "2024birthregistrations.xlsx"
)

_ONS_ASFR_REF = (
    "https://www.ons.gov.uk/peoplepopulationandcommunity/"
    "birthsdeathsandmarriages/livebirths/datasets/"
    "birthsinenglandandwalesbirthregistrations"
)

_LOCAL_FILENAME = "ons_asfr.xlsx"
_ASFR_SHEET = "Table_10"
_ASFR_HEADER_ROW = 5  # zero-indexed

DEFAULT_COUNTRY = "England, Wales and Elsewhere"
MOTHER_PARENT = "Mother"


def _local_path() -> Path:
    return Path(STORAGE) / _LOCAL_FILENAME


def _download_to(local_path: Path) -> None:
    """Fetch the ONS ASFR workbook to ``local_path``."""
    logger.info("Downloading ONS ASFR from %s", _ONS_ASFR_URL)
    r = requests.get(_ONS_ASFR_URL, headers=HEADERS, allow_redirects=True, timeout=120)
    r.raise_for_status()
    local_path.parent.mkdir(parents=True, exist_ok=True)
    local_path.write_bytes(r.content)


# Open-ended bands ("40 and over") are capped at ``low + OPEN_BAND_SPAN - 1``
# to stay consistent with the 5-year-wide closed bands elsewhere in the
# file. Expanding an open band uniformly across the full fertility window
# would otherwise massively over-state ASFR at ages 45+, where real
# single-year rates are an order of magnitude below the "40 and over"
# bulk rate. Using a 5-year cap matches where the overwhelming majority
# of 40+ births actually occur (40-44 accounts for ~95 % of all 40+
# births in recent UK data).
_OPEN_BAND_SPAN = 5


def _parse_age_band(label: str) -> tuple[int, int] | None:
    """Return inclusive (low, high) single-year ages for an ONS band label.

    Unknown labels (e.g. "All ages") return ``None``.
    """
    s = str(label).strip()
    if not s or s.lower() in {"all ages", "nan"}:
        return None

    m = re.match(r"under\s*(\d+)", s, flags=re.IGNORECASE)
    if m:
        high = int(m.group(1)) - 1
        # Real fertility is negligible below 15 even though ONS groups
        # every under-20 mother together; constrain to the conventional
        # start of the fertility window.
        return (15, high)

    m = re.match(r"(\d+)\s*to\s*(\d+)", s, flags=re.IGNORECASE)
    if m:
        return (int(m.group(1)), int(m.group(2)))

    m = re.match(r"(\d+)\s*and\s*over", s, flags=re.IGNORECASE)
    if m:
        low = int(m.group(1))
        return (low, low + _OPEN_BAND_SPAN - 1)

    return None


@lru_cache(maxsize=1)
def load_ons_fertility_rates(
    *, force_download: bool = False, path: str | None = None
) -> pd.DataFrame:
    """Return a long-format frame of ASFR by year / country / age band.

    Columns: ``year`` (int), ``country`` (str), ``age_low`` (int),
    ``age_high`` (int), ``rate_per_1000`` (float). Only rows for
    ``Parent == "Mother"`` are kept.
    """
    xlsx_path = Path(path) if path is not None else _local_path()
    if force_download or not xlsx_path.exists():
        _download_to(xlsx_path)

    raw = pd.read_excel(
        xlsx_path,
        sheet_name=_ASFR_SHEET,
        header=_ASFR_HEADER_ROW,
        engine="openpyxl",
    )

    # Locate the rate column regardless of minor header-label drift.
    rate_col = next(
        (c for c in raw.columns if "fertility rate" in str(c).lower()),
        None,
    )
    if rate_col is None:
        raise ValueError(
            f"Could not locate the ASFR column in {xlsx_path.name!r} "
            f"(header row {_ASFR_HEADER_ROW}). Columns: {list(raw.columns)}"
        )

    mothers = raw[raw["Parent"] == MOTHER_PARENT].copy()
    mothers["_band"] = mothers["Age group (years)"].map(_parse_age_band)
    mothers = mothers.dropna(subset=["_band"])

    low = mothers["_band"].map(lambda t: t[0]).astype(int)
    high = mothers["_band"].map(lambda t: t[1]).astype(int)
    rate = pd.to_numeric(mothers[rate_col], errors="coerce")

    out = pd.DataFrame(
        {
            "year": pd.to_numeric(mothers["Year"], errors="coerce").astype("Int64"),
            "country": mothers["Country"].astype(str),
            "age_low": low.values,
            "age_high": high.values,
            "rate_per_1000": rate.values,
        }
    )
    out = out.dropna(subset=["year", "rate_per_1000"]).reset_index(drop=True)
    out["year"] = out["year"].astype(int)
    return out


def _resolve_year(tables: pd.DataFrame, year: int | None) -> int:
    """Return the best available year for the caller's request.

    ``None`` picks the latest year in the table. An explicit year must
    exist in the table (no interpolation) — callers asking for a future
    year should fall back to the latest with an explicit check.
    """
    if year is None:
        return int(tables["year"].max())

    y = int(year)
    available = set(tables["year"].astype(int))
    if y in available:
        return y

    earlier = [a for a in available if a < y]
    if earlier:
        return max(earlier)

    raise KeyError(f"No ONS ASFR data for {y}; earliest available is {min(available)}.")


def get_fertility_rates(
    year: int | None = None,
    *,
    country: str = DEFAULT_COUNTRY,
    tables: pd.DataFrame | None = None,
) -> dict[int, float]:
    """Return ``{age: probability}`` for a single calendar year.

    Expands each 5-year age band uniformly to single-year ages and
    converts ``rate_per_1000`` to a per-woman-per-year probability.
    Ages outside any band default to absence (and therefore to zero when
    consumed by ``age_dataset``).

    Args:
        year: Calendar year. ``None`` picks the most recent year.
        country: One of the country labels in the ONS file. Defaults to
            ``"England, Wales and Elsewhere"``, which is the broadest
            registration universe available in this dataset.
        tables: Optional pre-loaded output of
            ``load_ons_fertility_rates``. Useful for tests that
            monkey-patch the data.
    """
    if tables is None:
        tables = load_ons_fertility_rates()
    picked_year = _resolve_year(tables, year)

    sub = tables[(tables["year"] == picked_year) & (tables["country"] == country)]
    if sub.empty:
        raise KeyError(
            f"No ASFR rows for country={country!r} in {picked_year}. "
            f"Available: {sorted(tables['country'].unique())}."
        )

    # If the same age range is covered by more than one overlapping band
    # (e.g. "40 and over" plus "40 to 44"), the narrower / later-entered
    # band wins via iteration order. Build the output age-by-age so the
    # narrower band always overwrites the wider one.
    sub = sub.sort_values(by="age_high", ascending=False)  # widen first
    sub_narrow = sub.sort_values(by="age_high", ascending=True)  # narrow last

    out: dict[int, float] = {}
    for _, row in sub_narrow.iterrows():
        lo, hi = int(row["age_low"]), int(row["age_high"])
        rate = float(row["rate_per_1000"]) / 1_000.0
        for age in range(lo, hi + 1):
            out[age] = rate
    return out
