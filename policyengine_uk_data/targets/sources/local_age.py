"""Local area age band targets from ONS subnational population estimates.

Reads pre-processed age CSV files for constituencies and local authorities,
aggregates single-year ages into 10-year bands, and applies boundary
change mapping (2010→2024) for constituencies.

Source: ONS mid-year population estimates
https://www.ons.gov.uk/peoplepopulationandcommunity/populationandmigration/populationestimates
"""

import logging
from pathlib import Path

import pandas as pd

from policyengine_uk_data.targets.sources._common import STORAGE

logger = logging.getLogger(__name__)

_CONST_DIR = (
    STORAGE.parent / "datasets" / "local_areas" / "constituencies" / "targets"
)
_LA_DIR = (
    STORAGE.parent
    / "datasets"
    / "local_areas"
    / "local_authorities"
    / "targets"
)

_REF = (
    "https://www.ons.gov.uk/peoplepopulationandcommunity/"
    "populationandmigration/populationestimates"
)

_AGE_BANDS = list(range(0, 80, 10))  # [0, 10, 20, ..., 70]


def _load_age_csv(path: Path) -> pd.DataFrame:
    """Load age.csv, returning code + single-year columns."""
    if not path.exists():
        logger.warning("Age CSV not found: %s", path)
        return pd.DataFrame()
    return pd.read_csv(path)


def _aggregate_to_bands(ages: pd.DataFrame) -> pd.DataFrame:
    """Sum single-year ages into 10-year bands.

    Returns DataFrame with columns: code, name, age/0_10, age/10_20, etc.
    """
    result = ages[["code", "name"]].copy()
    for lower in _AGE_BANDS:
        upper = lower + 10
        cols = [str(a) for a in range(lower, upper) if str(a) in ages.columns]
        result[f"age/{lower}_{upper}"] = ages[cols].sum(axis=1)
    return result


def get_constituency_age_targets() -> pd.DataFrame:
    """Age targets for 650 constituencies (2010 boundary codes).

    Returns DataFrame with 650 rows × (code, name, age/0_10, ..., age/70_80).
    Caller must apply mapping_matrix to transform to 2024 boundaries.
    """
    ages = _load_age_csv(_CONST_DIR / "age.csv")
    if ages.empty:
        return ages
    return _aggregate_to_bands(ages)


def get_la_age_targets() -> pd.DataFrame:
    """Age targets for 360 local authorities.

    Returns DataFrame with 360 rows × (code, name, age/0_10, ..., age/70_80).
    """
    ages = _load_age_csv(_LA_DIR / "age.csv")
    if ages.empty:
        return ages
    return _aggregate_to_bands(ages)


def get_uk_total_population(year: int) -> float:
    """UK total population from demographics.csv (in persons, not thousands)."""
    csv_path = STORAGE / "demographics.csv"
    if not csv_path.exists():
        return 69.9e6  # fallback
    demographics = pd.read_csv(csv_path)
    row = demographics[demographics.name == "uk_population"]
    col = str(year)
    if col in row.columns and not row[col].isna().all():
        return float(row[col].values[0]) * 1e6
    return 69.9e6


REFERENCE_URL = _REF
