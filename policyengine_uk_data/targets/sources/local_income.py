"""Local area income targets from HMRC SPI table 3.15.

Reads pre-processed SPI CSV files for constituencies and local authorities,
extracting employment and self-employment income (count + amount) per area.

National consistency adjustment (scaling local totals to match national SPI
projections) is applied by the caller, not here.

Source: HMRC self-assessment and PAYE statistics
https://www.gov.uk/government/statistics/income-and-tax-by-county-and-region-and-by-parliamentary-constituency
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
    "https://www.gov.uk/government/statistics/"
    "income-and-tax-by-county-and-region-and-by-parliamentary-constituency"
)

_INCOME_VARIABLES = ["self_employment_income", "employment_income"]


def _load_spi(path: Path) -> pd.DataFrame:
    if not path.exists():
        logger.warning("SPI CSV not found: %s", path)
        return pd.DataFrame()
    return pd.read_csv(path)


def get_constituency_income_targets() -> pd.DataFrame:
    """Income targets for 650 constituencies (2010 codes).

    Returns DataFrame with columns: code, name, and for each income
    variable: {var}_count, {var}_amount.
    """
    spi = _load_spi(_CONST_DIR / "spi_by_constituency.csv")
    if spi.empty:
        return spi
    cols = ["code", "name"]
    for v in _INCOME_VARIABLES:
        cols.extend([f"{v}_count", f"{v}_amount"])
    return spi[cols]


def get_la_income_targets() -> pd.DataFrame:
    """Income targets for 360 local authorities.

    Returns DataFrame with columns: code, name, and for each income
    variable: {var}_count, {var}_amount.
    """
    spi = _load_spi(_LA_DIR / "spi_by_la.csv")
    if spi.empty:
        return spi
    cols = ["code", "name"]
    for v in _INCOME_VARIABLES:
        cols.extend([f"{v}_count", f"{v}_amount"])
    return spi[cols]


def get_national_income_projections(year: int) -> pd.DataFrame:
    """National income projections for consistency adjustment.

    Returns the incomes_projection.csv rows for the requested year,
    filtered to the above-personal-allowance band (12570+).
    """
    path = STORAGE / "incomes_projection.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df = df[df.year == max(df.year.min(), year)]
    return df


INCOME_VARIABLES = _INCOME_VARIABLES
REFERENCE_URL = _REF
