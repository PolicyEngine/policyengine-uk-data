"""ONS population projections and demographic targets.

Downloads the ONS 2022-based principal population projection for the
UK to extract total population and gender × age band targets.

For regional age breakdowns (12 regions × 9 age bands), reads the
pre-existing demographics.csv which was extracted from ONS subnational
projections. The subnational projections don't have a stable machine-
readable download URL, so this is the pragmatic compromise.

Household type and tenure targets are from ONS families & households
datasets (also lacking stable machine-readable URLs).

Sources:
- UK projections: https://www.ons.gov.uk/peoplepopulationandcommunity/populationandmigration/populationprojections/datasets/z1zippedpopulationprojectionsdatafilesuk
- NRS Scotland: https://www.nrscotland.gov.uk/statistics-and-data/statistics/statistics-by-theme/population/population-estimates/mid-year-population-estimates
"""

import io
import logging
import zipfile
from functools import lru_cache
from pathlib import Path

import pandas as pd
import requests

from policyengine_uk_data.targets.schema import (
    GeographicLevel,
    Target,
    Unit,
)

logger = logging.getLogger(__name__)

_SOURCES_YAML = Path(__file__).parent.parent / "sources.yaml"
_STORAGE = Path(__file__).parents[2] / "storage"

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
    ),
}

_UK_ZIP_URL = (
    "https://www.ons.gov.uk/file?uri=/peoplepopulationandcommunity/"
    "populationandmigration/populationprojections/datasets/"
    "z1zippedpopulationprojectionsdatafilesuk/2022based/uk.zip"
)

_REF_REGION = (
    "https://www.ons.gov.uk/peoplepopulationandcommunity/"
    "populationandmigration/populationprojections/datasets/"
    "z1zippedpopulationprojectionsdatafilesuk"
)
_REF_NRS = (
    "https://www.nrscotland.gov.uk/statistics-and-data/statistics/"
    "statistics-by-theme/population/population-estimates/"
    "mid-year-population-estimates"
)

_YEARS = list(range(2022, 2030))

# Age band boundaries
_AGE_BANDS = [
    (0, 9),
    (10, 19),
    (20, 29),
    (30, 39),
    (40, 49),
    (50, 59),
    (60, 69),
    (70, 79),
    (80, 89),
]

_GENDER_BANDS = [
    (0, 14),
    (15, 29),
    (30, 44),
    (45, 59),
    (60, 74),
    (75, 90),
]


@lru_cache(maxsize=1)
def _download_uk_projection() -> pd.DataFrame:
    """Download and parse the UK principal population projection."""
    r = requests.get(
        _UK_ZIP_URL, headers=_HEADERS, allow_redirects=True, timeout=120
    )
    r.raise_for_status()
    z = zipfile.ZipFile(io.BytesIO(r.content))
    with z.open("uk/uk_ppp_machine_readable.xlsx") as f:
        df = pd.read_excel(
            io.BytesIO(f.read()),
            sheet_name="Population",
            engine="openpyxl",
        )
    return df


def _aggregate_ages(
    df: pd.DataFrame, sex: str, low: int, high: int, years: list[int]
) -> dict[int, float]:
    """Sum population for a sex and age range across years."""
    sex_filter = "Females" if sex == "female" else "Males"
    mask = (df["Sex"] == sex_filter) & (
        df["Age"].apply(lambda a: isinstance(a, int) and low <= a <= high)
    )
    subset = df[mask]
    result = {}
    for y in years:
        if y in subset.columns:
            result[y] = float(subset[y].sum())
    return result


def _parse_uk_totals(df: pd.DataFrame) -> list[Target]:
    """Extract UK total population and gender × age bands."""
    targets = []

    # UK total
    uk_pop = {}
    for y in _YEARS:
        if y in df.columns:
            uk_pop[y] = float(df[y].sum())
    if uk_pop:
        targets.append(
            Target(
                name="ons/uk_population",
                variable="age",
                source="ons",
                unit=Unit.COUNT,
                values=uk_pop,
                is_count=True,
                reference_url=_REF_REGION,
            )
        )

    # Gender × age bands
    for sex in ["female", "male"]:
        for low, high in _GENDER_BANDS:
            values = _aggregate_ages(df, sex, low, high, _YEARS)
            if values:
                targets.append(
                    Target(
                        name=f"ons/{sex}_{low}_{high}",
                        variable="age",
                        source="ons",
                        unit=Unit.COUNT,
                        values=values,
                        is_count=True,
                        reference_url=_REF_REGION,
                    )
                )

    return targets


def _parse_regional_from_csv() -> list[Target]:
    """Read regional age band targets from demographics.csv.

    This CSV was extracted from ONS subnational projections which
    lack a stable machine-readable download URL.
    """
    csv_path = _STORAGE / "demographics.csv"
    if not csv_path.exists():
        logger.warning("demographics.csv not found, skipping regional")
        return []

    demographics = pd.read_csv(csv_path)
    targets = []

    # Skip rows now handled by dedicated modules (ons_households.py,
    # ons_tenure.py) and rows handled elsewhere in this module
    _SKIP_PREFIXES = ("tenure_", "scotland_households")
    _SKIP_NAMES = {
        "couple_3_plus_children_households",
        "couple_no_children_households",
        "couple_non_dependent_children_only_households",
        "couple_under_3_children_households",
        "lone_households_over_65",
        "lone_households_under_65",
        "lone_parent_dependent_children_households",
        "lone_parent_non_dependent_children_households",
        "multi_family_households",
        "unrelated_adult_households",
    }

    for _, row in demographics.iterrows():
        name = row["name"]
        if name in _SKIP_NAMES or any(
            name.startswith(p) for p in _SKIP_PREFIXES
        ):
            continue
        values = {}
        for y in _YEARS:
            col = str(y)
            if col in row.index and pd.notna(row[col]):
                # Values in CSV are in thousands
                values[y] = float(row[col]) * 1e3
        if values:
            targets.append(
                Target(
                    name=f"ons/{name}",
                    variable="age",
                    source="ons",
                    unit=Unit.COUNT,
                    geographic_level=GeographicLevel.REGION,
                    values=values,
                    is_count=True,
                    reference_url=_REF_REGION,
                )
            )

    return targets


# Scotland-specific (from NRS/census — not in ONS projections)
_SCOTLAND_CHILDREN_UNDER_16 = {
    y: v * 1e3
    for y, v in {
        2022: 904,
        2023: 900,
        2024: 896,
        2025: 892,
        2026: 888,
        2027: 884,
        2028: 880,
    }.items()
}

_SCOTLAND_BABIES_UNDER_1 = {
    y: v * 1e3
    for y, v in {
        2022: 46,
        2023: 46,
        2024: 46,
        2025: 46,
        2026: 46,
        2027: 46,
        2028: 46,
    }.items()
}

_SCOTLAND_HOUSEHOLDS_3PLUS_CHILDREN = {
    y: v * 1e3
    for y, v in {
        2022: 82,
        2023: 82,
        2024: 82,
        2025: 82,
        2026: 82,
        2027: 82,
        2028: 82,
    }.items()
}


# Household types and tenure are now scraped from ONS in
# ons_households.py and ons_tenure.py respectively.


def get_targets() -> list[Target]:
    targets = []

    # UK total + gender × age from live download
    try:
        df = _download_uk_projection()
        targets.extend(_parse_uk_totals(df))
    except Exception as e:
        logger.error("Failed to download ONS UK projections: %s", e)

    # Regional age bands from demographics.csv
    targets.extend(_parse_regional_from_csv())

    # Scotland-specific (NRS/census — small number of static values)
    targets.append(
        Target(
            name="ons/scotland_children_under_16",
            variable="age",
            source="nrs",
            unit=Unit.COUNT,
            values=_SCOTLAND_CHILDREN_UNDER_16,
            is_count=True,
            geographic_level=GeographicLevel.COUNTRY,
            geo_code="S",
            geo_name="Scotland",
            reference_url=_REF_NRS,
        )
    )
    targets.append(
        Target(
            name="ons/scotland_babies_under_1",
            variable="age",
            source="nrs",
            unit=Unit.COUNT,
            values=_SCOTLAND_BABIES_UNDER_1,
            is_count=True,
            geographic_level=GeographicLevel.COUNTRY,
            geo_code="S",
            geo_name="Scotland",
            reference_url=(
                "https://www.nrscotland.gov.uk/publications/"
                "vital-events-reference-tables-2024/"
            ),
        )
    )
    targets.append(
        Target(
            name="ons/scotland_households_3plus_children",
            variable="is_child",
            source="scotland_census",
            unit=Unit.COUNT,
            values=_SCOTLAND_HOUSEHOLDS_3PLUS_CHILDREN,
            is_count=True,
            geographic_level=GeographicLevel.COUNTRY,
            geo_code="S",
            geo_name="Scotland",
            reference_url=(
                "https://www.scotlandscensus.gov.uk/census-results/"
                "at-a-glance/household-composition/"
            ),
        )
    )

    return targets
