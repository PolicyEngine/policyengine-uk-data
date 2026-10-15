"""ONS population projections and demographic targets.

Downloads the ONS 2024-based principal population projection for the
UK to extract total population and gender × age band targets.

Regional age breakdowns use demographics.csv below age 80 and a committed
ONS projection extract for 80+, including the terminal age group. See
tools/ons_age_targets for the sources and extraction method.

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

import pandas as pd
import requests

from policyengine_uk_data.targets.schema import (
    GeographicLevel,
    Target,
    Unit,
)
from policyengine_uk_data.targets.sources._common import HEADERS, STORAGE

logger = logging.getLogger(__name__)

_UK_ZIP_URL = (
    "https://www.ons.gov.uk/file?uri=/peoplepopulationandcommunity/"
    "populationandmigration/populationprojections/datasets/"
    "z1zippedpopulationprojectionsdatafilesuk/2024based/uk.zip"
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

# FRS ages are top-coded at 80, so the last band includes every older age.
_AGE_BANDS = [
    (0, 9),
    (10, 19),
    (20, 29),
    (30, 39),
    (40, 49),
    (50, 59),
    (60, 69),
    (70, 79),
    (80, None),
]

# Five-year bands. The previous 15-year bands (0-14, 15-29, ...) pinned only
# the band totals, leaving the age composition *within* a band unconstrained
# nationally: the calibration could satisfy 15-29 while starving 18-24, which
# is what collapsed the modelled young-adult population to ~3.4M against ONS's
# ~5.4M. Regional targets already use finer bands, but they do not constrain
# the national weights. The ONS projection is published by single year of age,
# so this is a change of aggregation only - no new source data.
_GENDER_BANDS = [
    (0, 4),
    (5, 9),
    (10, 14),
    (15, 19),
    (20, 24),
    (25, 29),
    (30, 34),
    (35, 39),
    (40, 44),
    (45, 49),
    (50, 54),
    (55, 59),
    (60, 64),
    (65, 69),
    (70, 74),
    (75, 79),
    (80, None),
]


@lru_cache(maxsize=1)
def _download_uk_projection() -> pd.DataFrame:
    """Download and parse the UK principal population projection."""
    r = requests.get(_UK_ZIP_URL, headers=HEADERS, allow_redirects=True, timeout=120)
    r.raise_for_status()
    z = zipfile.ZipFile(io.BytesIO(r.content))
    projection_member = _find_projection_member(z.namelist())
    with z.open(projection_member) as f:
        df = pd.read_excel(
            io.BytesIO(f.read()),
            sheet_name="Population",
            engine="openpyxl",
        )
    return df


def _find_projection_member(names: list[str]) -> str:
    """Find the UK principal projection workbook inside the ONS zip."""
    for name in names:
        if name.endswith("uk_ppp_machine_readable.xlsx"):
            return name
    raise RuntimeError(
        "ONS UK projection zip did not contain uk_ppp_machine_readable.xlsx"
    )


def _aggregate_ages(
    df: pd.DataFrame, sex: str, low: int, high: int | None, years: list[int]
) -> dict[int, float]:
    """Sum an inclusive age band; None includes the projection's terminal age."""
    sex_filter = "Females" if sex == "female" else "Males"
    # Use each group's lower age: the 2024-based workbook ends with
    # "105 - 109" and "110 and over"; older vintages can use "90+".
    ages = pd.to_numeric(
        df["Age"].astype(str).str.extract(r"^(\d+)", expand=False), errors="coerce"
    )
    age_mask = ages >= low
    if high is not None:
        age_mask &= ages <= high
    mask = (df["Sex"] == sex_filter) & age_mask
    subset = df[mask]
    result = {}
    for y in years:
        column = _year_column(subset, y)
        if column is not None:
            result[y] = float(subset[column].sum())
    return result


def _year_column(df: pd.DataFrame, year: int) -> int | str | None:
    """Return the workbook column for a year across ONS vintages."""
    if year in df.columns:
        return year
    string_year = str(year)
    if string_year in df.columns:
        return string_year
    return None


def _parse_uk_totals(df: pd.DataFrame) -> list[Target]:
    """Extract UK total population and gender × age bands."""
    targets = []

    # UK total
    uk_pop = {}
    for y in _YEARS:
        column = _year_column(df, y)
        if column is not None:
            uk_pop[y] = float(df[column].sum())
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
                        name=f"ons/{sex}_{low}_{'plus' if high is None else high}",
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
    """Read regional bands, replacing the legacy bounded top band with 80+."""
    csv_path = STORAGE / "demographics.csv"
    if not csv_path.exists():
        logger.warning("demographics.csv not found, skipping regional")
        return []

    demographics = pd.read_csv(csv_path)
    # Replace the legacy bounded 80-89 targets with direct ONS 80+ sums,
    # including 90+, instead of relabelling totals with a different scope.
    demographics = demographics[~demographics["name"].str.endswith("_age_80_89")]
    top_bands = pd.read_csv(STORAGE / "regional_age_80_plus.csv")
    top_bands["unit"] = "person"
    demographics = pd.concat([demographics, top_bands], ignore_index=True)
    targets = []

    # Skip rows now handled by dedicated modules (ons_households.py,
    # ons_tenure.py) and rows handled elsewhere in this module
    # female_/male_ rows in demographics.csv are the superseded 15-year gender
    # bands. The live ONS projection now supplies five-year bands under the
    # same naming scheme, so emitting the CSV rows as well would double-count
    # the population and constrain overlapping ranges against each other.
    _SKIP_PREFIXES = ("tenure_", "scotland_households", "female_", "male_")
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
        if name in _SKIP_NAMES or any(name.startswith(p) for p in _SKIP_PREFIXES):
            continue
        values = {}
        for y in _YEARS:
            col = str(y)
            if col in row.index and pd.notna(row[col]):
                scale = 1 if row["unit"] == "person" else 1e3
                values[y] = float(row[col]) * scale
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
                    reference_url=(
                        row["reference_url"]
                        if pd.notna(row.get("reference_url"))
                        else _REF_REGION
                    ),
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
