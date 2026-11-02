"""Reproduce the public ONS regional 80+ target extract (no survey data)."""

import argparse
import hashlib
import io
import zipfile
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
YEARS = [str(year) for year in range(2022, 2030)]
ONS = "https://www.ons.gov.uk/file?uri=/peoplepopulationandcommunity/populationandmigration/populationprojections/datasets/"
SOURCES = {
    "regions": (
        "localauthoritiesinenglandz1/2022basedfiveyearmigrationvariantprojection/2022snpppopulationsyoa5yr.zip",
        "171cd05fc753277388352c76abf8cc8b305ca537cd593a6e4467af7e8a8e4b5b",
    ),
    "scotland": (
        "z5zippedpopulationprojectionsdatafilesscotland/2022based/sc.zip",
        "6e0dc9250a9890680754ab501be56973c722a57950e4535c364fbc4494372b3d",
    ),
    "wales": (
        "z4zippedpopulationprojectionsdatafileswales/2022based/wa.zip",
        "32a7c2aacf9b3c945abeaa19a6d6d7aa62133f491bfca5a9700a393244a7b051",
    ),
    "ni": (
        "z6zippedpopulationprojectionsdatafilesnorthernireland/2022based/ni.zip",
        "69b006c2741943d97d153cd024e4b21ea94b52ef766efcde6bd737a400d95489",
    ),
}
REGIONS = {
    "E12000001": "north_east",
    "E12000002": "north_west",
    "E12000003": "yorkshire_and_the_humber",
    "E12000004": "east_midlands",
    "E12000005": "west_midlands",
    "E12000006": "east",
    "E12000007": "london",
    "E12000008": "south_east",
    "E12000009": "south_west",
}


def aggregate_english_regions(
    population: pd.DataFrame, crosswalk: pd.DataFrame
) -> pd.DataFrame:
    """Sum ages 80 through the terminal 90+ row in every English LA."""
    lookup = crosswalk[["la_code", "region_code"]].drop_duplicates()
    assert lookup.la_code.is_unique, "Local authorities have ambiguous regions"
    ages = population.AGE_GROUP.astype(str).str.extract(r"^(\d+)")[0]
    older = population.loc[pd.to_numeric(ages, errors="coerce") >= 80].copy()
    assert set(older.AGE_GROUP) == {str(age) for age in range(80, 90)} | {"90 and over"}
    older = older.merge(
        lookup,
        left_on="AREA_CODE",
        right_on="la_code",
        how="left",
        validate="many_to_one",
    )
    assert older.region_code.notna().all(), "Some local authorities lack a region"
    assert set(older.region_code) == set(REGIONS)
    assert older.AREA_CODE.nunique() == 309
    assert older.groupby("AREA_CODE").AGE_GROUP.nunique().eq(11).all()
    result = older.groupby("region_code")[YEARS].sum().round(3)
    assert result.sum().round(3).equals(older[YEARS].sum().round(3))
    result.index = result.index.map(REGIONS)
    return result


def aggregate_country(population: pd.DataFrame) -> pd.Series:
    """Include all single years and terminal age groups from 80 upwards."""
    ages = population.Age.astype(str).str.extract(r"^(\d+)")[0]
    older = population.loc[
        (pd.to_numeric(ages, errors="coerce") >= 80)
        & population.Sex.isin(["Females", "Males"])
    ]
    assert {"105 - 109", "110 and over"}.issubset(set(older.Age.astype(str)))
    older = older.rename(columns=lambda column: str(column))
    return older[YEARS].sum()


def build(source_dir: Path, output: Path) -> None:
    """Download pinned public sources and save their exact published sums."""
    source_dir.mkdir(parents=True, exist_ok=True)
    for name, (suffix, expected_sha256) in SOURCES.items():
        path = source_dir / f"ons-{name}-2022.zip"
        if not path.exists():
            response = requests.get(
                ONS + suffix,
                headers={
                    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
                },
                timeout=120,
            )
            response.raise_for_status()
            path.write_bytes(response.content)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected_sha256

    with zipfile.ZipFile(source_dir / "ons-regions-2022.zip") as archive:
        population = pd.read_csv(archive.open("2022 SNPP Population persons.csv"))
    crosswalk = pd.read_csv(ROOT / "policyengine_uk_data/storage/oa_crosswalk.csv.gz")
    totals = aggregate_english_regions(population, crosswalk)
    for country, prefix in [("scotland", "sc"), ("wales", "wa"), ("ni", "ni")]:
        with zipfile.ZipFile(source_dir / f"ons-{country}-2022.zip") as archive:
            population = pd.read_excel(
                io.BytesIO(
                    archive.read(f"{prefix}/{prefix}_ppp_machine_readable.xlsx")
                ),
                sheet_name="Population",
                engine="openpyxl",
            )
        totals.loc["northern_ireland" if country == "ni" else country] = (
            aggregate_country(population)
        )
    totals.index = [f"{region}_age_80_plus" for region in totals.index]
    totals.index.name = "name"
    reference_base = "https://www.ons.gov.uk/peoplepopulationandcommunity/populationandmigration/populationprojections/datasets/"
    totals["reference_url"] = reference_base + "localauthoritiesinenglandz1"
    for country in ["scotland", "wales", "ni"]:
        region = "northern_ireland" if country == "ni" else country
        totals.loc[f"{region}_age_80_plus", "reference_url"] = (
            reference_base + SOURCES[country][0].split("/")[0]
        )
    totals.sort_index().to_csv(output, float_format="%.3f")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=ROOT / ".job")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "policyengine_uk_data/storage/regional_age_80_plus.csv",
    )
    args = parser.parse_args()
    build(args.source_dir, args.output)
