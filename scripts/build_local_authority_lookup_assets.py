"""Build versioned LAD22 and LAD23 local-authority lookup assets.

This script intentionally leaves local_authorities_2021.csv unchanged.
It derives both versioned files from that established coordinate grid, then
applies the April 2023 English local-government reorganisation.
"""

from __future__ import annotations

import csv
from pathlib import Path


STORAGE_DIR = Path(__file__).parents[1] / "policyengine_uk_data" / "storage"
SOURCE_PATH = STORAGE_DIR / "local_authorities_2021.csv"

LAD22_AUTHORITIES = {
    "E07000026": (4, 18, "Allerdale"),
    "E07000027": (2, 17, "Barrow-in-Furness"),
    "E07000028": (4, 19, "Carlisle"),
    "E07000029": (3, 18, "Copeland"),
    "E07000030": (5, 18, "Eden"),
    "E07000031": (4, 17, "South Lakeland"),
    "E07000163": (6, 17, "Craven"),
    "E07000164": (10, 18, "Hambleton"),
    "E07000165": (8, 17, "Harrogate"),
    "E07000166": (7, 17, "Richmondshire"),
    "E07000167": (10, 16, "Ryedale"),
    "E07000168": (10, 17, "Scarborough"),
    "E07000169": (9, 16, "Selby"),
    "E07000187": (0, 1, "Mendip"),
    "E07000188": (-1, 2, "Sedgemoor"),
    "E07000189": (-1, 1, "South Somerset"),
    "E07000246": (-2, 1, "Somerset West and Taunton"),
}

LAD23_AUTHORITIES = {
    "E06000063": (4, 18, "Cumberland"),
    "E06000064": (4, 17, "Westmorland and Furness"),
    "E06000065": (8, 17, "North Yorkshire"),
    "E06000066": (0, 1, "Somerset"),
}

NORTHERN_IRELAND_MISSING_AUTHORITY = {
    "N09000011": (-3, 16, "Ards and North Down"),
}


def load_source() -> dict[str, tuple[int, int, str]]:
    with SOURCE_PATH.open(newline="") as source:
        return {
            row["code"]: (int(float(row["x"])), int(float(row["y"])), row["name"])
            for row in csv.DictReader(source)
        }


def write_asset(
    filename: str,
    authorities: dict[str, tuple[int, int, str]],
) -> None:
    path = STORAGE_DIR / filename
    with path.open("w", newline="") as destination:
        writer = csv.writer(destination)
        writer.writerow(("code", "x", "y", "name"))
        for code, (x, y, name) in sorted(authorities.items()):
            writer.writerow((code, x, y, name))


def main() -> None:
    source = load_source()

    lad22 = {
        code: value for code, value in source.items() if code not in LAD23_AUTHORITIES
    }
    lad22.update(LAD22_AUTHORITIES)
    lad22.update(NORTHERN_IRELAND_MISSING_AUTHORITY)

    lad23 = dict(source)
    lad23.update(LAD23_AUTHORITIES)
    lad23.update(NORTHERN_IRELAND_MISSING_AUTHORITY)

    write_asset("local_authorities_lad22.csv", lad22)
    write_asset("local_authorities_lad23.csv", lad23)


if __name__ == "__main__":
    main()
