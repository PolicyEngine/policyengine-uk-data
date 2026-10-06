"""Rebuild the English per-band council tax columns in ``la_council_tax.csv``.

Downloads MHCLG *Council tax levels set by local authorities in England
2026-27* (``Tables_1-9_2026-27.ods``) and reads sheet ``Table_9``, "Area
council tax for a dwelling occupied by two adults by band". That table
publishes, for every English billing authority, the band A-H area council
tax inclusive of all precepts.

The script is additive and England-only:

- It fills ``band_A_amount``..``band_H_amount`` for English rows and
  refreshes ``band_d_amount`` for those rows from Table 9 Band D.
- It leaves the new columns **blank for Wales and Scotland**. Table 9
  covers England only; the Welsh Government and Scottish Government
  publications are separate sources and Wales additionally has a Band I.
  Northern Ireland has no council tax at all.
- It never touches the VOA band-count columns, ``total_dwellings``,
  ``has_council_tax`` or ``total_council_tax_net``.

Every other cell is round-tripped as the literal text already in the CSV,
so re-running the script on an up-to-date file produces no diff.

Post-2023 South Yorkshire ONS codes used by MHCLG are re-mapped to the
pre-2023 codes used by ``local_authorities_2021.csv``.

Usage::

    python policyengine_uk_data/storage/build_la_council_tax.py

Source:
- https://www.gov.uk/government/statistics/council-tax-levels-set-by-local-authorities-in-england-2026-to-2027
"""

from __future__ import annotations

import csv
from io import BytesIO
from pathlib import Path
import re
import xml.etree.ElementTree as ET
import zipfile

import requests

STORAGE = Path(__file__).parent
CSV_PATH = STORAGE / "la_council_tax.csv"

MHCLG_RELEASE_URL = (
    "https://www.gov.uk/government/statistics/"
    "council-tax-levels-set-by-local-authorities-in-england-2026-to-2027"
)
MHCLG_TABLES_URL = (
    "https://assets.publishing.service.gov.uk/media/"
    "69de1fa63e81003ae0422508/Tables_1-9_2026-27.ods"
)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
    ),
}

SHEET = "Table_9"
BANDS = ["A", "B", "C", "D", "E", "F", "G", "H"]
BAND_AMOUNT_COLUMNS = [f"band_{band}_amount" for band in BANDS]

# Statutory England and Wales band ratios relative to Band D.
BAND_RATIOS = {
    "A": 6 / 9,
    "B": 7 / 9,
    "C": 8 / 9,
    "D": 1.0,
    "E": 11 / 9,
    "F": 13 / 9,
    "G": 15 / 9,
    "H": 18 / 9,
}

# MHCLG uses the post-2023 South Yorkshire ONS codes; the reference LA
# list (``local_authorities_2021.csv``) still uses the pre-2023 codes.
ONS_CODE_REMAP = {
    "E08000038": "E08000016",  # Barnsley
    "E08000039": "E08000019",  # Sheffield
}

_ODS_NS = {
    "table": "urn:oasis:names:tc:opendocument:xmlns:table:1.0",
    "text": "urn:oasis:names:tc:opendocument:xmlns:text:1.0",
}
_ONS_CODE = re.compile(r"^E\d{8}$")
# Repeat counts this large are the sheet's trailing filler, not real cells.
_REPEAT_LIMIT = 200


def _download_tables() -> bytes:
    response = requests.get(
        MHCLG_TABLES_URL, headers=HEADERS, allow_redirects=True, timeout=120
    )
    response.raise_for_status()
    return response.content


def _sheet_rows(ods_bytes: bytes, sheet_name: str) -> list[list[str]]:
    """Return the cell text of one ODS sheet, expanding repeat counts."""
    with zipfile.ZipFile(BytesIO(ods_bytes)) as archive:
        root = ET.fromstring(archive.read("content.xml"))

    table_ns = _ODS_NS["table"]
    for table in root.iter(f"{{{table_ns}}}table"):
        if table.get(f"{{{table_ns}}}name") != sheet_name:
            continue
        rows: list[list[str]] = []
        for row in table.findall(f"{{{table_ns}}}table-row"):
            row_repeat = int(row.get(f"{{{table_ns}}}number-rows-repeated", 1))
            cells: list[str] = []
            for cell in row.findall(f"{{{table_ns}}}table-cell"):
                col_repeat = int(cell.get(f"{{{table_ns}}}number-columns-repeated", 1))
                text = "".join(
                    "".join(p.itertext())
                    for p in cell.findall(f"{{{_ODS_NS['text']}}}p")
                )
                cells.extend([text] * min(col_repeat, _REPEAT_LIMIT))
            rows.extend([cells] * min(row_repeat, _REPEAT_LIMIT))
        return rows
    raise ValueError(f"Sheet {sheet_name!r} not found in the MHCLG workbook")


def _to_amount(cell: str) -> float:
    return float(cell.replace(",", "").replace("£", "").strip())


def parse_table_9(ods_bytes: bytes) -> dict[str, dict[str, float]]:
    """Return ``{ons_code: {band: amount}}`` for every English authority."""
    rows = _sheet_rows(ods_bytes, SHEET)
    header = next((r for r in rows if len(r) > 1 and r[0].strip() == "E Code"), None)
    if header is None:
        raise ValueError("Could not find the Table 9 header row")
    band_columns = {
        band: header.index(f"Band {band}") for band in BANDS if f"Band {band}" in header
    }
    missing = set(BANDS) - set(band_columns)
    if missing:
        raise ValueError(f"Table 9 is missing band columns: {sorted(missing)}")

    out: dict[str, dict[str, float]] = {}
    for row in rows:
        if len(row) <= max(band_columns.values()):
            continue
        code = row[1].strip()
        if not _ONS_CODE.match(code):
            continue
        code = ONS_CODE_REMAP.get(code, code)
        out[code] = {
            band: _to_amount(row[index]) for band, index in band_columns.items()
        }
    if not out:
        raise ValueError("Parsed no English authority rows from Table 9")
    return out


def check_band_ratios(
    table: dict[str, dict[str, float]], tolerance: float = 0.01
) -> list[tuple[str, str, float]]:
    """Return ``(code, band, deviation)`` where Table 9 breaks the ratios."""
    breaches = []
    for code, bands in table.items():
        band_d = bands["D"]
        for band, ratio in BAND_RATIOS.items():
            deviation = abs(bands[band] - band_d * ratio)
            if deviation > tolerance:
                breaches.append((code, band, deviation))
    return breaches


def build(ods_bytes: bytes | None = None) -> None:
    if ods_bytes is None:
        ods_bytes = _download_tables()
    table = parse_table_9(ods_bytes)

    with open(CSV_PATH, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    for column in BAND_AMOUNT_COLUMNS:
        if column not in fieldnames:
            fieldnames.append(column)

    matched = 0
    for row in rows:
        bands = table.get(row["code"]) if row["country"] == "ENGLAND" else None
        if bands is None:
            for column in BAND_AMOUNT_COLUMNS:
                row[column] = ""
            continue
        matched += 1
        for band in BANDS:
            row[f"band_{band}_amount"] = f"{bands[band]:.2f}"
        # Keep the existing literal when it already matches Table 9, so a
        # re-run never churns unrelated formatting.
        existing = row["band_d_amount"]
        if not existing or abs(float(existing) - bands["D"]) > 5e-3:
            row["band_d_amount"] = f"{bands['D']:.2f}"

    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    breaches = check_band_ratios(table)
    print(f"Table 9 authorities parsed: {len(table)}")
    print(f"CSV England rows populated: {matched} of {len(rows)} rows")
    print(f"Band-ratio breaches above £0.01: {len(breaches)}")
    for code, band, deviation in breaches[:20]:
        print(f"  {code} band {band}: off by £{deviation:.4f}")


if __name__ == "__main__":
    build()
