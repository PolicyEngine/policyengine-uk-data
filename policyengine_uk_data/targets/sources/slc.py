"""Student Loans Company (SLC) calibration targets.

Borrower counts for England only: Plan 2 and Plan 5.

Two target types are exposed:
- `above_threshold`: borrowers liable to repay and earning above threshold
- `liable`: all borrowers liable to repay, including below-threshold holders
- `maintenance_loan`: full-time undergraduate England maintenance-loan
  recipient counts and total amount paid

Source: Explore Education Statistics — Student loan forecasts for England,
Table 6a: Forecast number of student borrowers liable to repay and number
earning above repayment threshold, by product. We use the "Higher education
total" row which sums HE full-time, HE part-time, and Advanced Learner loans.
Academic year 20XX-YY maps to calendar year 20XX+1 (e.g., 2024-25 → 2025).

Maintenance-loan targets come from Student support for higher education in
England 2025, Table 3A: Maintenance Loans paid to full-time undergraduate
students. Academic year 20XX/YY maps to calendar year 20XX+1.

Data permalink:
https://explore-education-statistics.service.gov.uk/data-tables/permalink/6ff75517-7124-487c-cb4e-08de6eccf22d
"""

import json
import os
import re
from functools import lru_cache

import pandas as pd
import requests

from policyengine_uk_data.targets.schema import Target, Unit

_PERMALINK_ID = "6ff75517-7124-487c-cb4e-08de6eccf22d"
_PERMALINK_URL = (
    f"https://explore-education-statistics.service.gov.uk"
    f"/data-tables/permalink/{_PERMALINK_ID}"
)
_MAINTENANCE_LOAN_URL = (
    "https://assets.publishing.service.gov.uk/media/"
    "691d9e662c6b98ecdbc5003f/slcsp052025.xlsx"
)
_TESTING_DATA = {
    "plan_2": {
        "above_threshold": {
            2025: 3_985_000,
            2026: 4_460_000,
            2027: 4_825_000,
            2028: 5_045_000,
            2029: 5_160_000,
            2030: 5_205_000,
        },
        "liable": {
            2025: 8_940_000,
            2026: 9_710_000,
            2027: 10_360_000,
            2028: 10_615_000,
            2029: 10_600_000,
            2030: 10_525_000,
        },
    },
    "plan_5": {
        "above_threshold": {
            2025: 0,
            2026: 35_000,
            2027: 145_000,
            2028: 390_000,
            2029: 770_000,
            2030: 1_235_000,
        },
        "liable": {
            2025: 10_000,
            2026: 230_000,
            2027: 630_000,
            2028: 1_380_000,
            2029: 2_360_000,
            2030: 3_400_000,
        },
    },
}
_MAINTENANCE_LOAN_TESTING_DATA = {
    "recipients": {
        2014: 972_830,
        2015: 963_084,
        2016: 986_323,
        2017: 1_013_354,
        2018: 1_028_438,
        2019: 1_044_973,
        2020: 1_055_702,
        2021: 1_117_591,
        2022: 1_145_289,
        2023: 1_151_607,
        2024: 1_154_427,
        2025: 1_159_761,
    },
    "amount_paid": {
        2014: 3_783_626_551,
        2015: 3_784_628_482,
        2016: 3_996_708_360,
        2017: 4_870_158_274,
        2018: 5_746_431_691,
        2019: 6_555_506_426,
        2020: 7_113_141_652,
        2021: 7_914_340_039,
        2022: 8_332_837_845,
        2023: 8_594_103_415,
        2024: 8_881_701_387,
        2025: 8_591_659_718,
    },
}


def get_snapshot_data() -> dict:
    """Return the checked-in SLC snapshot used for tests and deterministic builds."""
    return {
        plan: {
            target_type: values.copy() for target_type, values in target_data.items()
        }
        for plan, target_data in _TESTING_DATA.items()
    }


def get_maintenance_loan_snapshot_data() -> dict:
    """Return the checked-in maintenance-loan snapshot."""
    return {
        key: values.copy() for key, values in _MAINTENANCE_LOAN_TESTING_DATA.items()
    }


@lru_cache(maxsize=1)
def _fetch_slc_data() -> dict:
    """Fetch and parse SLC Table 6a data from Explore Education Statistics.

    Returns:
        Nested dict of plan -> target type -> year -> count.
    """
    if os.environ.get("TESTING", "0") == "1":
        return get_snapshot_data()

    response = requests.get(_PERMALINK_URL, timeout=30)
    response.raise_for_status()

    # Extract JSON data from __NEXT_DATA__ script tag
    match = re.search(
        r'<script id="__NEXT_DATA__" type="application/json">(.+?)</script>',
        response.text,
    )
    if not match:
        raise ValueError("Could not find __NEXT_DATA__ in SLC permalink page")

    next_data = json.loads(match.group(1))
    table_json = next_data["props"]["pageProps"]["data"]["table"]["json"]

    # Parse header row to get years - columns go newest to oldest
    # Structure: Plan 2 (6 years), Plan 5 (6 years), Plan 3 (5 years)
    header_row = table_json["thead"][1]

    plan_2_years = []
    for i in range(6):
        year_text = header_row[i]["text"]  # e.g., "2029-30"
        start_year = int(year_text.split("-")[0])
        plan_2_years.append(start_year + 1)  # 2029-30 → 2030

    plan_5_years = []
    for i in range(6, 12):
        year_text = header_row[i]["text"]
        start_year = int(year_text.split("-")[0])
        plan_5_years.append(start_year + 1)

    tbody = table_json["tbody"]
    liable_row = None
    above_threshold_row = None
    for index, row in enumerate(tbody):
        header_text = row[0].get("text", "")
        if header_text == "Higher education total":
            liable_row = row
            if index + 1 < len(tbody):
                next_row = tbody[index + 1]
                next_header = next_row[0].get("text", "")
                if "earning above repayment threshold" in next_header:
                    above_threshold_row = next_row
            break

    if liable_row is None:
        raise ValueError("Could not find 'Higher education total' row")
    if above_threshold_row is None:
        raise ValueError("Could not find 'earning above threshold' row")

    def parse_values(row, start_index, years):
        data = {}
        for offset, year in enumerate(years):
            cell_idx = start_index + offset
            if cell_idx >= len(row):
                continue
            value_text = row[cell_idx].get("text", "")
            if value_text and value_text != "no data":
                data[year] = int(value_text.replace(",", ""))
        return data

    return {
        "plan_2": {
            "above_threshold": parse_values(
                above_threshold_row, start_index=1, years=plan_2_years
            ),
            "liable": parse_values(liable_row, start_index=2, years=plan_2_years),
        },
        "plan_5": {
            "above_threshold": parse_values(
                above_threshold_row, start_index=7, years=plan_5_years
            ),
            "liable": parse_values(liable_row, start_index=8, years=plan_5_years),
        },
    }


def _row_contains_text(df: pd.DataFrame, row_index: int, text: str) -> bool:
    row = df.iloc[row_index].dropna()
    return any(str(value).strip() == text for value in row)


def _find_row(df: pd.DataFrame, text: str, start: int = 0) -> int:
    for row_index in range(start, len(df)):
        if _row_contains_text(df, row_index, text):
            return row_index
    raise ValueError(f"Could not find row containing {text!r}")


@lru_cache(maxsize=1)
def _fetch_maintenance_loan_data() -> dict:
    """Fetch full-time England maintenance-loan recipient counts and spend."""
    if os.environ.get("TESTING", "0") == "1":
        return get_maintenance_loan_snapshot_data()

    df = pd.read_excel(_MAINTENANCE_LOAN_URL, sheet_name="Table 3A", header=None)

    count_header_row = _find_row(df, "Number of students paid (000s) [27]")
    count_year_row = count_header_row + 1
    count_total_row = _find_row(df, "Grand total", start=count_year_row + 1)

    amount_header_row = _find_row(df, "Amount paid (£m)")
    amount_year_row = amount_header_row + 1
    amount_total_row = _find_row(df, "Grand total", start=amount_year_row + 1)

    year_columns = {}
    for column, value in df.iloc[count_year_row].items():
        if isinstance(value, str) and re.fullmatch(r"\d{4}/\d{2}", value):
            year_columns[column] = int(value[:4]) + 1

    if not year_columns:
        raise ValueError("Could not find maintenance-loan year columns")

    recipients = {}
    amount_paid = {}
    for column, year in year_columns.items():
        count_value = df.iloc[count_total_row, column]
        amount_value = df.iloc[amount_total_row, column]
        if pd.notna(count_value):
            recipients[year] = int(round(float(count_value) * 1_000))
        if pd.notna(amount_value):
            amount_paid[year] = int(round(float(amount_value) * 1_000_000))

    return {
        "recipients": recipients,
        "amount_paid": amount_paid,
    }


def get_targets() -> list[Target]:
    """Generate SLC calibration targets by fetching live data."""
    slc_data = _fetch_slc_data()
    maintenance_loan_data = _fetch_maintenance_loan_data()

    targets = []

    for plan, plan_label in (("plan_2", "2"), ("plan_5", "5")):
        for target_type, suffix in (
            ("above_threshold", "above_threshold"),
            ("liable", "liable"),
        ):
            targets.append(
                Target(
                    name=f"slc/plan_{plan_label}_borrowers_{suffix}",
                    variable="student_loan_plan",
                    source="slc",
                    unit=Unit.COUNT,
                    is_count=True,
                    values=slc_data[plan][target_type],
                    reference_url=_PERMALINK_URL,
                )
            )

    targets.extend(
        [
            Target(
                name="slc/maintenance_loan_recipients",
                variable="maintenance_loan",
                source="slc",
                unit=Unit.COUNT,
                is_count=True,
                values=maintenance_loan_data["recipients"],
                reference_url=_MAINTENANCE_LOAN_URL,
            ),
            Target(
                name="slc/maintenance_loan_spend",
                variable="maintenance_loan",
                source="slc",
                unit=Unit.GBP,
                values=maintenance_loan_data["amount_paid"],
                reference_url=_MAINTENANCE_LOAN_URL,
            ),
        ]
    )

    return targets
