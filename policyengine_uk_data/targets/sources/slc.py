"""Student Loans Company (SLC) calibration targets.

Borrower counts for England only: Plan 2 and Plan 5.

Two target types are exposed:
- `above_threshold`: borrowers liable to repay and earning above threshold
- `liable`: all borrowers liable to repay, including below-threshold holders

Source: Explore Education Statistics — Student loan forecasts for England,
Table 6a: Forecast number of student borrowers liable to repay and number
earning above repayment threshold, by product. We use the "Higher education
total" row which sums HE full-time, HE part-time, and Advanced Learner loans.
Academic year 20XX-YY maps to calendar year 20XX+1 (e.g., 2024-25 → 2025).

Data permalink:
https://explore-education-statistics.service.gov.uk/data-tables/permalink/6ff75517-7124-487c-cb4e-08de6eccf22d
"""

import json
import os
import re
from functools import lru_cache

import requests

from policyengine_uk_data.targets.schema import Target, Unit

_PERMALINK_ID = "6ff75517-7124-487c-cb4e-08de6eccf22d"
_PERMALINK_URL = (
    f"https://explore-education-statistics.service.gov.uk"
    f"/data-tables/permalink/{_PERMALINK_ID}"
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


def get_snapshot_data() -> dict:
    """Return the checked-in SLC snapshot used for tests and deterministic builds."""
    return {
        plan: {
            target_type: values.copy() for target_type, values in target_data.items()
        }
        for plan, target_data in _TESTING_DATA.items()
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


def get_targets() -> list[Target]:
    """Generate SLC calibration targets by fetching live data."""
    slc_data = _fetch_slc_data()

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

    return targets
