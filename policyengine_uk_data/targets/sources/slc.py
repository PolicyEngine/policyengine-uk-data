"""Student Loans Company (SLC) calibration targets.

Borrower counts for England only: Plan 2 and Plan 5, restricted to
borrowers liable to repay and earning above the repayment threshold.
This matches the FRS coverage (PAYE deductions only).

Source: Explore Education Statistics — Student loan forecasts for England,
Table 6a: Forecast number of student borrowers liable to repay and number
earning above repayment threshold, by product. We use the "Higher education
total" row which sums HE full-time, HE part-time, and Advanced Learner loans.
Academic year 20XX-YY maps to calendar year 20XX+1 (e.g., 2024-25 → 2025).

Data permalink:
https://explore-education-statistics.service.gov.uk/data-tables/permalink/6ff75517-7124-487c-cb4e-08de6eccf22d
"""

import json
import re
import requests
from functools import lru_cache

from policyengine_uk_data.targets.schema import Target, Unit

_PERMALINK_ID = "6ff75517-7124-487c-cb4e-08de6eccf22d"
_PERMALINK_URL = (
    f"https://explore-education-statistics.service.gov.uk"
    f"/data-tables/permalink/{_PERMALINK_ID}"
)


@lru_cache(maxsize=1)
def _fetch_slc_data() -> dict:
    """Fetch and parse SLC Table 6a data from Explore Education Statistics.

    Returns:
        Dict with keys 'plan_2' and 'plan_5', each containing a dict
        mapping calendar year (int) to borrower count above threshold (int).
    """
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

    # Get Plan 2 years (first 6 columns)
    plan_2_years = []
    for i in range(6):
        year_text = header_row[i]["text"]  # e.g., "2029-30"
        start_year = int(year_text.split("-")[0])
        calendar_year = start_year + 1  # 2029-30 → 2030
        plan_2_years.append(calendar_year)

    # Get Plan 5 years (next 6 columns)
    plan_5_years = []
    for i in range(6, 12):
        year_text = header_row[i]["text"]
        start_year = int(year_text.split("-")[0])
        calendar_year = start_year + 1
        plan_5_years.append(calendar_year)

    # Find the "Higher education total" / "earning above threshold" row
    # This is the row following "Higher education total" with "liable to repay"
    tbody = table_json["tbody"]

    # Row 11 contains: header + 6 Plan 2 values + 6 Plan 5 values + 5 Plan 3
    target_row = None
    for row in tbody:
        header_text = row[0].get("text", "")
        if "earning above repayment threshold" in header_text:
            # Check if previous context was "Higher education total"
            # Actually, row 11 is after HE total row 10, and starts with
            # the "earning above" header (no group header due to rowSpan)
            target_row = row
            break

    if target_row is None:
        raise ValueError("Could not find 'earning above threshold' row")

    # Parse Plan 2 data (cells 1-6, mapping to plan_2_years)
    plan_2_data = {}
    for i, year in enumerate(plan_2_years):
        cell_idx = 1 + i  # Skip header cell
        value_text = target_row[cell_idx].get("text", "")
        if value_text and value_text not in ("no data", "0"):
            value = int(value_text.replace(",", ""))
            plan_2_data[year] = value

    # Parse Plan 5 data (cells 7-12, mapping to plan_5_years)
    plan_5_data = {}
    for i, year in enumerate(plan_5_years):
        cell_idx = 7 + i  # Skip header + Plan 2 cells
        value_text = target_row[cell_idx].get("text", "")
        if value_text and value_text not in ("no data", "0"):
            value = int(value_text.replace(",", ""))
            plan_5_data[year] = value

    return {"plan_2": plan_2_data, "plan_5": plan_5_data}


def get_targets() -> list[Target]:
    """Generate SLC calibration targets by fetching live data."""
    slc_data = _fetch_slc_data()

    targets = []

    targets.append(
        Target(
            name="slc/plan_2_borrowers_above_threshold",
            variable="student_loan_plan",
            source="slc",
            unit=Unit.COUNT,
            is_count=True,
            values=slc_data["plan_2"],
            reference_url=_PERMALINK_URL,
        )
    )

    targets.append(
        Target(
            name="slc/plan_5_borrowers_above_threshold",
            variable="student_loan_plan",
            source="slc",
            unit=Unit.COUNT,
            is_count=True,
            values=slc_data["plan_5"],
            reference_url=_PERMALINK_URL,
        )
    )

    return targets
