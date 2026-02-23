"""Student Loans Company (SLC) calibration targets.

Borrower counts for England only: Plan 2 and Plan 5.

Two types of targets are provided:
1. "above threshold" - borrowers liable to repay AND earning above threshold
   (matches FRS coverage via PAYE deductions)
2. "liable" - total borrowers liable to repay (includes below-threshold)

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
        Dict with nested structure:
        {
            'plan_2': {'above_threshold': {...}, 'liable': {...}},
            'plan_5': {'above_threshold': {...}, 'liable': {...}}
        }
        Each inner dict maps calendar year (int) to borrower count (int).
    """
    response = requests.get(_PERMALINK_URL, timeout=30)
    response.raise_for_status()

    match = re.search(
        r'<script id="__NEXT_DATA__" type="application/json">(.+?)</script>',
        response.text,
    )
    if not match:
        raise ValueError("Could not find __NEXT_DATA__ in SLC permalink page")

    next_data = json.loads(match.group(1))
    table_json = next_data["props"]["pageProps"]["data"]["table"]["json"]

    # Parse header row to get years (columns go newest to oldest)
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

    # Find "Higher education total" rows
    # Row 10: [0]="Higher education total", [1]="Number of borrowers liable...",
    #         [2-7]=Plan 2 data, [8-13]=Plan 5 data
    # Row 11: [0]="Number of borrowers...earning above...",
    #         [1-6]=Plan 2 data, [7-12]=Plan 5 data
    liable_row = None
    above_threshold_row = None

    for i, row in enumerate(tbody):
        header_text = row[0].get("text", "")
        if header_text == "Higher education total":
            # This row contains liable-to-repay data
            liable_row = row
            # Next row should be above-threshold data
            if i + 1 < len(tbody):
                next_row = tbody[i + 1]
                next_header = next_row[0].get("text", "")
                if "earning above" in next_header:
                    above_threshold_row = next_row
            break

    if above_threshold_row is None:
        raise ValueError("Could not find 'earning above threshold' row")
    if liable_row is None:
        raise ValueError("Could not find 'Higher education total' row")

    def parse_values(row, start_idx, years):
        """Parse numeric values from row starting at start_idx."""
        data = {}
        for i, year in enumerate(years):
            cell_idx = start_idx + i
            if cell_idx >= len(row):
                continue
            value_text = row[cell_idx].get("text", "")
            if value_text and value_text not in ("no data", "0"):
                data[year] = int(value_text.replace(",", ""))
        return data

    # Liable row: data starts at index 2 (after header and subheader)
    p2_liable = parse_values(liable_row, 2, plan_2_years)
    p5_liable = parse_values(liable_row, 8, plan_5_years)

    # Above threshold row: data starts at index 1 (after header only)
    p2_above = parse_values(above_threshold_row, 1, plan_2_years)
    p5_above = parse_values(above_threshold_row, 7, plan_5_years)

    return {
        "plan_2": {"above_threshold": p2_above, "liable": p2_liable},
        "plan_5": {"above_threshold": p5_above, "liable": p5_liable},
    }


def get_targets() -> list[Target]:
    """Generate SLC calibration targets by fetching live data."""
    slc_data = _fetch_slc_data()

    targets = []

    # Above-threshold targets (borrowers with PAYE deductions)
    targets.append(
        Target(
            name="slc/plan_2_borrowers_above_threshold",
            variable="student_loan_plan",
            source="slc",
            unit=Unit.COUNT,
            is_count=True,
            values=slc_data["plan_2"]["above_threshold"],
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
            values=slc_data["plan_5"]["above_threshold"],
            reference_url=_PERMALINK_URL,
        )
    )

    # Liable-to-repay targets (all borrowers including below-threshold)
    targets.append(
        Target(
            name="slc/plan_2_borrowers_liable",
            variable="student_loan_plan",
            source="slc",
            unit=Unit.COUNT,
            is_count=True,
            values=slc_data["plan_2"]["liable"],
            reference_url=_PERMALINK_URL,
        )
    )
    targets.append(
        Target(
            name="slc/plan_5_borrowers_liable",
            variable="student_loan_plan",
            source="slc",
            unit=Unit.COUNT,
            is_count=True,
            values=slc_data["plan_5"]["liable"],
            reference_url=_PERMALINK_URL,
        )
    )

    return targets
