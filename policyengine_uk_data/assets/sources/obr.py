"""OBR Economic and Fiscal Outlook data source.

Downloads official forecast tables from the Office for Budget Responsibility.
"""

from datetime import date
from io import BytesIO

import pandas as pd
import requests
from dagster import asset, AssetExecutionContext

# OBR EFO release metadata
OBR_RELEASES = {
    "march_2024": {
        "receipts_url": "https://obr.uk/download/march-2024-economic-and-fiscal-outlook-detailed-forecast-tables-receipts/",
        "expenditure_url": "https://obr.uk/download/march-2024-economic-and-fiscal-outlook-detailed-forecast-tables-expenditure/",
        "snapshot_date": date(2024, 3, 6),
        "source": "OBR March 2024 EFO",
        "source_url": "https://obr.uk/efo/economic-and-fiscal-outlook-march-2024/",
    },
    "november_2024": {
        "receipts_url": "https://obr.uk/download/october-2024-economic-and-fiscal-outlook-detailed-forecast-tables-receipts/",
        "expenditure_url": "https://obr.uk/download/october-2024-economic-and-fiscal-outlook-detailed-forecast-tables-expenditure/",
        "snapshot_date": date(2024, 10, 30),
        "source": "OBR October 2024 EFO",
        "source_url": "https://obr.uk/efo/economic-and-fiscal-outlook-october-2024/",
    },
    "march_2025": {
        "receipts_url": "https://obr.uk/download/march-2025-economic-and-fiscal-outlook-detailed-forecast-tables-receipts/",
        "expenditure_url": "https://obr.uk/download/march-2025-economic-and-fiscal-outlook-detailed-forecast-tables-expenditure/",
        "snapshot_date": date(2025, 3, 26),
        "source": "OBR March 2025 EFO",
        "source_url": "https://obr.uk/efo/economic-and-fiscal-outlook-march-2025/",
    },
    "november_2025": {
        "receipts_url": "https://obr.uk/download/november-2025-economic-and-fiscal-outlook-detailed-forecast-tables-receipts/",
        "expenditure_url": "https://obr.uk/download/november-2025-economic-and-fiscal-outlook-detailed-forecast-tables-expenditure/",
        "snapshot_date": date(2025, 11, 26),
        "source": "OBR November 2025 EFO",
        "source_url": "https://obr.uk/efo/economic-and-fiscal-outlook-november-2025/",
    },
}

HTTP_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
}

# Map OBR row labels to our metric codes - receipts (sheet 3.9)
RECEIPTS_MAPPING = {
    "Income tax (gross of tax credits)1": "income_tax",
    "National insurance contributions": "national_insurance",
    "Value added tax": "vat",
    "Corporation tax": "corporation_tax",
    "Fuel duties": "fuel_duty",
    "Capital gains tax": "capital_gains_tax",
    "Inheritance tax": "inheritance_tax",
    "Stamp duty land tax": "stamp_duty_land_tax",
    "Tobacco duties": "tobacco_duty",
    "Spirits duties": "alcohol_duty_spirits",
    "Wine duties": "alcohol_duty_wine",
    "Beer and cider duties": "alcohol_duty_beer_cider",
    "Air passenger duty": "air_passenger_duty",
    "Insurance premium tax": "insurance_premium_tax",
    "Climate change levy": "climate_change_levy",
    "Betting and gaming duties": "betting_gaming_duty",
    "Customs Duties": "customs_duties",
    "Bank levy": "bank_levy",
    "Apprenticeship Levy": "apprenticeship_levy",
    "Energy profits levy": "energy_profits_levy",
}

# Map OBR row labels to our metric codes - welfare spending (sheet 4.9)
WELFARE_MAPPING = {
    "State pension": "state_pension",
    "Universal credit": "universal_credit",
    "Housing benefit (not on JSA)1": "housing_benefit",
    "Disability living allowance and personal independence payments": "pip_dla",
    "Attendance allowance": "attendance_allowance",
    "Pension credit": "pension_credit",
    "Carer's allowance": "carers_allowance",
    "Jobseeker's allowance": "jobseekers_allowance",
    "Child benefit": "child_benefit",
    "Personal tax credits": "tax_credits",
    "Incapacity benefits2": "incapacity_benefits",
    "Statutory maternity pay": "statutory_maternity_pay",
    "Winter fuel payment": "winter_fuel_allowance",
    "Total welfare": "total_welfare_spending",
}

# Council tax by country (sheet 4.1)
COUNCIL_TAX_MAPPING = {
    "England council tax receipts": ("council_tax", "ENG"),
    "Scotland council tax receipts": ("council_tax", "SCT"),
    "Wales council tax receipts": ("council_tax", "WLS"),
    "Total council tax receipts": ("council_tax", "UK"),
}


def download_obr_file(url: str) -> pd.ExcelFile:
    """Download an OBR Excel file."""
    response = requests.get(url, headers=HTTP_HEADERS, timeout=60)
    response.raise_for_status()
    return pd.ExcelFile(BytesIO(response.content))


def find_year_columns(
    df: pd.DataFrame, search_year: str = "2024-25"
) -> tuple[int, dict]:
    """Find header row and year column mapping in OBR sheet."""
    header_row = None
    for i, row in df.iterrows():
        if search_year in str(row.values):
            header_row = i
            break

    if header_row is None:
        raise ValueError(f"Could not find header row with {search_year}")

    years_row = df.iloc[header_row]
    year_cols = {}
    for col_idx, val in enumerate(years_row):
        if isinstance(val, str) and "-" in val and val[:4].isdigit():
            year = int(val.split("-")[0])
            year_cols[col_idx] = year

    return header_row, year_cols


def parse_obr_sheet(
    df: pd.DataFrame,
    mapping: dict,
    release_info: dict,
    label_cols: list[int] = [1, 2],
    area_code: str = "UK",
    aggregate_duplicates: bool = False,
) -> list[dict]:
    """Generic parser for OBR sheets with year columns.

    If aggregate_duplicates=True, values with the same metric/area/year are summed.
    This handles cases like Universal credit appearing in multiple sections.
    """
    header_row, year_cols = find_year_columns(df)

    # Collect raw values, potentially with duplicates
    raw_values = {}  # (metric_code, area_code, year) -> value

    for i in range(header_row + 1, len(df)):
        row = df.iloc[i]

        # Get the label from specified columns
        label = None
        for col in label_cols:
            if (
                col < len(row)
                and pd.notna(row.iloc[col])
                and isinstance(row.iloc[col], str)
            ):
                label = row.iloc[col].strip()
                break

        if not label or label in ["of which:", "Note:"]:
            continue

        # Check if this label maps to a metric
        mapped = mapping.get(label)
        if not mapped:
            continue

        # Handle mapping that includes area code
        if isinstance(mapped, tuple):
            metric_code, obs_area_code = mapped
        else:
            metric_code = mapped
            obs_area_code = area_code

        for col_idx, year in year_cols.items():
            val = row.iloc[col_idx]
            if pd.notna(val) and isinstance(val, (int, float)):
                value = float(val) * 1e9  # £ billion to natural units
                key = (metric_code, obs_area_code, year)

                if aggregate_duplicates and key in raw_values:
                    raw_values[key] += value
                else:
                    raw_values[key] = value

    # Convert to observation list
    observations = []
    for (metric_code, obs_area_code, year), value in raw_values.items():
        observations.append(
            {
                "metric_code": metric_code,
                "area_code": obs_area_code,
                "valid_year": year,
                "snapshot_date": release_info["snapshot_date"].isoformat(),
                "value": value,
                "source": release_info["source"],
                "source_url": release_info["source_url"],
                "is_forecast": year >= 2025,
            }
        )

    return observations


def download_and_parse_receipts(release: str = "november_2025") -> list[dict]:
    """Download and parse OBR receipts data (sheet 3.9)."""
    release_info = OBR_RELEASES[release]
    xl = download_obr_file(release_info["receipts_url"])
    df = pd.read_excel(xl, sheet_name="3.9", header=None)
    return parse_obr_sheet(df, RECEIPTS_MAPPING, release_info)


def download_and_parse_welfare(release: str = "november_2025") -> list[dict]:
    """Download and parse OBR welfare spending data (sheet 4.9).

    Uses aggregate_duplicates=True because some benefits (e.g. Universal credit)
    appear in both welfare cap and outside-welfare-cap sections.
    """
    release_info = OBR_RELEASES[release]
    xl = download_obr_file(release_info["expenditure_url"])
    df = pd.read_excel(xl, sheet_name="4.9", header=None)
    return parse_obr_sheet(df, WELFARE_MAPPING, release_info, aggregate_duplicates=True)


def download_and_parse_council_tax(release: str = "november_2025") -> list[dict]:
    """Download and parse OBR council tax data (sheet 4.1)."""
    release_info = OBR_RELEASES[release]
    xl = download_obr_file(release_info["expenditure_url"])
    df = pd.read_excel(xl, sheet_name="4.1", header=None)
    return parse_obr_sheet(df, COUNCIL_TAX_MAPPING, release_info)


@asset(group_name="targets")
def obr_receipts_observations(context: AssetExecutionContext) -> list[dict]:
    """Download and parse all OBR forecasts from the Economic and Fiscal Outlook.

    Includes:
    - Tax receipts (income tax, NI, VAT, etc.) from sheet 3.9
    - Welfare spending (state pension, universal credit, etc.) from sheet 4.9
    - Council tax by country from sheet 4.1

    Processes all available EFO releases to capture forecast evolution over time.
    """
    all_observations = []

    # Process all releases
    for release_key, release_info in sorted(OBR_RELEASES.items()):
        context.log.info(f"\nProcessing {release_info['source']}...")
        release_obs = []

        try:
            # Tax receipts
            context.log.info("  Downloading receipts data...")
            receipts = download_and_parse_receipts(release_key)
            release_obs.extend(receipts)
            context.log.info(f"    Receipts: {len(receipts)} observations")

            # Welfare spending
            context.log.info("  Downloading welfare data...")
            welfare = download_and_parse_welfare(release_key)
            release_obs.extend(welfare)
            context.log.info(f"    Welfare: {len(welfare)} observations")

            # Council tax
            context.log.info("  Downloading council tax data...")
            council_tax = download_and_parse_council_tax(release_key)
            release_obs.extend(council_tax)
            context.log.info(f"    Council tax: {len(council_tax)} observations")

            all_observations.extend(release_obs)
            context.log.info(f"  {release_info['source']}: {len(release_obs)} total observations")

        except Exception as e:
            context.log.warning(f"  Failed to process {release_info['source']}: {e}")
            continue

    context.log.info(f"\nTotal across all releases: {len(all_observations)} OBR observations")

    # Summary by metric
    metrics = {}
    for obs in all_observations:
        m = obs["metric_code"]
        metrics[m] = metrics.get(m, 0) + 1

    context.log.info("\nObservations by metric:")
    for metric, count in sorted(metrics.items()):
        context.log.info(f"  {metric}: {count} observations")

    # Summary by release
    releases = {}
    for obs in all_observations:
        s = obs["source"]
        releases[s] = releases.get(s, 0) + 1

    context.log.info("\nObservations by release:")
    for release, count in sorted(releases.items()):
        context.log.info(f"  {release}: {count} observations")

    return all_observations
