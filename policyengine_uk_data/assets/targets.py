"""Dagster assets for the calibration targets database."""

from datetime import date
from pathlib import Path

import pandas as pd
from dagster import asset, AssetExecutionContext, AssetIn

from policyengine_uk_data.storage import STORAGE_FOLDER
from policyengine_uk_data.targets import TargetsDB, Area, Metric, Observation
from policyengine_uk_data.targets.seed import (
    SOURCES,
    UNIT_CONVERSIONS,
    UK_REGIONS,
    get_source_info,
)

TARGETS_DB_PATH = Path(__file__).parent.parent / "targets" / "targets.db"


@asset(group_name="targets")
def targets_areas(context: AssetExecutionContext) -> list[dict]:
    """Geographic area hierarchy: UK → countries → regions."""
    areas = [
        {
            "code": "UK",
            "name": "United Kingdom",
            "area_type": "uk",
            "parent_code": None,
        },
        {"code": "ENG", "name": "England", "area_type": "country", "parent_code": "UK"},
        {
            "code": "SCT",
            "name": "Scotland",
            "area_type": "country",
            "parent_code": "UK",
        },
        {"code": "WLS", "name": "Wales", "area_type": "country", "parent_code": "UK"},
        {
            "code": "NIR",
            "name": "Northern Ireland",
            "area_type": "country",
            "parent_code": "UK",
        },
    ]

    for code, name, parent in UK_REGIONS:
        areas.append(
            {"code": code, "name": name, "area_type": "region", "parent_code": parent}
        )

    context.log.info(f"Defined {len(areas)} areas")
    return areas


@asset(group_name="targets")
def targets_metrics(context: AssetExecutionContext) -> list[dict]:
    """Metric definitions: what statistics we track."""
    metrics = [
        # OBR fiscal
        {
            "code": "income_tax",
            "name": "Income tax revenue",
            "category": "obr",
            "unit": "gbp",
        },
        {
            "code": "national_insurance",
            "name": "National insurance contributions",
            "category": "obr",
            "unit": "gbp",
        },
        {"code": "vat", "name": "VAT revenue", "category": "obr", "unit": "gbp"},
        {
            "code": "corporation_tax",
            "name": "Corporation tax revenue",
            "category": "obr",
            "unit": "gbp",
        },
        {
            "code": "council_tax",
            "name": "Council tax revenue",
            "category": "obr",
            "unit": "gbp",
        },
        {
            "code": "fuel_duty",
            "name": "Fuel duty revenue",
            "category": "obr",
            "unit": "gbp",
        },
        {
            "code": "capital_gains_tax",
            "name": "Capital gains tax revenue",
            "category": "obr",
            "unit": "gbp",
        },
        {
            "code": "inheritance_tax",
            "name": "Inheritance tax revenue",
            "category": "obr",
            "unit": "gbp",
        },
        {
            "code": "stamp_duty_land_tax",
            "name": "Stamp duty land tax revenue",
            "category": "obr",
            "unit": "gbp",
        },
        {
            "code": "tobacco_duty",
            "name": "Tobacco duty revenue",
            "category": "obr",
            "unit": "gbp",
        },
        {
            "code": "alcohol_duty_spirits",
            "name": "Spirits duty revenue",
            "category": "obr",
            "unit": "gbp",
        },
        {
            "code": "alcohol_duty_wine",
            "name": "Wine duty revenue",
            "category": "obr",
            "unit": "gbp",
        },
        {
            "code": "alcohol_duty_beer_cider",
            "name": "Beer and cider duty revenue",
            "category": "obr",
            "unit": "gbp",
        },
        {
            "code": "air_passenger_duty",
            "name": "Air passenger duty revenue",
            "category": "obr",
            "unit": "gbp",
        },
        {
            "code": "insurance_premium_tax",
            "name": "Insurance premium tax revenue",
            "category": "obr",
            "unit": "gbp",
        },
        {
            "code": "climate_change_levy",
            "name": "Climate change levy revenue",
            "category": "obr",
            "unit": "gbp",
        },
        {
            "code": "betting_gaming_duty",
            "name": "Betting and gaming duty revenue",
            "category": "obr",
            "unit": "gbp",
        },
        {
            "code": "customs_duties",
            "name": "Customs duties revenue",
            "category": "obr",
            "unit": "gbp",
        },
        {
            "code": "bank_levy",
            "name": "Bank levy revenue",
            "category": "obr",
            "unit": "gbp",
        },
        {
            "code": "apprenticeship_levy",
            "name": "Apprenticeship levy revenue",
            "category": "obr",
            "unit": "gbp",
        },
        {
            "code": "energy_profits_levy",
            "name": "Energy profits levy revenue",
            "category": "obr",
            "unit": "gbp",
        },
        # Benefits (from OBR EFO)
        {
            "code": "child_benefit",
            "name": "Child benefit expenditure",
            "category": "obr",
            "unit": "gbp",
        },
        {
            "code": "state_pension",
            "name": "State pension expenditure",
            "category": "obr",
            "unit": "gbp",
        },
        {
            "code": "universal_credit",
            "name": "Universal credit expenditure",
            "category": "obr",
            "unit": "gbp",
        },
        {
            "code": "housing_benefit",
            "name": "Housing benefit expenditure",
            "category": "obr",
            "unit": "gbp",
        },
        {
            "code": "pension_credit",
            "name": "Pension credit expenditure",
            "category": "obr",
            "unit": "gbp",
        },
        {
            "code": "pip_dla",
            "name": "PIP and DLA expenditure",
            "category": "obr",
            "unit": "gbp",
        },
        {
            "code": "attendance_allowance",
            "name": "Attendance allowance expenditure",
            "category": "obr",
            "unit": "gbp",
        },
        {
            "code": "carers_allowance",
            "name": "Carer's allowance expenditure",
            "category": "obr",
            "unit": "gbp",
        },
        {
            "code": "jobseekers_allowance",
            "name": "Jobseeker's allowance expenditure",
            "category": "obr",
            "unit": "gbp",
        },
        {
            "code": "tax_credits",
            "name": "Tax credits expenditure",
            "category": "obr",
            "unit": "gbp",
        },
        {
            "code": "incapacity_benefits",
            "name": "Incapacity benefits expenditure",
            "category": "obr",
            "unit": "gbp",
        },
        {
            "code": "statutory_maternity_pay",
            "name": "Statutory maternity pay",
            "category": "obr",
            "unit": "gbp",
        },
        {
            "code": "winter_fuel_allowance",
            "name": "Winter fuel allowance expenditure",
            "category": "obr",
            "unit": "gbp",
        },
        {
            "code": "total_welfare_spending",
            "name": "Total welfare spending",
            "category": "obr",
            "unit": "gbp",
        },
        # Demographics
        {
            "code": "population",
            "name": "Total population",
            "category": "ons",
            "unit": "count",
        },
        {
            "code": "households",
            "name": "Number of households",
            "category": "ons",
            "unit": "count",
        },
        # Vehicles
        {
            "code": "no_vehicle_rate",
            "name": "Share of households with no vehicle",
            "category": "nts",
            "unit": "rate",
        },
        {
            "code": "one_vehicle_rate",
            "name": "Share of households with one vehicle",
            "category": "nts",
            "unit": "rate",
        },
        {
            "code": "two_plus_vehicle_rate",
            "name": "Share of households with 2+ vehicles",
            "category": "nts",
            "unit": "rate",
        },
        # Council tax bands
        {
            "code": "ct_band_a",
            "name": "Council tax band A dwellings",
            "category": "voa",
            "unit": "count",
        },
        {
            "code": "ct_band_b",
            "name": "Council tax band B dwellings",
            "category": "voa",
            "unit": "count",
        },
        {
            "code": "ct_band_c",
            "name": "Council tax band C dwellings",
            "category": "voa",
            "unit": "count",
        },
        {
            "code": "ct_band_d",
            "name": "Council tax band D dwellings",
            "category": "voa",
            "unit": "count",
        },
        {
            "code": "ct_band_e",
            "name": "Council tax band E dwellings",
            "category": "voa",
            "unit": "count",
        },
        {
            "code": "ct_band_f",
            "name": "Council tax band F dwellings",
            "category": "voa",
            "unit": "count",
        },
        {
            "code": "ct_band_g",
            "name": "Council tax band G dwellings",
            "category": "voa",
            "unit": "count",
        },
        {
            "code": "ct_band_h",
            "name": "Council tax band H dwellings",
            "category": "voa",
            "unit": "count",
        },
        # Scottish
        {
            "code": "scottish_child_payment",
            "name": "Scottish child payment expenditure",
            "category": "sss",
            "unit": "gbp",
        },
        # HMRC
        {
            "code": "salary_sacrifice_contributions",
            "name": "Total salary sacrifice contributions",
            "category": "hmrc",
            "unit": "gbp",
        },
        {
            "code": "salary_sacrifice_it_relief_basic",
            "name": "IT relief from salary sacrifice (basic rate)",
            "category": "hmrc",
            "unit": "gbp",
        },
        {
            "code": "salary_sacrifice_it_relief_higher",
            "name": "IT relief from salary sacrifice (higher rate)",
            "category": "hmrc",
            "unit": "gbp",
        },
        {
            "code": "salary_sacrifice_it_relief_additional",
            "name": "IT relief from salary sacrifice (additional rate)",
            "category": "hmrc",
            "unit": "gbp",
        },
        # DWP caseloads
        {
            "code": "uc_two_child_limit_children",
            "name": "Children affected by two-child limit",
            "category": "dwp",
            "unit": "count",
        },
        {
            "code": "uc_two_child_limit_households",
            "name": "Households affected by two-child limit",
            "category": "dwp",
            "unit": "count",
        },
        {
            "code": "pip_dl_standard_claimants",
            "name": "PIP daily living standard rate claimants",
            "category": "dwp",
            "unit": "count",
        },
        {
            "code": "pip_dl_enhanced_claimants",
            "name": "PIP daily living enhanced rate claimants",
            "category": "dwp",
            "unit": "count",
        },
        {
            "code": "benefit_capped_households",
            "name": "Households affected by benefit cap",
            "category": "dwp",
            "unit": "count",
        },
        {
            "code": "benefit_cap_total_reduction",
            "name": "Total annual benefit cap reduction",
            "category": "dwp",
            "unit": "gbp",
        },
        # Housing
        {
            "code": "rent_private",
            "name": "Total private rent payments",
            "category": "housing",
            "unit": "gbp",
        },
        {
            "code": "total_mortgage",
            "name": "Total mortgage payments",
            "category": "housing",
            "unit": "gbp",
        },
        # Savings
        {
            "code": "savings_interest_income",
            "name": "Household interest income",
            "category": "ons",
            "unit": "gbp",
        },
        # Scotland demographics
        {
            "code": "scotland_children_under_16",
            "name": "Children under 16 in Scotland",
            "category": "ons",
            "unit": "count",
        },
        {
            "code": "scotland_babies_under_1",
            "name": "Babies under 1 in Scotland",
            "category": "ons",
            "unit": "count",
        },
        {
            "code": "scotland_households_3plus_children",
            "name": "Scotland households with 3+ children",
            "category": "ons",
            "unit": "count",
        },
        {
            "code": "scotland_uc_households_child_under_1",
            "name": "UC households in Scotland with child under 1",
            "category": "dwp",
            "unit": "count",
        },
    ]

    context.log.info(f"Defined {len(metrics)} metrics")
    return metrics


@asset(group_name="targets")
def dwp_benefit_observations(context: AssetExecutionContext) -> list[dict]:
    """Load DWP benefit expenditure and caseload data from CSV.

    Source: DWP benefit expenditure and caseload tables.
    TODO: Replace with direct download from DWP.
    """
    observations = []

    tax_benefit = pd.read_csv(STORAGE_FOLDER / "tax_benefit.csv")
    years = [c for c in tax_benefit.columns if c.isdigit()]

    # Filter to DWP rows only
    dwp_rows = tax_benefit[
        tax_benefit["reference"].str.contains("dwp", case=False, na=False)
    ]

    for _, row in dwp_rows.iterrows():
        name = row["name"]
        unit = row["unit"]
        source_ref = row["reference"]
        source_info = get_source_info(source_ref)
        multiplier = UNIT_CONVERSIONS.get(unit, 1)

        for year in years:
            val = row[year]
            if pd.isna(val):
                continue
            observations.append(
                {
                    "metric_code": name,
                    "area_code": "UK",
                    "valid_year": int(year),
                    "snapshot_date": source_info["snapshot"].isoformat(),
                    "value": float(val) * multiplier,
                    "source": source_ref,
                    "source_url": source_info["url"],
                    "is_forecast": int(year) > 2023,
                }
            )

    context.log.info(f"Loaded {len(observations)} DWP benefit observations")
    return observations


@asset(group_name="targets")
def ons_demographics_observations(context: AssetExecutionContext) -> list[dict]:
    """Load ONS population projections from CSV.

    Source: ONS population projections by age, sex, and region.
    TODO: Replace with direct download from ONS.
    """
    observations = []

    demographics = pd.read_csv(STORAGE_FOLDER / "demographics.csv")
    years = [c for c in demographics.columns if c.isdigit()]

    for _, row in demographics.iterrows():
        name = row["name"]
        unit = row["unit"]
        source_ref = row["reference"]
        source_info = get_source_info(source_ref)
        multiplier = UNIT_CONVERSIONS.get(unit, 1)

        for year in years:
            val = row[year]
            if pd.isna(val):
                continue
            observations.append(
                {
                    "metric_code": name,
                    "area_code": "UK",
                    "valid_year": int(year),
                    "snapshot_date": source_info["snapshot"].isoformat(),
                    "value": float(val) * multiplier,
                    "source": source_ref,
                    "source_url": source_info["url"],
                    "is_forecast": int(year) > 2023,
                }
            )

    context.log.info(f"Loaded {len(observations)} ONS demographics observations")
    return observations


@asset(group_name="targets")
def observations_from_official_stats(context: AssetExecutionContext) -> list[dict]:
    """Hardcoded observations from official statistics."""
    observations = []

    # NTS vehicle ownership
    nts_info = SOURCES["nts_2024"]
    for year in range(2018, 2030):
        for metric, value in [
            ("no_vehicle_rate", 0.22),
            ("one_vehicle_rate", 0.44),
            ("two_plus_vehicle_rate", 0.34),
        ]:
            observations.append(
                {
                    "metric_code": metric,
                    "area_code": "UK",
                    "valid_year": year,
                    "snapshot_date": nts_info["snapshot"].isoformat(),
                    "value": value,
                    "source": "nts_2024",
                    "source_url": nts_info["url"],
                    "is_forecast": False,
                }
            )

    # ONS savings interest income
    ons_info = SOURCES["ons_national_accounts"]
    savings = {
        2020: 16.0e9,
        2021: 19.6e9,
        2022: 43.3e9,
        2023: 86.0e9,
        2024: 98.2e9,
        2025: 98.2e9,
        2026: 98.2e9,
        2027: 98.2e9,
        2028: 98.2e9,
        2029: 98.2e9,
    }
    for year, value in savings.items():
        observations.append(
            {
                "metric_code": "savings_interest_income",
                "area_code": "UK",
                "valid_year": year,
                "snapshot_date": ons_info["snapshot"].isoformat(),
                "value": value,
                "source": "ons_national_accounts",
                "source_url": ons_info["url"],
                "is_forecast": year > 2024,
            }
        )

    # HMRC salary sacrifice
    hmrc_info = SOURCES["hmrc_table_6_2"]
    ss_relief = {"basic": 1.6e9, "higher": 4.4e9, "additional": 1.2e9}
    for year in range(2024, 2030):
        uprating = 1.03 ** (year - 2024)
        for band, base in ss_relief.items():
            observations.append(
                {
                    "metric_code": f"salary_sacrifice_it_relief_{band}",
                    "area_code": "UK",
                    "valid_year": year,
                    "snapshot_date": hmrc_info["snapshot"].isoformat(),
                    "value": base * uprating,
                    "source": "hmrc_table_6_2",
                    "source_url": hmrc_info["url"],
                    "is_forecast": year > 2024,
                }
            )
        observations.append(
            {
                "metric_code": "salary_sacrifice_contributions",
                "area_code": "UK",
                "valid_year": year,
                "snapshot_date": hmrc_info["snapshot"].isoformat(),
                "value": 24e9 * uprating,
                "source": "hmrc_table_6_2",
                "source_url": hmrc_info["url"],
                "is_forecast": year > 2024,
            }
        )

    # DWP two-child limit
    dwp_tcl = SOURCES["dwp_two_child_limit"]
    for year in range(2024, 2030):
        observations.append(
            {
                "metric_code": "uc_two_child_limit_children",
                "area_code": "UK",
                "valid_year": year,
                "snapshot_date": dwp_tcl["snapshot"].isoformat(),
                "value": 1.6e6 * 1.12,
                "source": "dwp_two_child_limit",
                "source_url": dwp_tcl["url"],
                "is_forecast": year > 2024,
            }
        )
        observations.append(
            {
                "metric_code": "uc_two_child_limit_households",
                "area_code": "UK",
                "valid_year": year,
                "snapshot_date": dwp_tcl["snapshot"].isoformat(),
                "value": 440e3 * 1.12,
                "source": "dwp_two_child_limit",
                "source_url": dwp_tcl["url"],
                "is_forecast": year > 2024,
            }
        )

    # DWP PIP
    dwp_pip = SOURCES["dwp_pip_stats"]
    for year in range(2024, 2030):
        observations.append(
            {
                "metric_code": "pip_dl_standard_claimants",
                "area_code": "UK",
                "valid_year": year,
                "snapshot_date": dwp_pip["snapshot"].isoformat(),
                "value": 1_283_000,
                "source": "dwp_pip_stats",
                "source_url": dwp_pip["url"],
                "is_forecast": year > 2024,
            }
        )
        observations.append(
            {
                "metric_code": "pip_dl_enhanced_claimants",
                "area_code": "UK",
                "valid_year": year,
                "snapshot_date": dwp_pip["snapshot"].isoformat(),
                "value": 1_608_000,
                "source": "dwp_pip_stats",
                "source_url": dwp_pip["url"],
                "is_forecast": year > 2024,
            }
        )

    # Scottish child payment
    scp = SOURCES["scottish_budget"]
    scp_spend = {2024: 455.8e6, 2025: 471.0e6, 2026: 484.8e6}
    for year in range(2024, 2030):
        value = scp_spend.get(year, 471.0e6 * (1.03 ** (year - 2025)))
        observations.append(
            {
                "metric_code": "scottish_child_payment",
                "area_code": "SCT",
                "valid_year": year,
                "snapshot_date": scp["snapshot"].isoformat(),
                "value": value,
                "source": "scottish_budget",
                "source_url": scp["url"],
                "is_forecast": year > 2024,
            }
        )

    # Scotland UC babies
    dwp_sx = SOURCES["dwp_stat_xplore"]
    for year in range(2023, 2030):
        observations.append(
            {
                "metric_code": "scotland_uc_households_child_under_1",
                "area_code": "SCT",
                "valid_year": year,
                "snapshot_date": dwp_sx["snapshot"].isoformat(),
                "value": 14_000,
                "source": "dwp_stat_xplore",
                "source_url": dwp_sx["url"],
                "is_forecast": year > 2024,
            }
        )

    # Benefit cap
    dwp_bc = SOURCES["dwp_benefit_cap"]
    for year in range(2024, 2030):
        observations.append(
            {
                "metric_code": "benefit_capped_households",
                "area_code": "UK",
                "valid_year": year,
                "snapshot_date": dwp_bc["snapshot"].isoformat(),
                "value": 115_000,
                "source": "dwp_benefit_cap",
                "source_url": dwp_bc["url"],
                "is_forecast": year > 2025,
            }
        )
        observations.append(
            {
                "metric_code": "benefit_cap_total_reduction",
                "area_code": "UK",
                "valid_year": year,
                "snapshot_date": dwp_bc["snapshot"].isoformat(),
                "value": 60 * 52 * 115_000,
                "source": "dwp_benefit_cap",
                "source_url": dwp_bc["url"],
                "is_forecast": year > 2025,
            }
        )

    # Housing
    ons_rent = SOURCES["ons_private_rent"]
    for year in range(2024, 2030):
        observations.append(
            {
                "metric_code": "rent_private",
                "area_code": "UK",
                "valid_year": year,
                "snapshot_date": ons_rent["snapshot"].isoformat(),
                "value": 1_400 * 12 * 4.7e6,
                "source": "ons_private_rent",
                "source_url": ons_rent["url"],
                "is_forecast": year > 2025,
            }
        )
        observations.append(
            {
                "metric_code": "total_mortgage",
                "area_code": "UK",
                "valid_year": year,
                "snapshot_date": ons_rent["snapshot"].isoformat(),
                "value": 1_100 * 12 * 7.5e6,
                "source": "ons_private_rent",
                "source_url": ons_rent["url"],
                "is_forecast": year > 2025,
            }
        )

    # Scotland demographics
    nrs = SOURCES["nrs_scotland"]
    census = SOURCES["scotland_census"]
    for year in range(2022, 2030):
        observations.append(
            {
                "metric_code": "scotland_children_under_16",
                "area_code": "SCT",
                "valid_year": year,
                "snapshot_date": nrs["snapshot"].isoformat(),
                "value": 900_000,
                "source": "nrs_scotland",
                "source_url": nrs["url"],
                "is_forecast": year > 2024,
            }
        )
        observations.append(
            {
                "metric_code": "scotland_babies_under_1",
                "area_code": "SCT",
                "valid_year": year,
                "snapshot_date": nrs["snapshot"].isoformat(),
                "value": 46_000,
                "source": "nrs_scotland",
                "source_url": nrs["url"],
                "is_forecast": year > 2024,
            }
        )
        observations.append(
            {
                "metric_code": "scotland_households_3plus_children",
                "area_code": "SCT",
                "valid_year": year,
                "snapshot_date": census["snapshot"].isoformat(),
                "value": 60_000,
                "source": "scotland_census",
                "source_url": census["url"],
                "is_forecast": year > 2024,
            }
        )

    context.log.info(f"Loaded {len(observations)} observations from official stats")
    return observations


@asset(group_name="targets")
def observations_council_tax(context: AssetExecutionContext) -> list[dict]:
    """Council tax band observations by region."""
    observations = []
    voa = SOURCES["voa_council_tax"]

    ct_path = STORAGE_FOLDER / "council_tax_bands_2024.csv"
    if not ct_path.exists():
        context.log.warning("council_tax_bands_2024.csv not found")
        return observations

    ct_data = pd.read_csv(ct_path)
    region_mapping = {
        "North East": "NORTH_EAST",
        "North West": "NORTH_WEST",
        "Yorkshire and The Humber": "YORKSHIRE",
        "East Midlands": "EAST_MIDLANDS",
        "West Midlands": "WEST_MIDLANDS",
        "East": "EAST_OF_ENGLAND",
        "London": "LONDON",
        "South East": "SOUTH_EAST",
        "South West": "SOUTH_WEST",
        "Wales": "WLS",
    }

    for _, row in ct_data.iterrows():
        area_code = region_mapping.get(
            row["Region"], row["Region"].upper().replace(" ", "_")
        )
        for band in ["A", "B", "C", "D", "E", "F", "G", "H"]:
            for year in range(2024, 2030):
                observations.append(
                    {
                        "metric_code": f"ct_band_{band.lower()}",
                        "area_code": area_code,
                        "valid_year": year,
                        "snapshot_date": voa["snapshot"].isoformat(),
                        "value": float(row[band]),
                        "source": "voa_council_tax",
                        "source_url": voa["url"],
                        "is_forecast": year > 2024,
                    }
                )

    context.log.info(f"Loaded {len(observations)} council tax observations")
    return observations


@asset(
    group_name="targets",
    ins={
        "areas": AssetIn("targets_areas"),
        "metrics": AssetIn("targets_metrics"),
        "obr_data": AssetIn("obr_receipts_observations"),
        "dwp_benefits": AssetIn("dwp_benefit_observations"),
        "ons_demographics": AssetIn("ons_demographics_observations"),
        "other_stats": AssetIn("observations_from_official_stats"),
        "council_tax": AssetIn("observations_council_tax"),
    },
)
def targets_db(
    context: AssetExecutionContext,
    areas: list[dict],
    metrics: list[dict],
    obr_data: list[dict],
    dwp_benefits: list[dict],
    ons_demographics: list[dict],
    other_stats: list[dict],
    council_tax: list[dict],
) -> dict:
    """Assemble the calibration targets SQLite database."""
    db = TargetsDB(TARGETS_DB_PATH)
    db.clear()

    # Load areas
    for a in areas:
        db.add_area(Area(**a))

    # Load metrics
    for m in metrics:
        db.add_metric(Metric(**m))

    # Load all observations
    all_obs = []
    all_sources = obr_data + dwp_benefits + ons_demographics + other_stats + council_tax
    for obs_dict in all_sources:
        obs_dict["snapshot_date"] = date.fromisoformat(obs_dict["snapshot_date"])
        all_obs.append(Observation(**obs_dict))

    db.bulk_add_observations(all_obs)

    stats = db.stats()
    context.log.info(
        f"Database built: {stats['observations']} observations, "
        f"{stats['metrics']} metrics, {stats['areas']} areas"
    )

    return {
        "path": str(TARGETS_DB_PATH),
        "observations": stats["observations"],
        "metrics": stats["metrics"],
        "areas": stats["areas"],
    }
