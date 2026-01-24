"""Seed the targets database with areas, metrics, and observations.

Populates from:
1. tax_benefit.csv - OBR/DWP tax and benefit aggregates
2. demographics.csv - ONS population statistics
3. Hardcoded targets from loss.py
4. Council tax bands by region
"""

from datetime import date
from pathlib import Path

import pandas as pd
from rich.console import Console

from policyengine_uk_data.storage import STORAGE_FOLDER
from policyengine_uk_data.targets.database import TargetsDB
from policyengine_uk_data.targets.models import Area, Metric, Observation

console = Console()

# Source metadata with snapshot dates
SOURCES = {
    "obr_march_2024_efo": {
        "url": "https://obr.uk/efo/economic-and-fiscal-outlook-march-2024/",
        "snapshot": date(2024, 3, 6),
        "is_forecast": True,
    },
    "obr_october_2024_efo": {
        "url": "https://obr.uk/efo/economic-and-fiscal-outlook-october-2024/",
        "snapshot": date(2024, 10, 30),
        "is_forecast": True,
    },
    "dwp_benefit_tables": {
        "url": "https://www.gov.uk/government/collections/benefit-expenditure-and-caseload-tables",
        "snapshot": date(2024, 3, 1),
        "is_forecast": True,
    },
    "ons_age_sex_region": {
        "url": "https://www.ons.gov.uk/peoplepopulationandcommunity/populationandmigration/populationprojections",
        "snapshot": date(2024, 1, 15),
        "is_forecast": False,
    },
    "hmrc_spi": {
        "url": "https://www.gov.uk/government/statistics/personal-incomes-statistics",
        "snapshot": date(2024, 4, 1),
        "is_forecast": False,
    },
    "nts_2024": {
        "url": "https://www.gov.uk/government/statistics/national-travel-survey-2024",
        "snapshot": date(2024, 8, 1),
        "is_forecast": False,
    },
    "voa_council_tax": {
        "url": "https://www.gov.uk/government/statistics/council-tax-stock-of-properties-2024",
        "snapshot": date(2024, 9, 1),
        "is_forecast": False,
    },
    "dwp_stat_xplore": {
        "url": "https://stat-xplore.dwp.gov.uk/",
        "snapshot": date(2024, 6, 1),
        "is_forecast": False,
    },
    "scottish_budget": {
        "url": "https://www.gov.scot/publications/scottish-budget-2026-2027/",
        "snapshot": date(2024, 12, 4),
        "is_forecast": True,
    },
    "ons_national_accounts": {
        "url": "https://www.ons.gov.uk/economy/grossdomesticproductgdp/timeseries/haxv/ukea",
        "snapshot": date(2024, 9, 30),
        "is_forecast": False,
    },
    "hmrc_table_6_2": {
        "url": "https://assets.publishing.service.gov.uk/media/687a294e312ee8a5f0806b6d/Tables_6_1_and_6_2.csv",
        "snapshot": date(2024, 7, 1),
        "is_forecast": False,
    },
    "dwp_two_child_limit": {
        "url": "https://www.gov.uk/government/statistics/universal-credit-and-child-tax-credit-claimants-statistics-related-to-the-policy-to-provide-support-for-a-maximum-of-2-children-april-2024",
        "snapshot": date(2024, 4, 1),
        "is_forecast": False,
    },
    "dwp_pip_stats": {
        "url": "https://www.disabilityrightsuk.org/news/90-pip-standard-daily-living-component-recipients-would-fail-new-green-paper-test",
        "snapshot": date(2024, 5, 1),
        "is_forecast": False,
    },
    "dwp_benefit_cap": {
        "url": "https://www.gov.uk/government/statistics/benefit-cap-number-of-households-capped-to-february-2025",
        "snapshot": date(2025, 2, 1),
        "is_forecast": False,
    },
    "ons_private_rent": {
        "url": "https://www.ons.gov.uk/economy/inflationandpriceindices/bulletins/privaterentandhousepricesuk/january2025",
        "snapshot": date(2025, 1, 15),
        "is_forecast": False,
    },
    "nrs_scotland": {
        "url": "https://www.nrscotland.gov.uk/statistics-and-data/statistics/statistics-by-theme/population/population-estimates",
        "snapshot": date(2024, 6, 1),
        "is_forecast": False,
    },
    "scotland_census": {
        "url": "https://www.scotlandscensus.gov.uk/census-results/at-a-glance/household-composition/",
        "snapshot": date(2024, 3, 1),
        "is_forecast": False,
    },
}

UNIT_CONVERSIONS = {
    "gbp-bn": 1e9,
    "person-m": 1e6,
    "person-k": 1e3,
    "benefit-unit-m": 1e6,
    "household-k": 1e3,
}

UK_REGIONS = [
    ("NORTH_EAST", "North East", "ENG"),
    ("NORTH_WEST", "North West", "ENG"),
    ("YORKSHIRE", "Yorkshire and the Humber", "ENG"),
    ("EAST_MIDLANDS", "East Midlands", "ENG"),
    ("WEST_MIDLANDS", "West Midlands", "ENG"),
    ("EAST_OF_ENGLAND", "East of England", "ENG"),
    ("LONDON", "London", "ENG"),
    ("SOUTH_EAST", "South East", "ENG"),
    ("SOUTH_WEST", "South West", "ENG"),
]


def seed_areas(db: TargetsDB) -> int:
    """Seed geographic area hierarchy."""
    areas = [
        Area(code="UK", name="United Kingdom", area_type="uk", parent_code=None),
        Area(code="ENG", name="England", area_type="country", parent_code="UK"),
        Area(code="SCT", name="Scotland", area_type="country", parent_code="UK"),
        Area(code="WLS", name="Wales", area_type="country", parent_code="UK"),
        Area(code="NIR", name="Northern Ireland", area_type="country", parent_code="UK"),
    ]

    # Add English regions
    for code, name, parent in UK_REGIONS:
        areas.append(Area(code=code, name=name, area_type="region", parent_code=parent))

    for area in areas:
        db.add_area(area)

    return len(areas)


def seed_metrics(db: TargetsDB) -> int:
    """Seed metric definitions."""
    metrics = [
        # OBR fiscal aggregates
        Metric(code="income_tax", name="Income tax revenue", category="obr", unit="gbp"),
        Metric(code="national_insurance", name="National insurance contributions", category="obr", unit="gbp"),
        Metric(code="vat", name="VAT revenue", category="obr", unit="gbp"),
        Metric(code="corporation_tax", name="Corporation tax revenue", category="obr", unit="gbp"),
        Metric(code="council_tax", name="Council tax revenue", category="obr", unit="gbp"),
        Metric(code="fuel_duty", name="Fuel duty revenue", category="obr", unit="gbp"),
        Metric(code="capital_gains_tax", name="Capital gains tax revenue", category="obr", unit="gbp"),

        # Benefits
        Metric(code="child_benefit", name="Child benefit expenditure", category="dwp", unit="gbp"),
        Metric(code="state_pension", name="State pension expenditure", category="dwp", unit="gbp"),
        Metric(code="universal_credit", name="Universal credit expenditure", category="dwp", unit="gbp"),
        Metric(code="housing_benefit", name="Housing benefit expenditure", category="dwp", unit="gbp"),
        Metric(code="pension_credit", name="Pension credit expenditure", category="dwp", unit="gbp"),
        Metric(code="pip", name="Personal independence payment expenditure", category="dwp", unit="gbp"),
        Metric(code="attendance_allowance", name="Attendance allowance expenditure", category="dwp", unit="gbp"),
        Metric(code="carers_allowance", name="Carer's allowance expenditure", category="dwp", unit="gbp"),

        # Demographics
        Metric(code="population", name="Total population", category="ons", unit="count"),
        Metric(code="households", name="Number of households", category="ons", unit="count"),

        # Vehicle ownership
        Metric(code="no_vehicle_rate", name="Share of households with no vehicle", category="nts", unit="rate"),
        Metric(code="one_vehicle_rate", name="Share of households with one vehicle", category="nts", unit="rate"),
        Metric(code="two_plus_vehicle_rate", name="Share of households with 2+ vehicles", category="nts", unit="rate"),

        # Council tax bands
        Metric(code="ct_band_a", name="Council tax band A dwellings", category="voa", unit="count"),
        Metric(code="ct_band_b", name="Council tax band B dwellings", category="voa", unit="count"),
        Metric(code="ct_band_c", name="Council tax band C dwellings", category="voa", unit="count"),
        Metric(code="ct_band_d", name="Council tax band D dwellings", category="voa", unit="count"),
        Metric(code="ct_band_e", name="Council tax band E dwellings", category="voa", unit="count"),
        Metric(code="ct_band_f", name="Council tax band F dwellings", category="voa", unit="count"),
        Metric(code="ct_band_g", name="Council tax band G dwellings", category="voa", unit="count"),
        Metric(code="ct_band_h", name="Council tax band H dwellings", category="voa", unit="count"),

        # Scottish benefits
        Metric(code="scottish_child_payment", name="Scottish child payment expenditure", category="sss", unit="gbp"),

        # HMRC
        Metric(code="salary_sacrifice_contributions", name="Total salary sacrifice contributions", category="hmrc", unit="gbp"),
        Metric(code="salary_sacrifice_it_relief_basic", name="IT relief from salary sacrifice (basic rate)", category="hmrc", unit="gbp"),
        Metric(code="salary_sacrifice_it_relief_higher", name="IT relief from salary sacrifice (higher rate)", category="hmrc", unit="gbp"),
        Metric(code="salary_sacrifice_it_relief_additional", name="IT relief from salary sacrifice (additional rate)", category="hmrc", unit="gbp"),

        # DWP caseloads
        Metric(code="uc_two_child_limit_children", name="Children affected by two-child limit", category="dwp", unit="count"),
        Metric(code="uc_two_child_limit_households", name="Households affected by two-child limit", category="dwp", unit="count"),
        Metric(code="pip_dl_standard_claimants", name="PIP daily living standard rate claimants", category="dwp", unit="count"),
        Metric(code="pip_dl_enhanced_claimants", name="PIP daily living enhanced rate claimants", category="dwp", unit="count"),
        Metric(code="benefit_capped_households", name="Households affected by benefit cap", category="dwp", unit="count"),
        Metric(code="benefit_cap_total_reduction", name="Total annual benefit cap reduction", category="dwp", unit="gbp"),

        # Housing
        Metric(code="rent_private", name="Total private rent payments", category="housing", unit="gbp"),
        Metric(code="total_mortgage", name="Total mortgage payments", category="housing", unit="gbp"),

        # Savings
        Metric(code="savings_interest_income", name="Household interest income", category="ons", unit="gbp"),

        # Scotland demographics
        Metric(code="scotland_children_under_16", name="Children under 16 in Scotland", category="ons", unit="count"),
        Metric(code="scotland_babies_under_1", name="Babies under 1 in Scotland", category="ons", unit="count"),
        Metric(code="scotland_households_3plus_children", name="Scotland households with 3+ children", category="ons", unit="count"),
        Metric(code="scotland_uc_households_child_under_1", name="UC households in Scotland with child under 1", category="dwp", unit="count"),
    ]

    for metric in metrics:
        db.add_metric(metric)

    return len(metrics)


def get_source_info(source_ref: str) -> dict:
    """Get source URL, snapshot date, and is_forecast flag from reference."""
    # Strip year suffix like _2024 from reference
    base_ref = source_ref.split("_2")[0] if "_2" in source_ref else source_ref
    return SOURCES.get(base_ref, SOURCES.get(source_ref, {
        "url": "",
        "snapshot": date(2024, 3, 1),
        "is_forecast": False,
    }))


def seed_from_csv(db: TargetsDB) -> int:
    """Load observations from tax_benefit.csv and demographics.csv."""
    observations = []

    # Tax and benefit targets
    tax_benefit = pd.read_csv(STORAGE_FOLDER / "tax_benefit.csv")
    years = [c for c in tax_benefit.columns if c.isdigit()]

    for _, row in tax_benefit.iterrows():
        name = row["name"]
        unit = row["unit"]
        source_ref = row["reference"]
        source_info = get_source_info(source_ref)

        multiplier = UNIT_CONVERSIONS.get(unit, 1)

        for year in years:
            val = row[year]
            if pd.isna(val):
                continue

            observations.append(Observation(
                metric_code=name,
                area_code="UK",
                valid_year=int(year),
                snapshot_date=source_info["snapshot"],
                value=float(val) * multiplier,
                source=source_ref,
                source_url=source_info["url"],
                is_forecast=source_info["is_forecast"] and int(year) > 2023,
            ))

    # Demographics targets
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

            observations.append(Observation(
                metric_code=name,
                area_code="UK",
                valid_year=int(year),
                snapshot_date=source_info["snapshot"],
                value=float(val) * multiplier,
                source=source_ref,
                source_url=source_info["url"],
                is_forecast=source_info["is_forecast"] and int(year) > 2023,
            ))

    db.bulk_add_observations(observations)
    return len(observations)


def seed_hardcoded_targets(db: TargetsDB) -> int:
    """Add hardcoded targets with proper metadata."""
    observations = []

    # NTS vehicle ownership (constant rates)
    nts_info = SOURCES["nts_2024"]
    for year in range(2018, 2030):
        observations.extend([
            Observation(
                metric_code="no_vehicle_rate", area_code="UK", valid_year=year,
                snapshot_date=nts_info["snapshot"], value=0.22,
                source="nts_2024", source_url=nts_info["url"], is_forecast=False,
            ),
            Observation(
                metric_code="one_vehicle_rate", area_code="UK", valid_year=year,
                snapshot_date=nts_info["snapshot"], value=0.44,
                source="nts_2024", source_url=nts_info["url"], is_forecast=False,
            ),
            Observation(
                metric_code="two_plus_vehicle_rate", area_code="UK", valid_year=year,
                snapshot_date=nts_info["snapshot"], value=0.34,
                source="nts_2024", source_url=nts_info["url"], is_forecast=False,
            ),
        ])

    # ONS savings interest income
    ons_info = SOURCES["ons_national_accounts"]
    savings_income = {
        2020: 16.0e9, 2021: 19.6e9, 2022: 43.3e9, 2023: 86.0e9,
        2024: 98.2e9, 2025: 98.2e9, 2026: 98.2e9, 2027: 98.2e9, 2028: 98.2e9, 2029: 98.2e9,
    }
    for year, value in savings_income.items():
        observations.append(Observation(
            metric_code="savings_interest_income", area_code="UK", valid_year=year,
            snapshot_date=ons_info["snapshot"], value=value,
            source="ons_national_accounts", source_url=ons_info["url"],
            is_forecast=year > 2024,
        ))

    # HMRC salary sacrifice
    hmrc_info = SOURCES["hmrc_table_6_2"]
    ss_relief = {"basic": 1.6e9, "higher": 4.4e9, "additional": 1.2e9}
    for year in range(2024, 2030):
        uprating = 1.03 ** (year - 2024)
        for band, base_value in ss_relief.items():
            observations.append(Observation(
                metric_code=f"salary_sacrifice_it_relief_{band}", area_code="UK", valid_year=year,
                snapshot_date=hmrc_info["snapshot"], value=base_value * uprating,
                source="hmrc_table_6_2", source_url=hmrc_info["url"],
                is_forecast=year > 2024,
            ))
        observations.append(Observation(
            metric_code="salary_sacrifice_contributions", area_code="UK", valid_year=year,
            snapshot_date=hmrc_info["snapshot"], value=24e9 * uprating,
            source="hmrc_table_6_2", source_url=hmrc_info["url"],
            is_forecast=year > 2024,
        ))

    # DWP two-child limit
    dwp_tcl_info = SOURCES["dwp_two_child_limit"]
    uprating_24_25 = 1.12
    for year in range(2024, 2030):
        observations.extend([
            Observation(
                metric_code="uc_two_child_limit_children", area_code="UK", valid_year=year,
                snapshot_date=dwp_tcl_info["snapshot"], value=1.6e6 * uprating_24_25,
                source="dwp_two_child_limit", source_url=dwp_tcl_info["url"],
                is_forecast=year > 2024,
            ),
            Observation(
                metric_code="uc_two_child_limit_households", area_code="UK", valid_year=year,
                snapshot_date=dwp_tcl_info["snapshot"], value=440e3 * uprating_24_25,
                source="dwp_two_child_limit", source_url=dwp_tcl_info["url"],
                is_forecast=year > 2024,
            ),
        ])

    # DWP PIP claimants
    dwp_pip_info = SOURCES["dwp_pip_stats"]
    for year in range(2024, 2030):
        observations.extend([
            Observation(
                metric_code="pip_dl_standard_claimants", area_code="UK", valid_year=year,
                snapshot_date=dwp_pip_info["snapshot"], value=1_283_000,
                source="dwp_pip_stats", source_url=dwp_pip_info["url"],
                is_forecast=year > 2024,
            ),
            Observation(
                metric_code="pip_dl_enhanced_claimants", area_code="UK", valid_year=year,
                snapshot_date=dwp_pip_info["snapshot"], value=1_608_000,
                source="dwp_pip_stats", source_url=dwp_pip_info["url"],
                is_forecast=year > 2024,
            ),
        ])

    # Scottish child payment
    scp_info = SOURCES["scottish_budget"]
    scp_spend = {2024: 455.8e6, 2025: 471.0e6, 2026: 484.8e6}
    for year in range(2024, 2030):
        value = scp_spend.get(year, 471.0e6 * (1.03 ** (year - 2025)))
        observations.append(Observation(
            metric_code="scottish_child_payment", area_code="SCT", valid_year=year,
            snapshot_date=scp_info["snapshot"], value=value,
            source="scottish_budget", source_url=scp_info["url"],
            is_forecast=year > 2024,
        ))

    # DWP Scotland UC households with baby
    dwp_sx_info = SOURCES["dwp_stat_xplore"]
    for year in range(2023, 2030):
        observations.append(Observation(
            metric_code="scotland_uc_households_child_under_1", area_code="SCT", valid_year=year,
            snapshot_date=dwp_sx_info["snapshot"], value=14_000,
            source="dwp_stat_xplore", source_url=dwp_sx_info["url"],
            is_forecast=year > 2024,
        ))

    # DWP benefit cap
    dwp_bc_info = SOURCES["dwp_benefit_cap"]
    for year in range(2024, 2030):
        observations.extend([
            Observation(
                metric_code="benefit_capped_households", area_code="UK", valid_year=year,
                snapshot_date=dwp_bc_info["snapshot"], value=115_000,
                source="dwp_benefit_cap", source_url=dwp_bc_info["url"],
                is_forecast=year > 2025,
            ),
            Observation(
                metric_code="benefit_cap_total_reduction", area_code="UK", valid_year=year,
                snapshot_date=dwp_bc_info["snapshot"], value=60 * 52 * 115_000,
                source="dwp_benefit_cap", source_url=dwp_bc_info["url"],
                is_forecast=year > 2025,
            ),
        ])

    # Housing targets
    ons_rent_info = SOURCES["ons_private_rent"]
    for year in range(2024, 2030):
        observations.extend([
            Observation(
                metric_code="rent_private", area_code="UK", valid_year=year,
                snapshot_date=ons_rent_info["snapshot"], value=1_400 * 12 * 4.7e6,
                source="ons_private_rent", source_url=ons_rent_info["url"],
                is_forecast=year > 2025,
            ),
            Observation(
                metric_code="total_mortgage", area_code="UK", valid_year=year,
                snapshot_date=ons_rent_info["snapshot"], value=1_100 * 12 * 7.5e6,
                source="ons_private_rent", source_url=ons_rent_info["url"],
                is_forecast=year > 2025,
            ),
        ])

    # Scotland demographics
    nrs_info = SOURCES["nrs_scotland"]
    census_info = SOURCES["scotland_census"]
    for year in range(2022, 2030):
        observations.extend([
            Observation(
                metric_code="scotland_children_under_16", area_code="SCT", valid_year=year,
                snapshot_date=nrs_info["snapshot"], value=900_000,
                source="nrs_scotland", source_url=nrs_info["url"],
                is_forecast=year > 2024,
            ),
            Observation(
                metric_code="scotland_babies_under_1", area_code="SCT", valid_year=year,
                snapshot_date=nrs_info["snapshot"], value=46_000,
                source="nrs_scotland", source_url=nrs_info["url"],
                is_forecast=year > 2024,
            ),
            Observation(
                metric_code="scotland_households_3plus_children", area_code="SCT", valid_year=year,
                snapshot_date=census_info["snapshot"], value=60_000,
                source="scotland_census", source_url=census_info["url"],
                is_forecast=year > 2024,
            ),
        ])

    db.bulk_add_observations(observations)
    return len(observations)


def seed_council_tax_targets(db: TargetsDB) -> int:
    """Load council tax band targets by region."""
    observations = []
    voa_info = SOURCES["voa_council_tax"]

    ct_path = STORAGE_FOLDER / "council_tax_bands_2024.csv"
    if not ct_path.exists():
        return 0

    ct_data = pd.read_csv(ct_path)

    # Add regions as areas if not already present
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
        region_name = row["Region"]
        area_code = region_mapping.get(region_name, region_name.upper().replace(" ", "_"))

        for band in ["A", "B", "C", "D", "E", "F", "G", "H"]:
            metric_code = f"ct_band_{band.lower()}"
            for year in range(2024, 2030):
                observations.append(Observation(
                    metric_code=metric_code,
                    area_code=area_code,
                    valid_year=year,
                    snapshot_date=voa_info["snapshot"],
                    value=float(row[band]),
                    source="voa_council_tax",
                    source_url=voa_info["url"],
                    is_forecast=year > 2024,
                ))

    db.bulk_add_observations(observations)
    return len(observations)


def seed_all(db_path: Path | None = None) -> None:
    """Seed the entire targets database."""
    db = TargetsDB(db_path)
    db.clear()

    console.print("[bold]Seeding calibration targets database...[/bold]")

    area_count = seed_areas(db)
    console.print(f"  {area_count} areas")

    metric_count = seed_metrics(db)
    console.print(f"  {metric_count} metrics")

    csv_count = seed_from_csv(db)
    console.print(f"  {csv_count} observations from CSV files")

    hardcoded_count = seed_hardcoded_targets(db)
    console.print(f"  {hardcoded_count} hardcoded observations")

    ct_count = seed_council_tax_targets(db)
    console.print(f"  {ct_count} council tax observations")

    stats = db.stats()
    console.print(f"\n[green]Database seeded:[/green]")
    console.print(f"  {stats['observations']} observations")
    console.print(f"  {stats['metrics']} metrics")
    console.print(f"  {stats['areas']} areas")
    console.print(f"  Categories: {', '.join(stats.get('categories', []))}")


if __name__ == "__main__":
    seed_all()
