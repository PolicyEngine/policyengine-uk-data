"""
Loss functions and target matrices for dataset calibration.

This module creates target matrices comparing PolicyEngine UK model outputs
against official statistics from OBR, ONS, HMRC, DWP and other sources.
Used for calibrating household weights to match aggregate targets.
"""

import numpy as np
import pandas as pd
from policyengine_uk_data.storage import STORAGE_FOLDER
from policyengine_uk_data.utils import uprate_values
from policyengine_uk.data import UKSingleYearDataset
from policyengine_uk_data.utils.uc_data import uc_national_payment_dist

tax_benefit = pd.read_csv(STORAGE_FOLDER / "tax_benefit.csv")
tax_benefit["name"] = tax_benefit["name"].apply(lambda x: f"obr/{x}")
demographics = pd.read_csv(STORAGE_FOLDER / "demographics.csv")
demographics["name"] = demographics["name"].apply(lambda x: f"ons/{x}")
statistics = pd.concat([tax_benefit, demographics])
dfs = []

MIN_YEAR = 2018
MAX_YEAR = 2029

# NTS 2024 vehicle ownership targets
# https://www.gov.uk/government/statistics/national-travel-survey-2024
NTS_NO_VEHICLE_RATE = 0.22
NTS_ONE_VEHICLE_RATE = 0.44
NTS_TWO_PLUS_VEHICLE_RATE = 0.34

for time_period in range(MIN_YEAR, MAX_YEAR + 1):
    time_period_df = statistics[
        ["name", "unit", "reference", str(time_period)]
    ].rename(columns={str(time_period): "value"})
    time_period_df["time_period"] = time_period
    dfs.append(time_period_df)

statistics = pd.concat(dfs)
statistics = statistics[statistics.value.notnull()]


def create_target_matrix(
    dataset: UKSingleYearDataset,
    time_period: str = None,
    reform=None,
) -> np.ndarray:
    """
    Create target matrix for calibration against official statistics.

    Creates a matrix A such that for household weights w, target vector b
    and a perfectly calibrated PolicyEngine UK: A * w = b

    Compares model outputs against:
    - OBR tax and benefit aggregates
    - ONS demographic and regional statistics
    - HMRC income distribution data
    - DWP benefit caseload data
    - VOA council tax statistics

    Args:
        dataset: PolicyEngine UK dataset to analyse.
        time_period: Year for target statistics (uses dataset default if None).
        reform: Policy reform to apply during analysis.

    Returns:
        Tuple of (target_matrix, target_values) for calibration.
    """

    # First- tax-benefit outcomes from the DWP and OBR.

    from policyengine_uk import Microsimulation

    if time_period is None:
        time_period = dataset.time_period

    sim = Microsimulation(dataset=dataset, reform=reform)
    sim.default_calculation_period = time_period

    family = sim.populations["benunit"]

    pe = lambda variable: sim.calculate(variable, map_to="household").values

    household_from_family = lambda values: sim.map_result(
        values, "benunit", "household"
    )
    household_from_person = lambda values: sim.map_result(
        values, "person", "household"
    )

    def pe_count(*variables):
        total = 0
        for variable in variables:
            entity = sim.tax_benefit_system.variables[variable].entity.key
            total += sim.map_result(
                sim.calculate(variable) > 0,
                entity,
                "household",
            )

        return total

    df = pd.DataFrame()

    df["obr/attendance_allowance"] = pe("attendance_allowance")
    df["obr/carers_allowance"] = pe("carers_allowance")
    df["obr/dla"] = pe("dla")
    df["obr/esa"] = pe("esa_income") + pe("esa_contrib")
    df["obr/esa_contrib"] = pe("esa_contrib")
    df["obr/esa_income"] = pe("esa_income")
    df["obr/housing_benefit"] = pe("housing_benefit")
    df["obr/pip"] = pe("pip")
    df["obr/statutory_maternity_pay"] = pe("statutory_maternity_pay")
    df["obr/attendance_allowance_count"] = pe_count("attendance_allowance")
    df["obr/carers_allowance_count"] = pe_count("carers_allowance")
    df["obr/dla_count"] = pe_count("dla")
    df["obr/esa_count"] = pe_count("esa_income", "esa_contrib")
    df["obr/housing_benefit_count"] = pe_count("housing_benefit")
    df["obr/pension_credit_count"] = pe_count("pension_credit")
    df["obr/pip_count"] = pe_count("pip")

    on_uc = sim.calculate("universal_credit") > 0
    unemployed = family.any(sim.calculate("employment_status") == "UNEMPLOYED")

    df["obr/universal_credit_jobseekers_count"] = household_from_family(
        on_uc * unemployed
    )
    df["obr/universal_credit_non_jobseekers_count"] = household_from_family(
        on_uc * ~unemployed
    )

    # df["obr/winter_fuel_allowance_count"] = pe_count("winter_fuel_allowance")
    df["obr/capital_gains_tax"] = pe("capital_gains_tax")
    df["obr/child_benefit"] = pe("child_benefit")

    country = sim.calculate("country")
    ct = pe("council_tax")
    df["obr/council_tax"] = ct
    df["obr/council_tax_england"] = ct * (country == "ENGLAND")
    df["obr/council_tax_scotland"] = ct * (country == "SCOTLAND")
    df["obr/council_tax_wales"] = ct * (country == "WALES")

    df["obr/domestic_rates"] = pe("domestic_rates")
    df["obr/fuel_duties"] = pe("fuel_duty")
    df["obr/income_tax"] = pe("income_tax")
    df["obr/jobseekers_allowance"] = pe("jsa_income") + pe("jsa_contrib")
    df["obr/pension_credit"] = pe("pension_credit")
    df["obr/state_pension"] = pe("state_pension")
    # df["obr/tax_credits"] = pe("tax_credits")
    df["obr/tv_licence_fee"] = pe("tv_licence")

    uc = sim.calculate("universal_credit")
    df["obr/universal_credit"] = household_from_family(uc)
    df["obr/universal_credit_jobseekers"] = household_from_family(
        uc * unemployed
    )
    df["obr/universal_credit_non_jobseekers"] = household_from_family(
        uc * ~unemployed
    )

    df["obr/vat"] = pe("vat")
    # df["obr/winter_fuel_allowance"] = pe("winter_fuel_allowance")

    # Not strictly from the OBR but from the 2024 Independent Schools Council census. OBR will be using that.
    df["obr/private_school_students"] = pe("attends_private_school")

    # Salary sacrifice NI relief - SPP estimates £4.1bn total (£1.2bn employee + £2.9bn employer)
    # Calculate relief via counterfactual: what additional NI would be paid if SS became income
    ss_contributions = sim.calculate(
        "pension_contributions_via_salary_sacrifice"
    )
    employment_income = sim.calculate("employment_income")

    # Run counterfactual simulation with SS converted to employment income
    counterfactual_sim = Microsimulation(dataset=dataset, reform=reform)
    counterfactual_sim.set_input(
        "pension_contributions_via_salary_sacrifice",
        time_period,
        np.zeros_like(ss_contributions),
    )
    counterfactual_sim.set_input(
        "employment_income",
        time_period,
        employment_income + ss_contributions,
    )

    # NI relief = counterfactual NI - baseline NI
    ni_employee_baseline = sim.calculate("ni_employee")
    ni_employer_baseline = sim.calculate("ni_employer")
    ni_employee_cf = counterfactual_sim.calculate("ni_employee", time_period)
    ni_employer_cf = counterfactual_sim.calculate("ni_employer", time_period)

    employee_ni_relief = ni_employee_cf - ni_employee_baseline
    employer_ni_relief = ni_employer_cf - ni_employer_baseline

    df["obr/salary_sacrifice_employee_ni_relief"] = household_from_person(
        employee_ni_relief
    )
    df["obr/salary_sacrifice_employer_ni_relief"] = household_from_person(
        employer_ni_relief
    )

    # Population statistics from the ONS.

    region = sim.calculate("region", map_to="person")
    region_to_target_name_map = {
        "NORTH_EAST": "north_east",
        "SOUTH_EAST": "south_east",
        "EAST_MIDLANDS": "east_midlands",
        "WEST_MIDLANDS": "west_midlands",
        "YORKSHIRE": "yorkshire_and_the_humber",
        "EAST_OF_ENGLAND": "east",
        "LONDON": "london",
        "SOUTH_WEST": "south_west",
        "NORTH_WEST": "north_west",
        "WALES": "wales",
        "SCOTLAND": "scotland",
        "NORTHERN_IRELAND": "northern_ireland",
    }
    age = sim.calculate("age")

    # Ensure local populations are consistent with national population
    local_population_total = 0
    for pe_region_name, region_name in region_to_target_name_map.items():
        for lower_age in range(0, 90, 10):
            upper_age = lower_age + 10
            name = f"ons/{region_name}_age_{lower_age}_{upper_age - 1}"
            local_population_total += (
                demographics[demographics.name == name][
                    str(time_period)
                ].values[0]
                * 1e3
            )

    population_scaling_factor = (
        demographics[demographics.name == "ons/uk_population"][
            str(time_period)
        ].values[0]
        * 1e6
        / local_population_total
    ) * 0.9

    for pe_region_name, region_name in region_to_target_name_map.items():
        for lower_age in range(0, 90, 10):
            upper_age = lower_age + 10
            name = f"ons/{region_name}_age_{lower_age}_{upper_age - 1}"
            statistics.loc[
                (statistics.name == name)
                & (statistics.time_period == int(time_period)),
                "value",
            ] *= population_scaling_factor

    for pe_region_name, region_name in region_to_target_name_map.items():
        for lower_age in range(0, 90, 10):
            upper_age = lower_age + 10
            name = f"ons/{region_name}_age_{lower_age}_{upper_age - 1}"
            person_in_criteria = (
                (region == pe_region_name)
                & (age >= lower_age)
                & (age < upper_age)
            )
            df[name] = household_from_person(person_in_criteria)

    df["ons/uk_population"] = household_from_person(age >= 0)

    # Scotland-specific calibration targets
    # Children under 16 in Scotland
    # Source: NRS mid-year population estimates
    # https://www.nrscotland.gov.uk/statistics-and-data/statistics/statistics-by-theme/population/population-estimates/mid-year-population-estimates
    scotland_children_under_16 = (region.values == "SCOTLAND") & (age < 16)
    df["ons/scotland_children_under_16"] = household_from_person(
        scotland_children_under_16
    )

    # Babies under 1 in Scotland
    # Source: NRS Vital Events - births registered in Scotland
    # https://www.nrscotland.gov.uk/publications/vital-events-reference-tables-2024/
    # ~46,000 births per year (45,763 in 2024)
    scotland_babies_under_1 = (region.values == "SCOTLAND") & (age < 1)
    df["ons/scotland_babies_under_1"] = household_from_person(
        scotland_babies_under_1
    )

    # Households with 3+ children in Scotland
    # Source: Scotland Census 2022 - Household composition
    # https://www.scotlandscensus.gov.uk/census-results/at-a-glance/household-composition/
    # Count children per household, filter to Scotland households with 3+
    is_child = sim.calculate("is_child").values
    children_per_household = household_from_person(is_child)
    household_region = sim.calculate("region", map_to="household").values
    scotland_3plus_children = (household_region == "SCOTLAND") & (
        children_per_household >= 3
    )
    df["ons/scotland_households_3plus_children"] = (
        scotland_3plus_children.astype(float)
    )

    targets = (
        statistics[statistics.time_period == int(time_period)]
        .set_index("name")
        .loc[df.columns]
    )

    targets.value = np.select(
        [
            targets.unit == "gbp-bn",
            targets.unit == "person-m",
            targets.unit == "person-k",
            targets.unit == "benefit-unit-m",
            targets.unit == "household-k",
        ],
        [
            targets.value * 1e9,
            targets.value * 1e6,
            targets.value * 1e3,
            targets.value * 1e6,
            targets.value * 1e3,
        ],
    )

    # Finally, incomes from HMRC

    target_names = []
    target_values = []

    # Note: savings_interest_income is excluded because SPI significantly
    # underestimates it. Savings income is calibrated from ONS National
    # Accounts D.41g household interest data separately below.
    INCOME_VARIABLES = [
        "employment_income",
        "self_employment_income",
        "state_pension",
        "private_pension_income",
        "property_income",
        "dividend_income",
    ]

    income_df = sim.calculate_dataframe(["total_income"] + INCOME_VARIABLES)

    incomes = pd.read_csv(STORAGE_FOLDER / "incomes_projection.csv")
    incomes = incomes[incomes.year.astype(str) == str(time_period)]
    for i, row in incomes.iterrows():
        lower = row.total_income_lower_bound
        upper = row.total_income_upper_bound
        in_income_band = (income_df.total_income >= lower) & (
            income_df.total_income < upper
        )
        for variable in INCOME_VARIABLES:
            name_amount = (
                "hmrc/"
                + variable
                + f"_income_band_{i}_{lower:_.0f}_to_{upper:_.0f}"
            )
            df[name_amount] = household_from_person(
                income_df[variable] * in_income_band
            )
            target_values.append(row[variable + "_amount"])
            target_names.append(name_amount)
            name_count = (
                "hmrc/"
                + variable
                + f"_count_income_band_{i}_{lower:_.0f}_to_{upper:_.0f}"
            )
            df[name_count] = household_from_person(
                (income_df[variable] > 0) * in_income_band
            )
            target_values.append(row[variable + "_count"])
            target_names.append(name_count)

    # Savings interest income from ONS National Accounts D.41
    # Source: ONS HAXV - Households (S.14): Interest (D.41) Resources
    # https://www.ons.gov.uk/economy/grossdomesticproductgdp/timeseries/haxv/ukea
    # SPI significantly underestimates savings income (~£3bn vs £43-98bn actual)
    # because it only captures taxable interest, not tax-free ISAs/NS&I
    ONS_SAVINGS_INCOME = {
        2020: 16.0e9,
        2021: 19.6e9,
        2022: 43.3e9,
        2023: 86.0e9,
        2024: 98.2e9,
        2025: 98.2e9,  # Projected (held flat)
        2026: 98.2e9,
        2027: 98.2e9,
        2028: 98.2e9,
        2029: 98.2e9,
    }
    savings_income = sim.calculate("savings_interest_income")
    df["ons/savings_interest_income"] = household_from_person(savings_income)
    target_names.append("ons/savings_interest_income")
    target_values.append(ONS_SAVINGS_INCOME.get(int(time_period), 55.0e9))

    # HMRC Table 6.2 - Salary sacrifice income tax relief by tax rate
    # This helps calibrate the distribution of SS users by income level
    # 2023-24 values (£m): Basic £1,600, Higher £4,400, Additional £1,200
    # Total IT relief from SS: £7,200m
    # Use true counterfactual: IT relief = counterfactual IT - baseline IT
    income_tax_baseline = sim.calculate("income_tax")
    income_tax_cf = counterfactual_sim.calculate("income_tax", time_period)
    it_relief = income_tax_cf - income_tax_baseline

    # Get tax band from counterfactual adjusted net income (where SS is wages)
    adjusted_net_income_cf = counterfactual_sim.calculate(
        "adjusted_net_income", time_period
    )
    basic_rate_threshold = (
        sim.tax_benefit_system.parameters.gov.hmrc.income_tax.rates.uk[
            0
        ].threshold(time_period)
    )
    higher_rate_threshold = (
        sim.tax_benefit_system.parameters.gov.hmrc.income_tax.rates.uk[
            1
        ].threshold(time_period)
    )
    additional_rate_threshold = (
        sim.tax_benefit_system.parameters.gov.hmrc.income_tax.rates.uk[
            2
        ].threshold(time_period)
    )

    # Determine tax band for each person based on counterfactual income
    is_basic_rate = (adjusted_net_income_cf > basic_rate_threshold) & (
        adjusted_net_income_cf <= higher_rate_threshold
    )
    is_higher_rate = (adjusted_net_income_cf > higher_rate_threshold) & (
        adjusted_net_income_cf <= additional_rate_threshold
    )
    is_additional_rate = adjusted_net_income_cf > additional_rate_threshold

    # Allocate the true IT relief to tax bands
    ss_it_relief_basic = it_relief * is_basic_rate
    ss_it_relief_higher = it_relief * is_higher_rate
    ss_it_relief_additional = it_relief * is_additional_rate

    df["hmrc/salary_sacrifice_it_relief_basic"] = household_from_person(
        ss_it_relief_basic
    )
    df["hmrc/salary_sacrifice_it_relief_higher"] = household_from_person(
        ss_it_relief_higher
    )
    df["hmrc/salary_sacrifice_it_relief_additional"] = household_from_person(
        ss_it_relief_additional
    )

    # Total gross salary sacrifice contributions
    # This is derived from the IT relief: £7.2bn IT relief at ~30% avg rate
    # implies ~£24bn gross contributions (but we target the relief directly)
    df["hmrc/salary_sacrifice_contributions"] = household_from_person(
        ss_contributions
    )

    # HMRC Table 6.2 - Salary sacrifice income tax relief by tax rate (2023-24)
    # https://assets.publishing.service.gov.uk/media/687a294e312ee8a5f0806b6d/Tables_6_1_and_6_2.csv
    # Values in £bn
    SS_IT_RELIEF_BASIC_2024 = 1.6e9
    SS_IT_RELIEF_HIGHER_2024 = 4.4e9
    SS_IT_RELIEF_ADDITIONAL_2024 = 1.2e9
    SS_CONTRIBUTIONS_2024 = 24e9  # £7.2bn IT relief / 0.30 avg rate

    # Uprate by ~3% per year for wage growth
    years_from_2024 = max(0, int(time_period) - 2024)
    uprating_factor = 1.03**years_from_2024

    target_names.append("hmrc/salary_sacrifice_it_relief_basic")
    target_values.append(SS_IT_RELIEF_BASIC_2024 * uprating_factor)

    target_names.append("hmrc/salary_sacrifice_it_relief_higher")
    target_values.append(SS_IT_RELIEF_HIGHER_2024 * uprating_factor)

    target_names.append("hmrc/salary_sacrifice_it_relief_additional")
    target_values.append(SS_IT_RELIEF_ADDITIONAL_2024 * uprating_factor)

    target_names.append("hmrc/salary_sacrifice_contributions")
    target_values.append(SS_CONTRIBUTIONS_2024 * uprating_factor)

    # Salary sacrifice headcount targets
    # Source: HMRC, "Salary sacrifice reform for pension contributions"
    # https://www.gov.uk/government/publications/salary-sacrifice-reform-for-pension-contributions-effective-from-6-april-2029
    # 7.7mn total SS users (3.3mn above £2k cap, 4.3mn below £2k cap)
    # The £2,000 cap is defined at 2023-24 FRS prices. The dataset is
    # uprated to 2025 for calibration then downrated back to 2023 for
    # saving. To keep the above/below classification consistent across
    # price years, evaluate SS amounts at 2023-24 base-year prices.
    ss_uprating_factors = pd.read_csv(
        STORAGE_FOLDER / "uprating_factors.csv"
    ).set_index("Variable")
    ss_price_adjustment = (
        ss_uprating_factors.loc[
            "pension_contributions_via_salary_sacrifice", "2023"
        ]
        / ss_uprating_factors.loc[
            "pension_contributions_via_salary_sacrifice", str(time_period)
        ]
    )
    ss_at_base_prices = ss_contributions * ss_price_adjustment
    ss_has_contributions = ss_at_base_prices > 0
    ss_below_cap = ss_has_contributions & (ss_at_base_prices <= 2000)
    ss_above_cap = ss_has_contributions & (ss_at_base_prices > 2000)

    df["obr/salary_sacrifice_users_total"] = household_from_person(
        ss_has_contributions
    )
    df["obr/salary_sacrifice_users_below_cap"] = household_from_person(
        ss_below_cap
    )
    df["obr/salary_sacrifice_users_above_cap"] = household_from_person(
        ss_above_cap
    )

    # HMRC/ASHE 2024 baseline headcounts
    SS_TOTAL_USERS_2024 = 7_700_000
    SS_BELOW_CAP_USERS_2024 = 4_300_000
    SS_ABOVE_CAP_USERS_2024 = 3_300_000
    # OBR (5 Feb 2026, para 1.7): SS population grows 0.9% faster than
    # total employee numbers. With ~1.5% employment growth, ~2.4%/year.
    ss_headcount_factor = 1.024 ** max(0, int(time_period) - 2024)

    target_names.append("obr/salary_sacrifice_users_total")
    target_values.append(SS_TOTAL_USERS_2024 * ss_headcount_factor)

    target_names.append("obr/salary_sacrifice_users_below_cap")
    target_values.append(SS_BELOW_CAP_USERS_2024 * ss_headcount_factor)

    target_names.append("obr/salary_sacrifice_users_above_cap")
    target_values.append(SS_ABOVE_CAP_USERS_2024 * ss_headcount_factor)

    # Add two-child limit targets.
    child_is_affected = (
        sim.map_result(
            sim.calculate("uc_is_child_limit_affected", map_to="household"),
            "household",
            "person",
        )
        > 0
    ) * sim.calculate("is_child", map_to="person").values
    child_in_uc_household = (
        sim.calculate("universal_credit", map_to="person").values > 0
    )
    children_in_capped_households = sim.map_result(
        child_is_affected * child_in_uc_household, "person", "household"
    )
    capped_households = (children_in_capped_households > 0) * 1.0
    df["dwp/uc_two_child_limit_affected_child_count"] = (
        children_in_capped_households
    )
    target_names.append("dwp/uc_two_child_limit_affected_child_count")
    UPRATING_24_25 = 1.12  # https://ifs.org.uk/articles/two-child-limit-poverty-incentives-and-cost, table at the end
    target_values.append(1.6e6 * UPRATING_24_25)  # DWP statistics for 2024/25
    # https://www.gov.uk/government/statistics/universal-credit-and-child-tax-credit-claimants-statistics-related-to-the-policy-to-provide-support-for-a-maximum-of-2-children-april-2024/universal-credit-and-child-tax-credit-claimants-statistics-related-to-the-policy-to-provide-support-for-a-maximum-of-two-children-april-2024
    df["dwp/uc_two_child_limit_affected_household_count"] = capped_households
    target_names.append("dwp/uc_two_child_limit_affected_household_count")
    target_values.append(440e3 * UPRATING_24_25)  # DWP statistics for 2024/25

    # PIP daily living standard and enhanced claimant counts
    # https://www.disabilityrightsuk.org/news/90-pip-standard-daily-living-component-recipients-would-fail-new-green-paper-test?srsltid=AfmBOoqSq3cQwtZnQBe-qLN7PT1mUBVtZ0ZINYtoG5bG5O9_ObQ90Y0n

    pip_dl_category = sim.calculate("pip_dl_category")
    on_standard = sim.map_result(
        pip_dl_category == "STANDARD", "person", "household"
    )
    on_enhanced = sim.map_result(
        pip_dl_category == "ENHANCED", "person", "household"
    )

    df["dwp/pip_dl_standard_claimants"] = on_standard
    target_names.append("dwp/pip_dl_standard_claimants")
    target_values.append(1_283_000)

    df["dwp/pip_dl_enhanced_claimants"] = on_enhanced
    target_names.append("dwp/pip_dl_enhanced_claimants")
    target_values.append(1_608_000)

    # Scottish Child Payment total spend
    # Source: Scottish Budget 2026-27, Table 5.08
    # https://www.gov.scot/publications/scottish-budget-2026-2027/pages/6/
    scp = sim.calculate("scottish_child_payment")
    df["sss/scottish_child_payment"] = household_from_person(scp)
    SCP_SPEND = {
        2024: 455.8e6,
        2025: 471.0e6,
        2026: 484.8e6,
    }
    # Extrapolate for other years using 3% annual growth
    scp_target = SCP_SPEND.get(
        int(time_period), 471.0e6 * (1.03 ** (int(time_period) - 2025))
    )
    target_names.append("sss/scottish_child_payment")
    target_values.append(scp_target)

    # UC households in Scotland with child under 1
    # Source: DWP Stat-Xplore, UC Households dataset, November 2023
    # https://stat-xplore.dwp.gov.uk/
    # Filters: Scotland, Age of Youngest Child = 0
    # ~14,000 households (13,992 in November 2023)
    uc_amount = sim.calculate("universal_credit")
    on_uc_family = uc_amount > 0
    on_uc_household = household_from_family(on_uc_family) > 0

    child_under_1 = is_child & (age < 1)
    has_child_under_1 = household_from_person(child_under_1) > 0

    scotland_uc_child_under_1 = (
        (household_region == "SCOTLAND") & on_uc_household & has_child_under_1
    )
    df["dwp/scotland_uc_households_child_under_1"] = (
        scotland_uc_child_under_1.astype(float)
    )
    target_names.append("dwp/scotland_uc_households_child_under_1")
    target_values.append(14_000)  # 13,992 rounded, November 2023

    # Council Tax band counts

    ct_data = pd.read_csv(STORAGE_FOLDER / "council_tax_bands_2024.csv")
    uk_population = (
        sim.tax_benefit_system.parameters.gov.economic_assumptions.indices.ons.population
    )
    uprating = uk_population(time_period) / uk_population(2024)

    # England and Wales data from https://www.gov.uk/government/statistics/council-tax-stock-of-properties-2024

    # Scotland data from https://www.gov.scot/publications/council-tax-datasets/ (Number of chargeable dwellings, 2024)

    for i, row in ct_data.iterrows():
        selected_region = row["Region"]
        in_region = sim.calculate("region").values == selected_region
        for band in ["A", "B", "C", "D", "E", "F", "G", "H"]:
            name = f"voa/council_tax/{selected_region}/{band}"
            in_band = sim.calculate("council_tax_band") == band
            df[name] = (in_band * in_region).astype(float)
            target_names.append(name)
            target_values.append(float(row[band]) * uprating)
        # Add total row
        name = f"voa/council_tax/{selected_region}/total"
        df[name] = (in_region).astype(float)
        target_names.append(name)
        target_values.append(float(row["Total"]) * uprating)

    # Benefit cap counts

    benefit_cap_reduction = sim.calculate(
        "benefit_cap_reduction", map_to="household"
    ).values
    df["dwp/benefit_capped_households"] = (benefit_cap_reduction > 0).astype(
        float
    )
    target_names.append("dwp/benefit_capped_households")
    target_values.append(
        115_000
    )  # https://www.gov.uk/government/statistics/benefit-cap-number-of-households-capped-to-february-2025/benefit-cap-number-of-households-capped-to-february-2025

    df["dwp/benefit_cap_total_reduction"] = benefit_cap_reduction.astype(float)
    target_names.append("dwp/benefit_cap_total_reduction")
    target_values.append(
        60 * 52 * 115_000
    )  # same source as above, multiply avg cap amount by total capped population

    # UC national payment distribution

    uc_payment_dist = uc_national_payment_dist
    uc_payments = sim.calculate("universal_credit", map_to="benunit").values
    uc_family_type = sim.calculate("family_type", map_to="benunit").values

    for i, row in uc_payment_dist.iterrows():
        lower = row.uc_annual_payment_min
        upper = row.uc_annual_payment_max
        family_type = row.family_type
        in_band = (
            (uc_payments >= lower)
            & (uc_payments < upper)
            & (uc_family_type == family_type)
        )
        name = f"dwp/uc_payment_dist/{family_type}_annual_payment_{lower:_.0f}_to_{upper:_.0f}"
        df[name] = household_from_family(in_band)
        target_names.append(name)
        target_values.append(row.household_count)

    # Vehicle ownership calibration targets
    # NTS 2024: 22% no car, 44% one car, 34% two+ cars
    # https://www.gov.uk/government/statistics/national-travel-survey-2024
    # Total households (~29.6m) from council tax data (consistent with other calibration)
    total_households = ct_data["Total"].sum() * uprating
    num_vehicles = pe("num_vehicles")

    df["nts/households_no_vehicle"] = (num_vehicles == 0).astype(float)
    target_names.append("nts/households_no_vehicle")
    target_values.append(total_households * NTS_NO_VEHICLE_RATE)

    df["nts/households_one_vehicle"] = (num_vehicles == 1).astype(float)
    target_names.append("nts/households_one_vehicle")
    target_values.append(total_households * NTS_ONE_VEHICLE_RATE)

    df["nts/households_two_plus_vehicles"] = (num_vehicles >= 2).astype(float)
    target_names.append("nts/households_two_plus_vehicles")
    target_values.append(total_households * NTS_TWO_PLUS_VEHICLE_RATE)

    RENT_ESTIMATE = {
        "private_renter": 1_400
        * 12
        * 4.7e6,  # https://www.ons.gov.uk/economy/inflationandpriceindices/bulletins/privaterentandhousepricesuk/january2025
        "owner_mortgage": 1_100 * 12 * 7.5e6,
    }

    # Housing affordability targets
    # Total mortgage payments (capital + interest)
    mortgage_capital = pe("mortgage_capital_repayment")
    mortgage_interest = pe("mortgage_interest_repayment")
    total_mortgage = mortgage_capital + mortgage_interest
    df["housing/total_mortgage"] = total_mortgage
    target_names.append("housing/total_mortgage")
    target_values.append(RENT_ESTIMATE["owner_mortgage"])

    # Total rent by tenure type
    rent = pe("rent")
    tenure_type = sim.calculate("tenure_type", map_to="household").values

    df["housing/rent_private"] = rent * (tenure_type == "RENT_PRIVATELY")
    target_names.append("housing/rent_private")
    target_values.append(RENT_ESTIMATE["private_renter"])

    combined_targets = pd.concat(
        [
            targets,
            pd.DataFrame(
                {
                    "value": target_values,
                },
                index=target_names,
            ),
        ]
    )

    combined_targets.to_csv("test.csv")

    return df, combined_targets.value


def get_loss_results(
    dataset, time_period, reform=None, household_weights=None
):
    """
    Calculate loss metrics comparing model outputs to targets.

    Args:
        dataset: PolicyEngine UK dataset to evaluate.
        time_period: Year for comparison.
        reform: Policy reform to apply.
        household_weights: Custom weights (uses dataset weights if None).

    Returns:
        DataFrame with estimate vs target comparisons and error metrics.
    """
    matrix, targets = create_target_matrix(dataset, time_period, reform)
    from policyengine_uk import Microsimulation

    if household_weights is None:
        weights = (
            Microsimulation(dataset=dataset, reform=reform)
            .calculate("household_weight", time_period)
            .values
        )
    else:
        weights = household_weights
    estimates = weights @ matrix
    df = pd.DataFrame(
        {
            "name": estimates.index,
            "estimate": estimates.values,
            "target": targets,
        },
    )
    df["error"] = df["estimate"] - df["target"]
    df["abs_error"] = df["error"].abs()
    df["rel_error"] = df["error"] / df["target"]
    df["abs_rel_error"] = df["rel_error"].abs()
    return df.reset_index(drop=True)
