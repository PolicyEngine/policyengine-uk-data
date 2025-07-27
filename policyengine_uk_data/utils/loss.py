import numpy as np
import pandas as pd
from policyengine_uk_data.storage import STORAGE_FOLDER
from policyengine_uk_data.utils import uprate_values
from policyengine_uk.data import UKSingleYearDataset

tax_benefit = pd.read_csv(STORAGE_FOLDER / "tax_benefit.csv")
tax_benefit["name"] = tax_benefit["name"].apply(lambda x: f"obr/{x}")
demographics = pd.read_csv(STORAGE_FOLDER / "demographics.csv")
demographics["name"] = demographics["name"].apply(lambda x: f"ons/{x}")
statistics = pd.concat([tax_benefit, demographics])
dfs = []

MIN_YEAR = 2018
MAX_YEAR = 2029

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
    Create a target matrix A, s.t. for household weights w, the target vector b and a perfectly calibrated PolicyEngine UK:

    A * w = b

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

    df["obr/winter_fuel_allowance_count"] = pe_count("winter_fuel_allowance")
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
    df["obr/stamp_duty_land_tax"] = pe("expected_sdlt")
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
    df["obr/winter_fuel_allowance"] = pe("winter_fuel_allowance")

    # Not strictly from the OBR but from the 2024 Independent Schools Council census. OBR will be using that.
    df["obr/private_school_students"] = pe("attends_private_school")

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

    INCOME_VARIABLES = [
        "employment_income",
        "self_employment_income",
        "state_pension",
        "private_pension_income",
        "property_income",
        "savings_interest_income",
        "dividend_income",
    ]

    income_df = sim.calculate_dataframe(["total_income"] + INCOME_VARIABLES)

    incomes = pd.read_csv(STORAGE_FOLDER / "incomes_projection.csv")
    incomes = incomes[incomes.year == time_period]
    for i, row in incomes.iterrows():
        lower = row.total_income_lower_bound
        upper = row.total_income_upper_bound
        in_income_band = (income_df.total_income >= lower) & (
            income_df.total_income < upper
        )
        for variable in INCOME_VARIABLES:
            name_amount = (
                "hmrc/" + variable + f"_income_band_{i}_{lower:_}_to_{upper:_}"
            )
            df[name_amount] = household_from_person(
                income_df[variable] * in_income_band
            )
            target_values.append(row[variable + "_amount"])
            target_names.append(name_amount)
            name_count = (
                "hmrc/"
                + variable
                + f"_count_income_band_{i}_{lower:_}_to_{upper:_}"
            )
            df[name_count] = household_from_person(
                (income_df[variable] > 0) * in_income_band
            )
            target_values.append(row[variable + "_count"])
            target_names.append(name_count)

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
        for band in ["A", "B", "C", "D", "E", "F", "G", "H", "I"]:
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

    return df, combined_targets.value


def get_loss_results(
    dataset, time_period, reform=None, household_weights=None
):
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
