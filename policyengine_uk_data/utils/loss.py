import numpy as np
import pandas as pd
from policyengine_uk_data.storage import STORAGE_FOLDER
from policyengine_uk_data.utils import uprate_values

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
    dataset: str,
    time_period: str,
    reform=None,
) -> np.ndarray:
    """
    Create a target matrix A, s.t. for household weights w, the target vector b and a perfectly calibrated PolicyEngine UK:

    A * w = b

    """

    # First- tax-benefit outcomes from the DWP and OBR.

    from policyengine_uk import Microsimulation

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
    df["obr/tax_credits"] = pe("tax_credits")
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
        "WALES": "wales",
        "SCOTLAND": "scotland",
        "NORTHERN_IRELAND": "northern_ireland",
    }
    age = sim.calculate("age")
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
