import numpy as np
import pandas as pd
from policyengine_uk_data.storage import STORAGE_FOLDER
from policyengine_uk_data.utils import uprate_values
import warnings
from policyengine_uk import Microsimulation
from policyengine_uk_data.utils.reweight import reweight
from policyengine_uk_data.datasets import SPI_2020_21

warnings.filterwarnings("ignore")

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

    household_from_person = lambda values: sim.map_result(
        values, "person", "household"
    )

    df = pd.DataFrame()

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

    incomes = pd.read_csv(STORAGE_FOLDER / "incomes.csv")
    for variable in INCOME_VARIABLES:
        incomes[variable + "_count"] = uprate_values(
            incomes[variable + "_count"], "household_weight", 2021, time_period
        )
        incomes[variable + "_amount"] = uprate_values(
            incomes[variable + "_amount"], variable, 2021, time_period
        )

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

    combined_targets = pd.DataFrame(
        {
            "value": target_values,
        },
        index=target_names,
    )

    return df, combined_targets.value


def get_loss_results(dataset, time_period, reform=None):
    matrix, targets = create_target_matrix(dataset, time_period, reform)
    from policyengine_uk import Microsimulation

    weights = (
        Microsimulation(dataset=dataset, reform=reform)
        .calculate("household_weight", time_period)
        .values
    )
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


def create_income_projections():
    loss_matrix, targets_array = create_target_matrix(SPI_2020_21, 2022)

    sim = Microsimulation(dataset=SPI_2020_21)
    household_weights = sim.calculate("household_weight", 2022).values

    reweighted_weights = reweight(
        household_weights,
        loss_matrix,
        targets_array,
        epochs=1_000,
    )

    sim = Microsimulation(dataset=SPI_2020_21)
    sim.set_input("household_weight", 2022, reweighted_weights)

    INCOME_VARIABLES = [
        "employment_income",
        "self_employment_income",
        "state_pension",
        "private_pension_income",
        "property_income",
        "savings_interest_income",
        "dividend_income",
    ]

    incomes = pd.read_csv(STORAGE_FOLDER / "incomes.csv")

    projection_df = pd.DataFrame()
    lower_bounds = incomes.total_income_lower_bound
    upper_bounds = incomes.total_income_upper_bound

    for year in range(2022, 2030):
        year_df = pd.DataFrame()
        year_df["total_income_lower_bound"] = lower_bounds
        year_df["total_income_upper_bound"] = upper_bounds
        for variable in INCOME_VARIABLES:
            count_values = []
            amount_values = []
            for i, (lower, upper) in enumerate(
                zip(lower_bounds, upper_bounds)
            ):
                in_band = sim.calculate("total_income", year).between(
                    lower, upper
                )
                value = sim.calculate(variable, year)
                count_in_band_with_nonzero_value = round(
                    ((value > 0) * in_band).sum()
                )
                amount_in_band = round(value[in_band].sum())
                count_values.append(count_in_band_with_nonzero_value)
                amount_values.append(amount_in_band)
            year_df[f"{variable}_count"] = count_values
            year_df[f"{variable}_amount"] = amount_values
        year_df["year"] = year
        projection_df = pd.concat([projection_df, year_df])

    projection_df.to_csv(
        STORAGE_FOLDER / "incomes_projection.csv", index=False
    )


if __name__ == "__main__":
    create_income_projections()
