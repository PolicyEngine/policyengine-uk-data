from policyengine_uk_data.storage import STORAGE_FOLDER
import pandas as pd
from policyengine_uk.data import UKSingleYearDataset

START_YEAR = 2020
END_YEAR = 2034


def create_policyengine_uprating_factors_table():
    from policyengine_uk.system import system

    df = pd.DataFrame()

    variable_names = []
    years = []
    index_values = []

    for variable in system.variables.values():
        if variable.uprating is not None:
            parameter = system.parameters.get_child(variable.uprating)
            start_value = parameter(START_YEAR)
            for year in range(START_YEAR, END_YEAR + 1):
                variable_names.append(variable.name)
                years.append(year)
                growth = parameter(year) / start_value
                index_values.append(round(growth, 3))

    df["Variable"] = variable_names
    df["Year"] = years
    df["Value"] = index_values

    # Convert to there is a column for each year
    df = df.pivot(index="Variable", columns="Year", values="Value")
    df = df.sort_values("Variable")
    df.to_csv(STORAGE_FOLDER / "uprating_factors.csv")

    # Create a table with growth factors by year

    df_growth = df.copy()
    for year in range(END_YEAR, START_YEAR, -1):
        df_growth[year] = round(df_growth[year] / df_growth[year - 1] - 1, 3)
    df_growth[START_YEAR] = 0

    df_growth.to_csv(STORAGE_FOLDER / "uprating_growth_factors.csv")
    return df


def uprate_values(values, variable_name, start_year=2020, end_year=2034):
    uprating_factors = pd.read_csv(STORAGE_FOLDER / "uprating_factors.csv")
    uprating_factors = uprating_factors.set_index("Variable")
    uprating_factors = uprating_factors.loc[variable_name]

    initial_index = uprating_factors[str(start_year)]
    end_index = uprating_factors[str(end_year)]
    relative_change = end_index / initial_index

    return values * relative_change


def uprate_dataset(dataset: UKSingleYearDataset, target_year=2034):
    import numpy as np

    dataset = dataset.copy()
    uprating_factors = pd.read_csv(STORAGE_FOLDER / "uprating_factors.csv")
    uprating_factors = uprating_factors.set_index("Variable")
    start_year = dataset.time_period

    for table in dataset.tables:
        for variable in table.columns:
            if variable in uprating_factors.index:
                factor = (
                    uprating_factors.loc[variable, str(target_year)]
                    / uprating_factors.loc[variable, str(start_year)]
                )
                table[variable] *= factor

    dataset.time_period = target_year

    # Re-derive Plan 1/2/5 from age at the target year.
    # Plan 4 (Scotland) and Postgraduate are left unchanged.
    if "age" in dataset.person.columns and "student_loan_plan" in dataset.person.columns:
        age = dataset.person["age"][:]
        existing = dataset.person["student_loan_plan"][:]
        mask = np.isin(existing, ["PLAN_1", "PLAN_2", "PLAN_5"])
        start = target_year - age + 18
        plan = existing.copy()
        plan[mask & (start < 2012)] = "PLAN_1"
        plan[mask & (start >= 2012) & (start < 2023)] = "PLAN_2"
        plan[mask & (start >= 2023)] = "PLAN_5"
        dataset.person["student_loan_plan"] = plan

    return dataset


if __name__ == "__main__":
    create_policyengine_uprating_factors_table()
