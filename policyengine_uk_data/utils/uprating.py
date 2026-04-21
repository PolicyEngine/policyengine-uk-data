from policyengine_uk_data.storage import STORAGE_FOLDER
import pandas as pd
from policyengine_uk.data import UKSingleYearDataset

START_YEAR = 2020
END_YEAR = 2034


class UpratingYearOutOfRangeError(ValueError):
    """Raised when a caller asks for an uprating factor outside the table range.

    The uprating factor table is written by `create_policyengine_uprating_factors_table()`
    for years in ``[START_YEAR, END_YEAR]`` and does not include columns for years
    beyond ``END_YEAR``. Silently returning a KeyError (or a stale last-year factor)
    would produce wrong values; raising a specific, actionable error instead tells
    the caller to either regenerate the table with a later ``END_YEAR`` or pick a
    supported target year.
    """


def _check_year_in_range(year: int, *, kind: str) -> None:
    if year < START_YEAR or year > END_YEAR:
        raise UpratingYearOutOfRangeError(
            f"{kind}={year} is outside the uprating table range "
            f"[{START_YEAR}, {END_YEAR}]. Regenerate the table via "
            f"`create_policyengine_uprating_factors_table()` with a later "
            f"END_YEAR, or pick a supported year."
        )


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


def uprate_values(values, variable_name, start_year=START_YEAR, end_year=END_YEAR):
    _check_year_in_range(start_year, kind="start_year")
    _check_year_in_range(end_year, kind="end_year")

    uprating_factors = pd.read_csv(STORAGE_FOLDER / "uprating_factors.csv")
    uprating_factors = uprating_factors.set_index("Variable")
    uprating_factors = uprating_factors.loc[variable_name]

    initial_index = uprating_factors[str(start_year)]
    end_index = uprating_factors[str(end_year)]
    relative_change = end_index / initial_index

    return values * relative_change


def uprate_dataset(dataset: UKSingleYearDataset, target_year: int = END_YEAR):
    _check_year_in_range(target_year, kind="target_year")

    dataset = dataset.copy()
    uprating_factors = pd.read_csv(STORAGE_FOLDER / "uprating_factors.csv")
    uprating_factors = uprating_factors.set_index("Variable")
    start_year = dataset.time_period
    _check_year_in_range(int(start_year), kind="dataset.time_period")

    for table in dataset.tables:
        for variable in table.columns:
            if variable in uprating_factors.index:
                factor = (
                    uprating_factors.loc[variable, str(target_year)]
                    / uprating_factors.loc[variable, str(start_year)]
                )
                table[variable] *= factor

    dataset.time_period = target_year

    return dataset


if __name__ == "__main__":
    create_policyengine_uprating_factors_table()
