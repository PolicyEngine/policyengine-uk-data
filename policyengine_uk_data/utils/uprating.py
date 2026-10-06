from pathlib import Path

from policyengine_uk_data.storage import STORAGE_FOLDER
import pandas as pd
import yaml
from policyengine_uk.data import UKSingleYearDataset

START_YEAR = 2020
END_YEAR = 2034
# Decimal places stored in uprating_factors.csv. At 3 places the rounding
# alone moved year-on-year factors by up to 0.1%.
UPRATING_TABLE_DECIMALS = 6

# These variables are named as spending, but PolicyEngine UK derives fuel
# litres through ``litres = spending / price`` and uprates household weights
# separately. Uprate fuel proxies by road-fuel volume per weighted household
# and model pump prices so weighted litres follow HMRC/OBR clearances.
VOLUME_OVERRIDDEN_VARIABLES = ("petrol_spending", "diesel_spending")
FUEL_PRICE_PARAMETER_NAME = {
    "petrol_spending": "petrol",
    "diesel_spending": "diesel",
}
HOUSEHOLD_WEIGHT_UPRATING_INDEX = {
    2020: 1.0,
    2021: 1.0,
    2022: 1.003,
    2023: 1.017,
    2024: 1.027,
    2025: 1.039,
    2026: 1.046,
    2027: 1.054,
    2028: 1.058,
    2029: 1.064,
    2030: 1.064,
    2031: 1.064,
    2032: 1.064,
    2033: 1.064,
    2034: 1.064,
}


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


def policyengine_uk_uprating_indices() -> dict[str, list[str]]:
    """The growth index policyengine-uk projects each dataset variable by.

    Read from the ``uprating_indices.yaml`` that
    ``policyengine_uk.data.economic_assumptions`` loads: a mapping from a
    year-on-year growth parameter to the variables it uprates.
    """
    from policyengine_uk.data import economic_assumptions

    path = Path(economic_assumptions.__file__).with_name("uprating_indices.yaml")
    with open(path) as f:
        return yaml.safe_load(f)


def policyengine_uk_load_time_index(
    start_year: int = START_YEAR, end_year: int = END_YEAR
) -> pd.DataFrame:
    """Level index (``start_year`` = 1) that policyengine-uk applies to data.

    policyengine-uk projects a single-year dataset to later years with
    ``economic_assumptions.apply_single_year_uprating``: each year, every
    variable listed in ``uprating_indices.yaml`` is multiplied by one plus the
    year-on-year growth parameter it is listed under. Variables the file does
    not list are carried forward unchanged, and a variable's own ``uprating``
    attribute only applies after the last year the dataset is projected to,
    so neither appears here. Council tax and rent, which the engine uprates by
    country and region, are not single indices and are not covered.
    """
    from policyengine_uk.system import system

    rows = {}
    for index_name, variables in policyengine_uk_uprating_indices().items():
        growth = system.parameters.get_child(index_name)
        level = [1.0]
        for year in range(start_year + 1, end_year + 1):
            level.append(level[-1] * (1 + growth(str(year))))
        for variable in variables:
            if variable in rows:
                raise ValueError(
                    f"{variable} is listed under two indices in "
                    "policyengine-uk's uprating_indices.yaml."
                )
            rows[variable] = level
    df = pd.DataFrame.from_dict(
        rows, orient="index", columns=range(start_year, end_year + 1)
    )
    df.index.name = "Variable"
    return df.sort_index()


def build_uprating_factors_table() -> pd.DataFrame:
    """The uprating factor table: policyengine-uk's load-time index, two overrides.

    ``uprate_dataset`` moves the build between the FRS base year and the
    calibration year with this table, and policyengine-uk moves the saved file
    with its own load-time index, so the two must agree for calibrated values
    to be the values the model runs on.
    """
    df = policyengine_uk_load_time_index().round(UPRATING_TABLE_DECIMALS)

    # Keep the population calibration row stable. Current PolicyEngine UK
    # population indices would inflate final calibrated population above the
    # existing fidelity guard.
    df = _apply_household_weight_uprating_override(df)

    # Ensure petrol/diesel use sourced road-fuel clearances and model pump
    # prices. This keeps litres aligned after PolicyEngine divides by price.
    df = _apply_road_fuel_litre_proxy_override(df)
    return df


def create_policyengine_uprating_factors_table():
    df = build_uprating_factors_table()
    df.to_csv(STORAGE_FOLDER / "uprating_factors.csv")

    # Create a table with growth factors by year

    df_growth = df.copy()
    for year in range(END_YEAR, START_YEAR, -1):
        df_growth[year] = round(
            df_growth[year] / df_growth[year - 1] - 1, UPRATING_TABLE_DECIMALS
        )
    df_growth[START_YEAR] = 0

    df_growth.to_csv(STORAGE_FOLDER / "uprating_growth_factors.csv")
    return df


def _apply_household_weight_uprating_override(df: pd.DataFrame) -> pd.DataFrame:
    """Restore the household-weight index used by the calibrated dataset."""
    if "household_weight" not in df.index:
        return df

    missing_years = [
        year
        for year in range(START_YEAR, END_YEAR + 1)
        if year not in HOUSEHOLD_WEIGHT_UPRATING_INDEX
    ]
    if missing_years:
        raise ValueError(
            "Household-weight uprating index missing years: "
            + ", ".join(str(year) for year in missing_years)
        )
    for year in range(START_YEAR, END_YEAR + 1):
        df.loc["household_weight", year] = HOUSEHOLD_WEIGHT_UPRATING_INDEX[year]
    return df


def fuel_spending_litre_proxy_index(
    *,
    variable: str,
    base_year: int = START_YEAR,
    end_year: int = END_YEAR,
    parameters=None,
    household_weight_index: dict[int, float] | pd.Series | None = None,
) -> dict[int, float]:
    """Return the spending-proxy index that preserves fuel litres.

    PolicyEngine derives litres as ``spending / price`` and separately uprates
    household weights. Therefore the spending proxy must grow with physical
    road-fuel volumes per weighted household and the model pump-price
    denominator.
    """
    from policyengine_uk.system import system
    from policyengine_uk_data.sources.road_fuel_volume import (
        road_fuel_volume_index,
    )

    if variable not in FUEL_PRICE_PARAMETER_NAME:
        raise ValueError(f"Unsupported fuel variable: {variable}")

    if parameters is None:
        parameters = system.parameters

    volume_index = road_fuel_volume_index(base_year=base_year, end_year=end_year)
    price_parameter = getattr(
        parameters.household.consumption.fuel.prices,
        FUEL_PRICE_PARAMETER_NAME[variable],
    )
    base_price = price_parameter(base_year)

    if household_weight_index is None:
        population = parameters.gov.economic_assumptions.indices.ons.population
        base_household_weight_index = population(base_year)

        def household_weight_relative(year: int) -> float:
            return population(year) / base_household_weight_index

    else:

        def household_weight_value(year: int) -> float:
            if hasattr(household_weight_index, "index"):
                if year in household_weight_index.index:
                    return float(household_weight_index.loc[year])
                return float(household_weight_index.loc[str(year)])
            try:
                return float(household_weight_index[year])
            except KeyError:
                return float(household_weight_index[str(year)])

        base_household_weight_index = household_weight_value(base_year)

        def household_weight_relative(year: int) -> float:
            return household_weight_value(year) / base_household_weight_index

    return {
        year: (
            volume_index[year]
            * price_parameter(year)
            / base_price
            / household_weight_relative(year)
        )
        for year in volume_index
    }


def _apply_road_fuel_litre_proxy_override(df: pd.DataFrame) -> pd.DataFrame:
    """Set petrol/diesel growth to litre-preserving spending-proxy indices.

    Historical and forecast litres come from HMRC clearances and OBR-implied
    clearances. Multiplying by model pump prices and dividing by the household
    weight index keeps weighted ``spending / price`` equal to those litre
    controls when datasets are uprated.
    """
    if not any(v in df.index for v in VOLUME_OVERRIDDEN_VARIABLES):
        return df

    for variable in VOLUME_OVERRIDDEN_VARIABLES:
        if variable not in df.index:
            continue
        household_weight_index = (
            df.loc["household_weight"] if "household_weight" in df.index else None
        )
        index = fuel_spending_litre_proxy_index(
            variable=variable,
            base_year=START_YEAR,
            end_year=END_YEAR,
            household_weight_index=household_weight_index,
        )
        missing_years = [
            year for year in range(START_YEAR, END_YEAR + 1) if year not in index
        ]
        if missing_years:
            raise ValueError(
                f"{variable} litre-proxy index missing years: "
                + ", ".join(str(year) for year in missing_years)
            )
        for year in range(START_YEAR, END_YEAR + 1):
            df.loc[variable, year] = round(index[year], UPRATING_TABLE_DECIMALS)
    return df


def _apply_road_fuel_volume_override(df: pd.DataFrame) -> pd.DataFrame:
    """Backward-compatible alias for the litre-preserving fuel override."""
    return _apply_road_fuel_litre_proxy_override(df)


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
