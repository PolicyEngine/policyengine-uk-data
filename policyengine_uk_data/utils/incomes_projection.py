import pandas as pd
from policyengine_uk.data import UKSingleYearDataset

from policyengine_uk_data.datasets.spi import (
    SPI_FISCAL_YEAR,
    SPI_H5_FILENAME,
    SPI_RELEASE_NAME,
    SPI_TAB_FILENAME,
    create_spi,
)
from policyengine_uk_data.storage import STORAGE_FOLDER
from policyengine_uk_data.targets.sources.hmrc_spi import (
    SPI_INCOME_TABLE_VARIABLES,
    _SPI_YEAR,
    get_income_band_table,
)
from policyengine_uk_data.utils.uprating import uprate_values


SPI_DATASET = str(STORAGE_FOLDER / SPI_H5_FILENAME)
BASE_YEAR = _SPI_YEAR
MAX_YEAR = 2029

# The projection artifact keeps savings interest for diagnostics and future
# consumers, even though current target creation excludes it.
ALL_INCOME_VARIABLES = SPI_INCOME_TABLE_VARIABLES


def _read_spi_dataset_year(dataset_path) -> int:
    with pd.HDFStore(dataset_path, mode="r") as store:
        return int(store["time_period"].iloc[0])


def ensure_spi_dataset() -> str:
    """Create the SPI H5 projection input from the current TAB release if needed.

    Kept for workflows that need the private SPI microdata directly. The
    projection CSV is now generated from official aggregate ODS tables instead
    of a reweighted SPI microdataset.
    """
    dataset_path = STORAGE_FOLDER / SPI_H5_FILENAME
    if (
        dataset_path.exists()
        and _read_spi_dataset_year(dataset_path) == SPI_FISCAL_YEAR
    ):
        return str(dataset_path)

    tab_path = STORAGE_FOLDER / SPI_RELEASE_NAME / SPI_TAB_FILENAME
    if not tab_path.exists():
        raise FileNotFoundError(
            f"Missing SPI TAB file for projections: {tab_path}. "
            "Run make download before refreshing income projections."
        )

    create_spi(tab_path, SPI_FISCAL_YEAR).save(dataset_path)
    dataset_year = _read_spi_dataset_year(dataset_path)
    if dataset_year != SPI_FISCAL_YEAR:
        raise ValueError(
            f"Built SPI dataset {dataset_path} for {dataset_year}, "
            f"expected {SPI_FISCAL_YEAR}."
        )
    return str(dataset_path)


def load_spi_dataset() -> UKSingleYearDataset:
    dataset = UKSingleYearDataset(ensure_spi_dataset())
    dataset.household["region"] = dataset.household["region"].replace(
        {"UNKNOWN": "SOUTH_EAST"}
    )
    return dataset


def load_base_income_table() -> pd.DataFrame:
    """Load the current official SPI income-band table.

    The row with lower bound 12,570 and upper bound infinity is the aggregate
    above-personal-allowance row used by local-area income scaling.
    """
    return get_income_band_table(include_aggregate=True)


def project_income_table(
    base_income_table: pd.DataFrame,
    *,
    base_year: int = BASE_YEAR,
    max_year: int = MAX_YEAR,
) -> pd.DataFrame:
    """Project SPI income-band rows from the official base year.

    Counts follow the household-weight/population index. Amounts follow the
    PolicyEngine uprating index for each income variable. This intentionally
    preserves the official base-year band distribution instead of reweighting
    older SPI microdata to a newer aggregate target set.
    """
    projected = []
    bounds = base_income_table[["total_income_lower_bound", "total_income_upper_bound"]]

    for year in range(base_year, max_year + 1):
        year_df = bounds.copy()
        for variable in ALL_INCOME_VARIABLES:
            count_col = f"{variable}_count"
            amount_col = f"{variable}_amount"

            if count_col in base_income_table:
                year_df[count_col] = (
                    uprate_values(
                        base_income_table[count_col],
                        "household_weight",
                        base_year,
                        year,
                    )
                    .round()
                    .astype(int)
                )
            if amount_col in base_income_table:
                year_df[amount_col] = (
                    uprate_values(
                        base_income_table[amount_col],
                        variable,
                        base_year,
                        year,
                    )
                    .round()
                    .astype(int)
                )
        year_df["year"] = year
        projected.append(year_df)

    return pd.concat(projected, ignore_index=True)


def create_income_projections() -> pd.DataFrame:
    base_income_table = load_base_income_table()
    projection_df = project_income_table(base_income_table)
    projection_df.to_csv(STORAGE_FOLDER / "incomes_projection.csv", index=False)
    base_income_table.to_csv(STORAGE_FOLDER / "incomes.csv", index=False)
    return projection_df


if __name__ == "__main__":
    create_income_projections()
