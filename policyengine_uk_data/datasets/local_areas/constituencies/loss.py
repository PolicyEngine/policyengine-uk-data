from policyengine_uk import Microsimulation
import pandas as pd
import numpy as np
from pathlib import Path

from policyengine_uk_data.utils.loss import (
    create_target_matrix as create_national_target_matrix,
)
from policyengine_uk_data.storage import STORAGE_FOLDER
from policyengine_uk_data.datasets.local_areas.constituencies.boundary_changes.mapping_matrix import (
    mapping_matrix,
)
from policyengine_uk.data import UKSingleYearDataset
from policyengine_uk_data.utils.uc_data import uc_pc_households

FOLDER = Path(__file__).parent


def create_constituency_target_matrix(
    dataset: UKSingleYearDataset,
    time_period: int = None,
    reform=None,
):
    if time_period is None:
        time_period = dataset.time_period
    ages = pd.read_csv(FOLDER / "targets" / "age.csv")
    national_demographics = pd.read_csv(STORAGE_FOLDER / "demographics.csv")
    incomes = pd.read_csv(FOLDER / "targets" / "spi_by_constituency.csv")

    sim = Microsimulation(dataset=dataset, reform=reform)
    sim.default_calculation_period = dataset.time_period

    national_incomes = pd.read_csv(STORAGE_FOLDER / "incomes_projection.csv")
    national_incomes = national_incomes[
        national_incomes.year
        == max(national_incomes.year.min(), int(dataset.time_period))
    ]

    matrix = pd.DataFrame()
    y = pd.DataFrame()

    INCOME_VARIABLES = [
        "self_employment_income",
        "employment_income",
    ]

    for income_variable in INCOME_VARIABLES:
        income_values = sim.calculate(income_variable).values
        in_spi_frame = sim.calculate("income_tax").values > 0
        matrix[f"hmrc/{income_variable}/amount"] = sim.map_result(
            income_values * in_spi_frame, "person", "household"
        )
        local_targets = incomes[f"{income_variable}_amount"].values
        local_target_sum = local_targets.sum()
        national_target = national_incomes[
            (national_incomes.total_income_lower_bound == 12_570)
            & (national_incomes.total_income_upper_bound == np.inf)
        ][income_variable + "_amount"].iloc[0]
        national_consistency_adjustment_factor = (
            national_target / local_target_sum
        )
        y[f"hmrc/{income_variable}/amount"] = (
            local_targets * national_consistency_adjustment_factor
        )
        matrix[f"hmrc/{income_variable}/count"] = sim.map_result(
            (income_values != 0) * in_spi_frame, "person", "household"
        )
        local_targets = incomes[f"{income_variable}_count"].values
        local_target_sum = local_targets.sum()
        national_target = national_incomes[
            (national_incomes.total_income_lower_bound == 12_570)
            & (national_incomes.total_income_upper_bound == np.inf)
        ][income_variable + "_count"].iloc[0]
        y[f"hmrc/{income_variable}/count"] = (
            incomes[f"{income_variable}_count"].values
            * national_consistency_adjustment_factor
        )

    uk_total_population = (
        national_demographics[national_demographics.name == "uk_population"][
            str(time_period)
        ].values[0]
        * 1e6
    )

    age = sim.calculate("age").values
    targets_total_pop = 0
    for lower_age in range(0, 80, 10):
        upper_age = lower_age + 10

        in_age_band = (age >= lower_age) & (age < upper_age)

        age_str = f"{lower_age}_{upper_age}"
        matrix[f"age/{age_str}"] = sim.map_result(
            in_age_band, "person", "household"
        )

        age_count = ages[
            [str(age) for age in range(lower_age, upper_age)]
        ].sum(axis=1)

        age_str = f"{lower_age}_{upper_age}"
        y[f"age/{age_str}"] = age_count.values
        targets_total_pop += age_count.values.sum()

    # Adjust for consistency
    for lower_age in range(0, 80, 10):
        upper_age = lower_age + 10

        in_age_band = (age >= lower_age) & (age < upper_age)

        age_str = f"{lower_age}_{upper_age}"
        y[f"age/{age_str}"] *= uk_total_population / targets_total_pop * 0.9

    # UC household count by constituency
    y["uc_households"] = uc_pc_households.household_count.values
    matrix["uc_households"] = sim.map_result(
        (sim.calculate("universal_credit").values > 0).astype(int),
        "benunit",
        "household",
    )

    const_2024 = pd.read_csv(STORAGE_FOLDER / "constituencies_2024.csv")
    const_2010 = pd.read_csv(STORAGE_FOLDER / "constituencies_2010.csv")

    y_2010 = y.copy()
    y_2010["name"] = const_2010["name"].values

    y_columns = list(y.columns)
    y_values = mapping_matrix @ y.values  # Transform to 2024 constituencies

    y = pd.DataFrame(y_values, columns=y_columns)

    y_2024 = y.copy()
    y_2024["name"] = const_2024["name"].values

    country_mask = create_country_mask(
        household_countries=sim.calculate("country").values,
        codes=const_2024.code,
    )
    return matrix, y, country_mask


def create_country_mask(
    household_countries: np.ndarray, codes: pd.Series
) -> np.ndarray:
    # Create a matrix R to accompany the loss matrix M s.t. (W x M) x R = Y_
    # where Y_ is the target matrix for the country where no target is constructed from weights from a different country.

    constituency_countries = codes.apply(lambda code: code[0]).map(
        {
            "E": "ENGLAND",
            "W": "WALES",
            "S": "SCOTLAND",
            "N": "NORTHERN_IRELAND",
        }
    )

    r = np.zeros((len(codes), len(household_countries)))

    for i in range(len(codes)):
        r[i] = household_countries == constituency_countries[i]

    return r
