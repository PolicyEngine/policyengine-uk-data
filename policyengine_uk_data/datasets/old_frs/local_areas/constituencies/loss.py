import torch
from policyengine_uk import Microsimulation
import pandas as pd
import numpy as np

# Fill in missing constituencies with average column values
import pandas as pd
import numpy as np
from pathlib import Path

from policyengine_uk_data.utils.loss import (
    create_target_matrix as create_national_target_matrix,
)
from policyengine_uk_data.storage import STORAGE_FOLDER
from policyengine_uk_data.datasets.old_frs.local_areas.constituencies.boundary_changes.mapping_matrix import (
    mapping_matrix,
)

FOLDER = Path(__file__).parent


def create_constituency_target_matrix(
    dataset: str = "enhanced_frs_2022_23",
    time_period: int = 2025,
    reform=None,
    uprate: bool = True,
):
    ages = pd.read_csv(FOLDER / "targets" / "age.csv")
    incomes = pd.read_csv(FOLDER / "targets" / "spi_by_constituency.csv")
    employment_incomes = pd.read_csv(
        FOLDER / "targets" / "employment_income.csv"
    )

    sim = Microsimulation(dataset=dataset, reform=reform)
    sim.default_calculation_period = time_period

    national_incomes = pd.read_csv(STORAGE_FOLDER / "incomes_projection.csv")
    national_incomes = national_incomes[national_incomes.year == 2025]

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

    age = sim.calculate("age").values
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

    employment_income = sim.calculate("employment_income").values
    bounds = list(
        employment_incomes.employment_income_lower_bound.sort_values().unique()
    ) + [np.inf]

    for lower_bound, upper_bound in zip(bounds[:-1], bounds[1:]):
        if (
            lower_bound <= 15_000
        ):  # Skip some targets with very small sample sizes
            continue
        if upper_bound >= 200_000:
            continue

        national_data_row = national_incomes[
            national_incomes.total_income_lower_bound == lower_bound
        ]["employment_income_amount"].iloc[0]

        count_target = employment_incomes[
            (employment_incomes.employment_income_lower_bound == lower_bound)
            & (employment_incomes.employment_income_upper_bound == upper_bound)
        ].employment_income_count.values

        amount_target = employment_incomes[
            (employment_incomes.employment_income_lower_bound == lower_bound)
            & (employment_incomes.employment_income_upper_bound == upper_bound)
        ].employment_income_amount.values

        sum_of_local_area_values = amount_target.sum()

        adjustment = national_data_row / sum_of_local_area_values

        if count_target.mean() < 200:
            print(
                f"Skipping employment income band {lower_bound} to {upper_bound} due to low count target mean: {count_target.mean()}"
            )
            continue

        if amount_target.mean() < 200 * 30e3:
            print(
                f"Skipping employment income band {lower_bound} to {upper_bound} due to low amount target mean: {amount_target.mean()}"
            )
            continue

        in_bound = (
            (employment_income >= lower_bound)
            & (employment_income < upper_bound)
            & (employment_income != 0)
            & (age >= 16)
        )
        band_str = f"{lower_bound}_{upper_bound}"
        matrix[f"hmrc/employment_income/amount/{band_str}"] = sim.map_result(
            employment_income * in_bound, "person", "household"
        )
        y[f"hmrc/employment_income/amount/{band_str}"] = (
            amount_target * adjustment
        )

    if uprate:
        y = uprate_targets(y, time_period)

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


def uprate_targets(y: pd.DataFrame, target_year: int = 2025) -> pd.DataFrame:
    # Uprate age targets from 2020, taxable income targets from 2021, employment income targets from 2023.
    # Use PolicyEngine uprating factors.
    from policyengine_uk_data.datasets.old_frs.frs import FRS_2020_21

    sim = Microsimulation(dataset=FRS_2020_21)
    matrix_20, y_20, _ = create_constituency_target_matrix(
        FRS_2020_21, 2020, uprate=False
    )
    matrix_21, y_21, _ = create_constituency_target_matrix(
        FRS_2020_21, 2021, uprate=False
    )
    matrix_23, y_23, _ = create_constituency_target_matrix(
        FRS_2020_21, 2023, uprate=False
    )
    matrix_final, y_final, _ = create_constituency_target_matrix(
        FRS_2020_21, target_year, uprate=False
    )

    weights_20 = sim.calculate("household_weight", 2020)
    weights_21 = sim.calculate("household_weight", 2021)
    weights_23 = sim.calculate("household_weight", 2023)
    weights_final = sim.calculate("household_weight", target_year)

    rel_change_20_final = (weights_final @ matrix_final) / (
        weights_20 @ matrix_20
    ) - 1
    is_uprated_from_2020 = [
        col.startswith("age/") for col in matrix_20.columns
    ]
    uprating_from_2020 = np.zeros_like(matrix_20.columns, dtype=float)
    uprating_from_2020[is_uprated_from_2020] = rel_change_20_final[
        is_uprated_from_2020
    ]

    uprating = uprating_from_2020
    y = y * (1 + uprating)

    return y
