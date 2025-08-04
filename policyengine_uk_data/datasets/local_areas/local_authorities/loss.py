import torch
from policyengine_uk import Microsimulation
import pandas as pd
import numpy as np
from pathlib import Path

from policyengine_uk_data.utils.loss import (
    create_target_matrix as create_national_target_matrix,
)
from policyengine_uk_data.storage import STORAGE_FOLDER
from policyengine_uk.data import UKSingleYearDataset

FOLDER = Path(__file__).parent


def create_local_authority_target_matrix(
    dataset: UKSingleYearDataset,
    time_period: int = None,
    reform=None,
    uprate: bool = True,
):
    if time_period is None:
        time_period = dataset.time_period
    ages = pd.read_csv(FOLDER / "targets" / "age.csv")
    incomes = pd.read_csv(FOLDER / "targets" / "spi_by_la.csv")
    employment_incomes = pd.read_csv(
        FOLDER / "targets" / "employment_income.csv"
    )
    la_codes = pd.read_csv(STORAGE_FOLDER / "local_authorities_2021.csv")

    sim = Microsimulation(dataset=dataset, reform=reform)
    sim.default_calculation_period = time_period

    matrix = pd.DataFrame()
    y = pd.DataFrame()

    INCOME_VARIABLES = [
        "self_employment_income",
        "employment_income",
    ]

    national_incomes = pd.read_csv(STORAGE_FOLDER / "incomes_projection.csv")
    national_incomes = national_incomes[national_incomes.year == 2025]

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
    national_demographics = pd.read_csv(STORAGE_FOLDER / "demographics.csv")
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
        y[f"age/{age_str}"] *= uk_total_population / targets_total_pop

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

    country_mask = create_country_mask(
        household_countries=sim.calculate("country").values,
        codes=la_codes.code,
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
