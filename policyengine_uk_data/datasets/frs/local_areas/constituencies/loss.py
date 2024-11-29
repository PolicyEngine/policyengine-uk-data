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

FOLDER = Path(__file__).parent


def create_constituency_target_matrix(
    dataset: str = "enhanced_frs_2022_23", time_period: int = 2025, reform=None
):
    ages = pd.read_csv(FOLDER / "targets" / "age.csv")
    incomes = pd.read_csv(FOLDER / "targets" / "total_income.csv")
    employment_incomes = pd.read_csv(
        FOLDER / "targets" / "employment_income.csv"
    )

    sim = Microsimulation(dataset=dataset, reform=reform)
    sim.default_calculation_period = time_period

    matrix = pd.DataFrame()
    y = pd.DataFrame()

    total_income = sim.calculate("total_income").values
    matrix["hmrc/total_income/amount"] = sim.map_result(
        total_income, "person", "household"
    )
    y["hmrc/total_income/amount"] = incomes["total_income_amount"]

    matrix["hmrc/total_income/count"] = sim.map_result(
        total_income != 0, "person", "household"
    )
    y["hmrc/total_income/count"] = incomes["total_income_count"]

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
        if lower_bound >= 70_000 or lower_bound < 12_570:
            continue
        in_bound = (
            (employment_income >= lower_bound)
            & (employment_income < upper_bound)
            & (employment_income != 0)
            & (age >= 16)
        )
        band_str = f"{lower_bound}_{upper_bound}"
        matrix[f"hmrc/employment_income/count/{band_str}"] = sim.map_result(
            in_bound, "person", "household"
        )
        y[f"hmrc/employment_income/count/{band_str}"] = employment_incomes[
            (employment_incomes.employment_income_lower_bound == lower_bound)
            & (employment_incomes.employment_income_upper_bound == upper_bound)
        ].employment_income_count.values

        matrix[f"hmrc/employment_income/amount/{band_str}"] = sim.map_result(
            employment_income * in_bound, "person", "household"
        )
        y[f"hmrc/employment_income/amount/{band_str}"] = employment_incomes[
            (employment_incomes.employment_income_lower_bound == lower_bound)
            & (employment_incomes.employment_income_upper_bound == upper_bound)
        ].employment_income_amount.values

    return matrix, y
