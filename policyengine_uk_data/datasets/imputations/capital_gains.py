import pandas as pd
import numpy as np
from policyengine_core.data import Dataset
from policyengine_uk_data.utils.stack import stack_datasets

# Fit a spline to each income band's percentiles
from scipy.interpolate import UnivariateSpline

from policyengine_uk_data.storage import STORAGE_FOLDER
from tqdm import tqdm
import copy

import torch
from torch.optim import Adam
from tqdm import tqdm
from policyengine_uk.data import UKSingleYearDataset
import logging
from policyengine_uk_data.utils.subsample import subsample_dataset

capital_gains = pd.read_csv(
    STORAGE_FOLDER / "capital_gains_distribution_advani_summers.csv.gz"
)
capital_gains["maximum_total_income"] = (
    capital_gains.minimum_total_income.shift(-1).fillna(np.inf)
)

# Silence verbose logging
logging.getLogger("root").setLevel(logging.WARNING)


def impute_cg_to_doubled_dataset(
    dataset: UKSingleYearDataset,
) -> tuple[np.ndarray, np.ndarray]:
    """Assumes that the capital gains distribution is the same for all years."""

    from policyengine_uk import Microsimulation
    from policyengine_uk.system import system

    sim = Microsimulation(dataset=dataset)
    ti = sim.calculate("total_income").values
    household_weight = sim.calculate("household_weight").values
    first_half = (
        np.concatenate(
            [
                np.ones(len(household_weight) // 2),
                np.zeros(len(household_weight) // 2),
            ]
        )
        > 0
    )
    # Give capital gains to one adult aged 15+ in each household
    adult_index = sim.calculate("adult_index").values == 1
    in_person_second_half = np.zeros(len(ti)) > 0
    in_person_second_half[len(ti) // 2 :] = True
    has_cg = np.zeros(len(ti)) > 0
    has_cg[adult_index & in_person_second_half] = True
    blend_factor = torch.tensor(
        np.zeros(first_half.sum()), requires_grad=True, dtype=torch.float32
    )
    household_weight = torch.tensor(household_weight, dtype=torch.float32)
    sigmoid = torch.nn.Sigmoid()

    def loss(blend_factor):
        loss = 0

        blended_household_weight = household_weight.clone()
        adjusted_blend_factor = sigmoid(blend_factor)
        blended_household_weight[first_half] = (
            adjusted_blend_factor * blended_household_weight[first_half]
        )
        blended_household_weight[~first_half] = (
            1 - adjusted_blend_factor
        ) * blended_household_weight[first_half]
        for i in range(len(capital_gains)):
            lower = capital_gains.minimum_total_income.iloc[i]
            upper = capital_gains.maximum_total_income.iloc[i]
            true_pct_with_gains = capital_gains.percent_with_gains.iloc[i]

            ti_in_range = (ti >= lower) * (ti < upper)
            cg_in_income_range = has_cg * ti_in_range
            household_ti_in_range_count = torch.tensor(
                sim.map_result(ti_in_range, "person", "household", how="sum")
            )
            household_cg_in_income_range_count = torch.tensor(
                sim.map_result(
                    cg_in_income_range, "person", "household", how="sum"
                )
            )
            pred_ti_in_range = (
                blended_household_weight * household_ti_in_range_count
            ).sum()
            pred_cg_in_income_range = (
                blended_household_weight * household_cg_in_income_range_count
            ).sum()
            pred_pct_with_gains = pred_cg_in_income_range / torch.clip(
                pred_ti_in_range, 1
            )
            loss += (pred_pct_with_gains - true_pct_with_gains) ** 2

        return loss

    optimiser = Adam([blend_factor], lr=1e-1)
    progress = range(100)
    logging.info("Splitting household weights into has-gains and no-gains")
    for i in progress:
        optimiser.zero_grad()
        loss_value = loss(blend_factor)
        loss_value.backward()
        optimiser.step()
        if loss_value.item() < 1e-3:
            break

    new_household_weight = household_weight.detach().numpy()
    original_household_weight = new_household_weight.copy()
    blend_factor = sigmoid(blend_factor).detach().numpy()
    new_household_weight[first_half] = (
        blend_factor * original_household_weight[first_half]
    )
    new_household_weight[~first_half] = (
        1 - blend_factor
    ) * original_household_weight[first_half]

    # Impute actual capital gains amounts given gains
    new_cg = np.zeros(len(ti))

    logging.info("Imputing capital gains among those with gains")

    # Use seeded generator for reproducibility
    generator = np.random.default_rng(seed=100)

    for i in range(len(capital_gains)):
        row = capital_gains.iloc[i]
        spline = UnivariateSpline(
            [0.05, 0.1, 0.25, 0.5, 0.75, 0.90, 0.95],
            [row.p05, row.p10, row.p25, row.p50, row.p75, row.p90, row.p95],
            k=1,
        )
        lower = row.minimum_total_income
        upper = row.maximum_total_income
        ti_in_range = (ti >= lower) * (ti < upper)
        in_target_range = has_cg * ti_in_range > 0
        quantiles = generator.random(int(in_target_range.sum()))
        pred_capital_gains = spline(quantiles)
        new_cg[in_target_range] = pred_capital_gains

    return new_cg, new_household_weight


def impute_capital_gains(dataset: UKSingleYearDataset) -> UKSingleYearDataset:
    zero_weight_copy = dataset.copy()
    zero_weight_copy.household.household_weight = 1
    data = stack_datasets(
        dataset,
        zero_weight_copy,
    )

    pred_cg, household_weight = impute_cg_to_doubled_dataset(data)

    data.person["capital_gains"] = pred_cg
    data.household["household_weight"] = household_weight

    data.validate()
    return data
