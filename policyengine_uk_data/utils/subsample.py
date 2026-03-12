"""
Dataset subsampling utilities for PolicyEngine UK.

This module provides functions to create smaller samples from full datasets,
useful for testing and development workflows.
"""

from policyengine_uk.data import UKSingleYearDataset
import numpy as np


def subsample_dataset(
    dataset: UKSingleYearDataset,
    sample_size: int,
    seed: int = 42,
):
    """
    Subsample a UKSingleYearDataset to a specified number of households.

    Households are sampled with probability proportional to their weight,
    and weights are rescaled so the subsampled dataset preserves population
    totals.

    Parameters:
        dataset (UKSingleYearDataset): The dataset to subsample.
        sample_size (int): The number of households to retain.
        seed (int): Random seed for reproducibility.

    Returns:
        UKSingleYearDataset: A new dataset with the specified sample size.
    """
    rng = np.random.default_rng(seed)
    household_df = dataset.household
    weights = household_df.household_weight.values.astype(float)
    total_weight = np.nansum(weights)

    # Sample proportional to weight when weights are available,
    # otherwise fall back to uniform sampling
    if total_weight > 0 and not np.any(np.isnan(weights)):
        probs = weights / total_weight
    else:
        probs = None

    indices = rng.choice(
        len(household_df),
        size=sample_size,
        replace=False,
        p=probs,
    )
    household_ids = household_df.household_id.values[indices]

    person_filter = dataset.person.person_household_id.isin(household_ids)
    benunit_ids = dataset.person.person_benunit_id[person_filter]
    benunit_filter = dataset.benunit.benunit_id.isin(benunit_ids)
    household_filter = dataset.household.household_id.isin(household_ids)

    # Rescale weights so the subsample preserves the original population total
    sub_household = dataset.household[household_filter].copy()
    sub_weight_sum = sub_household.household_weight.sum()
    if total_weight > 0 and sub_weight_sum > 0:
        scale = total_weight / sub_weight_sum
        sub_household["household_weight"] = sub_household.household_weight * scale

    subsampled_dataset = UKSingleYearDataset(
        person=dataset.person[person_filter].reset_index(drop=True),
        benunit=dataset.benunit[benunit_filter].reset_index(drop=True),
        household=sub_household.reset_index(drop=True),
        fiscal_year=dataset.time_period,
    )

    return subsampled_dataset
