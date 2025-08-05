from policyengine_uk.data import UKSingleYearDataset
import numpy as np


def subsample_dataset(
    dataset: UKSingleYearDataset,
    sample_size: int,
    seed: int = 42,
):
    """
    Subsample a UKSingleYearDataset to a specified sample size.

    Parameters:
        dataset (UKSingleYearDataset): The dataset to subsample.
        sample_size (int): The number of samples to retain.
        seed (int): Random seed for reproducibility.

    Returns:
        UKSingleYearDataset: A new dataset with the specified sample size.
    """
    np.random.seed(seed)
    household_ids = np.random.choice(
        dataset.household.household_id.values,
        size=sample_size,
        replace=False,
    )
    person_filter = dataset.person.person_household_id.isin(household_ids)
    benunit_ids = dataset.person.person_benunit_id[
        dataset.person.person_household_id.isin(household_ids)
    ]
    benunit_filter = dataset.benunit.benunit_id.isin(benunit_ids)
    household_filter = dataset.household.household_id.isin(household_ids)

    subsampled_dataset = UKSingleYearDataset(
        person=dataset.person[person_filter],
        benunit=dataset.benunit[benunit_filter],
        household=dataset.household[household_filter],
        fiscal_year=dataset.time_period,
    )

    return subsampled_dataset
