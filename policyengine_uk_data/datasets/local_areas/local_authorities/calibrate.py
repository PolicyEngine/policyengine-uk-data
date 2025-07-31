from policyengine_uk import Microsimulation
import pandas as pd
import numpy as np
import h5py
from microcalibrate.calibration import Calibration
from policyengine_uk_data.storage import STORAGE_FOLDER

from policyengine_uk_data.datasets.local_areas.local_authorities.loss import (
    create_local_authority_target_matrix,
    create_national_target_matrix,
)
from policyengine_uk.data import UKSingleYearDataset


def calibrate(
    dataset: UKSingleYearDataset,
    epochs: int = 528,
    verbose: bool = False,
):
    dataset = dataset.copy()
    matrix, y, r = create_local_authority_target_matrix(
        dataset, dataset.time_period
    )

    m_national, y_national = create_national_target_matrix(
        dataset, dataset.time_period
    )

    sim = Microsimulation(dataset=dataset)
    sim.default_calculation_period = dataset.time_period

    count_local_authority = 360

    # Initial weights
    original_weights = (
        sim.calculate("household_weight").values / count_local_authority
    )

    # Create combined estimate matrix and targets
    # Combine local authority and national matrices
    la_expanded = np.repeat(
        matrix.values, count_local_authority, axis=0
    ).reshape(count_local_authority, matrix.shape[0], matrix.shape[1])
    national_expanded = np.tile(m_national.values, (count_local_authority, 1))

    # Create estimate function that combines local authority and national estimates
    def estimate_function(weights):
        # weights shape: (360, num_households)
        # Local authority estimates: sum over households for each LA
        la_estimates = np.sum(
            weights[:, :, np.newaxis] * la_expanded, axis=1
        ).flatten()

        # National estimates: sum all weights then multiply by national matrix
        national_weights = weights.sum(axis=0)
        national_estimates = national_expanded[0] @ national_weights

        return np.concatenate([la_estimates, national_estimates])

    # Create combined targets
    la_targets = y.values.flatten()
    national_targets = y_national.values
    combined_targets = np.concatenate([la_targets, national_targets])

    # Initialize weights with some noise
    initial_weights = (
        np.ones((count_local_authority, len(original_weights)))
        * original_weights
    )
    initial_weights *= 1 + np.random.random(initial_weights.shape) * 0.01

    # Apply region mask
    initial_weights = initial_weights * r

    calibrator = Calibration(
        estimate_function=estimate_function,
        weights=initial_weights,
        targets=combined_targets,
        noise_level=0.05,
        epochs=epochs,
        learning_rate=0.01,
        dropout_rate=0.05,
    )

    performance_df = calibrator.calibrate()

    # Get final weights
    final_weights = calibrator.weights * r

    # Save weights
    with h5py.File(STORAGE_FOLDER / "local_authority_weights.h5", "w") as f:
        f.create_dataset("2025", data=final_weights)

    dataset.household.household_weight = final_weights.sum(axis=0)

    return dataset


if __name__ == "__main__":
    calibrate()
