from policyengine_uk import Microsimulation
import pandas as pd
import numpy as np
import h5py
from microcalibrate.calibration import Calibration

from policyengine_uk_data.datasets.local_areas.constituencies.loss import (
    create_constituency_target_matrix,
    create_national_target_matrix,
)
from policyengine_uk_data.datasets.local_areas.constituencies.boundary_changes.mapping_matrix import (
    mapping_matrix,
)
from pathlib import Path
from policyengine_uk_data.storage import STORAGE_FOLDER
from policyengine_uk.data import UKSingleYearDataset

FOLDER = Path(__file__).parent


def calibrate(
    dataset: UKSingleYearDataset,
    epochs: int = 528,
    excluded_training_targets=[],
    log_csv="calibration_log.csv",
    verbose: bool = False,
):
    dataset = dataset.copy()
    matrix_, y_, country_mask = create_constituency_target_matrix(dataset)
    m_national_, y_national_ = create_national_target_matrix(dataset)

    sim = Microsimulation(dataset=dataset)
    sim.default_calculation_period = dataset.time_period

    COUNT_CONSTITUENCIES = 650

    # Initial weights
    original_weights = (
        sim.calculate("household_weight").values / COUNT_CONSTITUENCIES
    )

    # Create combined estimate matrix and targets
    # Combine constituency and national matrices
    constituency_expanded = np.repeat(
        matrix_.values, COUNT_CONSTITUENCIES, axis=0
    ).reshape(COUNT_CONSTITUENCIES, matrix_.shape[0], matrix_.shape[1])
    national_expanded = np.tile(m_national_.values, (COUNT_CONSTITUENCIES, 1))

    # Create estimate function that combines constituency and national estimates
    def estimate_function(weights):
        # weights shape: (650, num_households)
        # Constituency estimates: sum over households for each constituency
        constituency_estimates = np.sum(
            weights[:, :, np.newaxis] * constituency_expanded, axis=1
        ).flatten()

        # National estimates: sum all weights then multiply by national matrix
        national_weights = weights.sum(axis=0)
        national_estimates = national_expanded[0] @ national_weights

        return np.concatenate([constituency_estimates, national_estimates])

    # Create combined targets
    constituency_targets = y_.values.flatten()
    national_targets = y_national_.values
    combined_targets = np.concatenate([constituency_targets, national_targets])

    # Initialize weights with some noise
    initial_weights = (
        np.ones((COUNT_CONSTITUENCIES, len(original_weights)))
        * original_weights
    )
    initial_weights *= 1 + np.random.random(initial_weights.shape) * 0.01

    # Apply country mask
    initial_weights = initial_weights * country_mask

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
    final_weights = calibrator.weights * country_mask

    # Save weights
    with h5py.File(
        STORAGE_FOLDER / "parliamentary_constituency_weights.h5", "w"
    ) as f:
        f.create_dataset("2025", data=final_weights)

    dataset.household.household_weight = final_weights.sum(axis=0)

    # Convert performance to match original format if log_csv is specified
    if log_csv:
        performance_df.to_csv(log_csv, index=False)

    return dataset


if __name__ == "__main__":
    calibrate()
