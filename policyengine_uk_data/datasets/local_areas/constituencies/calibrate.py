from policyengine_uk import Microsimulation
import pandas as pd
import numpy as np
import h5py
import torch
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
    epochs: int = 128,
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
    # Don't expand national matrix - we only need the original for calculations

    # Create estimate function that combines constituency and national estimates
    def estimate_function(weights, matrix=None):
        # Use torch operations to preserve gradients
        if isinstance(weights, torch.Tensor):
            device = weights.device
        else:
            device = torch.device("cpu")
            weights = torch.tensor(weights, dtype=torch.float32, device=device)

        # Convert numpy arrays to torch tensors on the same device
        constituency_expanded_torch = torch.tensor(
            constituency_expanded, dtype=torch.float32, device=device
        )
        m_national_torch = torch.tensor(
            m_national_.values.T, dtype=torch.float32, device=device
        )

        # weights shape: (650, num_households)
        # Constituency estimates: sum over households for each constituency
        constituency_estimates = torch.sum(
            weights[:, :, None] * constituency_expanded_torch, dim=1
        ).flatten()

        # National estimates: sum all weights then multiply by national matrix
        national_weights = weights.sum(dim=0)

        # m_national_ is the matrix with households as rows and targets as columns
        # We need to transpose it to get targets as rows and households as columns for A @ w
        national_estimates = m_national_torch @ national_weights

        result = torch.cat([constituency_estimates, national_estimates])
        return result

    # Create combined targets
    constituency_targets = y_.values.flatten()
    national_targets = y_national_.values
    combined_targets = np.concatenate([constituency_targets, national_targets])

    # Create target names
    constituency_target_names = [
        f"{col}_{const}"
        for col in y_.columns
        for const in range(COUNT_CONSTITUENCIES)
    ]
    national_target_names = list(y_national_.index)
    combined_target_names = np.array(
        constituency_target_names + national_target_names
    )

    # Initialize weights with some noise
    initial_weights = (
        np.ones((COUNT_CONSTITUENCIES, len(original_weights)))
        * original_weights
    )
    initial_weights *= 1 + np.random.random(initial_weights.shape) * 0.01

    # Apply country mask
    initial_weights = initial_weights * country_mask

    calibrator = Calibration(
        weights=initial_weights,
        targets=combined_targets,
        target_names=combined_target_names,
        estimate_function=estimate_function,
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
