from policyengine_uk import Microsimulation
import pandas as pd
import numpy as np
import h5py
import torch
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
    def estimate_function(weights, matrix=None):
        # Use torch operations to preserve gradients
        if isinstance(weights, torch.Tensor):
            device = weights.device
        else:
            device = torch.device('cpu')
            weights = torch.tensor(weights, dtype=torch.float32, device=device)
            
        # Convert numpy arrays to torch tensors on the same device
        la_expanded_torch = torch.tensor(la_expanded, dtype=torch.float32, device=device)
        m_national_torch = torch.tensor(m_national.values.T, dtype=torch.float32, device=device)
            
        # weights shape: (360, num_households)
        # Local authority estimates: sum over households for each LA
        la_estimates = torch.sum(
            weights[:, :, None] * la_expanded_torch, dim=1
        ).flatten()

        # National estimates: sum all weights then multiply by national matrix
        national_weights = weights.sum(dim=0)
        
        # m_national is the matrix with households as rows and targets as columns
        # We need to transpose it to get targets as rows and households as columns for A @ w
        national_estimates = m_national_torch @ national_weights

        result = torch.cat([la_estimates, national_estimates])
        return result

    # Create combined targets
    la_targets = y.values.flatten()
    national_targets = y_national.values
    combined_targets = np.concatenate([la_targets, national_targets])
    
    # Create target names
    la_target_names = [f"{col}_{la}" for col in y.columns for la in range(count_local_authority)]
    national_target_names = list(y_national.index)
    combined_target_names = np.array(la_target_names + national_target_names)

    # Initialize weights with some noise
    initial_weights = (
        np.ones((count_local_authority, len(original_weights)))
        * original_weights
    )
    initial_weights *= 1 + np.random.random(initial_weights.shape) * 0.01

    # Apply region mask
    initial_weights = initial_weights * r

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
    final_weights = calibrator.weights * r

    # Save weights
    with h5py.File(STORAGE_FOLDER / "local_authority_weights.h5", "w") as f:
        f.create_dataset("2025", data=final_weights)

    dataset.household.household_weight = final_weights.sum(axis=0)

    return dataset


if __name__ == "__main__":
    calibrate()
