import torch
from policyengine_uk import Microsimulation
import pandas as pd
import numpy as np
import h5py
from policyengine_uk_data.storage import STORAGE_FOLDER
from policyengine_uk.data import UKSingleYearDataset


def calibrate_local_areas(
    dataset: UKSingleYearDataset,
    matrix_fn,
    national_matrix_fn,
    area_count: int,
    weight_file: str,
    dataset_key: str = "2025",
    epochs: int = 128,
    excluded_training_targets=[],
    log_csv=None,
    verbose: bool = False,
    area_name: str = "area",
    get_performance=None,
):
    """
    Generic calibration function for local areas (constituencies, local authorities, etc.)

    Args:
        dataset: The dataset to calibrate
        matrix_fn: Function that returns (matrix, targets, mask) for the local areas
        national_matrix_fn: Function that returns (matrix, targets) for national totals
        area_count: Number of areas (e.g., 650 for constituencies, 360 for local authorities)
        weight_file: Name of the h5 file to save weights to
        dataset_key: Key to use in the h5 file
        epochs: Number of training epochs
        excluded_training_targets: List of targets to exclude from training (for validation)
        log_csv: CSV file to log performance to
        verbose: Whether to print progress
        area_name: Name of the area type for logging
    """
    dataset = dataset.copy()
    matrix, y, r = matrix_fn(dataset)
    m_c, y_c = matrix.copy(), y.copy()
    m_national, y_national = national_matrix_fn(dataset)
    m_n, y_n = m_national.copy(), y_national.copy()

    # Weights - area_count x num_households
    original_weights = np.log(
        dataset.household.household_weight.values / area_count
        + np.random.random(len(dataset.household.household_weight.values))
        * 0.01
    )
    weights = torch.tensor(
        np.ones((area_count, len(original_weights))) * original_weights,
        dtype=torch.float32,
        requires_grad=True,
    )

    # Set up validation targets if specified
    validation_targets_local = (
        matrix.columns.isin(excluded_training_targets)
        if hasattr(matrix, "columns")
        else None
    )
    validation_targets_national = (
        m_national.columns.isin(excluded_training_targets)
        if hasattr(m_national, "columns")
        else None
    )
    dropout_targets = len(excluded_training_targets) > 0

    # Convert to tensors
    metrics = torch.tensor(
        matrix.values if hasattr(matrix, "values") else matrix,
        dtype=torch.float32,
    )
    y = torch.tensor(
        y.values if hasattr(y, "values") else y, dtype=torch.float32
    )
    matrix_national = torch.tensor(
        m_national.values if hasattr(m_national, "values") else m_national,
        dtype=torch.float32,
    )
    y_national = torch.tensor(
        y_national.values if hasattr(y_national, "values") else y_national,
        dtype=torch.float32,
    )
    r = torch.tensor(r, dtype=torch.float32)

    def sre(x, y):
        one_way = ((1 + x) / (1 + y) - 1) ** 2
        other_way = ((1 + y) / (1 + x) - 1) ** 2
        return torch.min(one_way, other_way)

    def loss(w, validation: bool = False):
        pred_local = (w.unsqueeze(-1) * metrics.unsqueeze(0)).sum(dim=1)
        if dropout_targets and validation_targets_local is not None:
            if validation:
                mask = validation_targets_local
            else:
                mask = ~validation_targets_local
            pred_local = pred_local[:, mask]
            mse_local = torch.mean(sre(pred_local, y[:, mask]))
        else:
            mse_local = torch.mean(sre(pred_local, y))

        pred_national = (w.sum(axis=0) * matrix_national.T).sum(axis=1)
        if dropout_targets and validation_targets_national is not None:
            if validation:
                mask = validation_targets_national
            else:
                mask = ~validation_targets_national
            pred_national = pred_national[mask]
            mse_national = torch.mean(sre(pred_national, y_national[mask]))
        else:
            mse_national = torch.mean(sre(pred_national, y_national))

        return mse_local + mse_national

    def pct_close(w, t=0.1, local=True, national=True):
        """Return the percentage of metrics that are within t% of the target"""
        numerator = 0
        denominator = 0

        if local:
            pred_local = (w.unsqueeze(-1) * metrics.unsqueeze(0)).sum(dim=1)
            e_local = torch.sum(
                torch.abs((pred_local / (1 + y) - 1)) < t
            ).item()
            c_local = pred_local.shape[0] * pred_local.shape[1]
            numerator += e_local
            denominator += c_local

        if national:
            pred_national = (w.sum(axis=0) * matrix_national.T).sum(axis=1)
            e_national = torch.sum(
                torch.abs((pred_national / (1 + y_national) - 1)) < t
            ).item()
            c_national = pred_national.shape[0]
            numerator += e_national
            denominator += c_national

        return numerator / denominator

    def dropout_weights(weights, p):
        if p == 0:
            return weights
        # Replace p% of the weights with the mean value of the rest of them
        mask = torch.rand_like(weights) < p
        mean = weights[~mask].mean()
        masked_weights = weights.clone()
        masked_weights[mask] = mean
        return masked_weights

    optimizer = torch.optim.Adam([weights], lr=1e-1)
    final_weights = (torch.exp(weights) * r).detach().numpy()
    performance = pd.DataFrame()

    for epoch in range(epochs):
        optimizer.zero_grad()
        weights_ = torch.exp(dropout_weights(weights, 0.05)) * r
        l = loss(weights_)
        l.backward()
        optimizer.step()

        local_close = pct_close(weights_, local=True, national=False)
        national_close = pct_close(weights_, local=False, national=True)

        if verbose and (epoch % 1 == 0):
            if dropout_targets:
                validation_loss = loss(weights_, validation=True)
                print(
                    f"Training loss: {l.item():,.3f}, Validation loss: {validation_loss.item():,.3f}, Epoch: {epoch}, "
                    f"{area_name}<10%: {local_close:.1%}, National<10%: {national_close:.1%}"
                )
            else:
                print(
                    f"Loss: {l.item()}, Epoch: {epoch}, {area_name}<10%: {local_close:.1%}, National<10%: {national_close:.1%}"
                )

        if epoch % 10 == 0:
            final_weights = (torch.exp(weights) * r).detach().numpy()

            # Log performance if requested and get_performance function is available
            if log_csv:
                performance_step = get_performance(
                    final_weights,
                    m_c,
                    y_c,
                    m_n,
                    y_n,
                    excluded_training_targets,
                )
                performance_step["epoch"] = epoch
                performance_step["loss"] = performance_step.rel_abs_error**2
                performance_step["target_name"] = [
                    f"{area}/{metric}"
                    for area, metric in zip(
                        performance_step.name, performance_step.metric
                    )
                ]
                performance = pd.concat(
                    [performance, performance_step], ignore_index=True
                )
                performance.to_csv(log_csv, index=False)

            # Save weights
            with h5py.File(STORAGE_FOLDER / weight_file, "w") as f:
                f.create_dataset(dataset_key, data=final_weights)

            dataset.household.household_weight = final_weights.sum(axis=0)

    return dataset
