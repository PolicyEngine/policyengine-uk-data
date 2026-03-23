import logging
from contextlib import nullcontext
from pathlib import Path
from typing import Optional, Union

import torch
import pandas as pd
import numpy as np
import h5py
from policyengine_uk_data.storage import STORAGE_FOLDER
from policyengine_uk.data import UKSingleYearDataset
from policyengine_uk_data.utils.progress import ProcessingProgress

logger = logging.getLogger(__name__)

# Population gets this multiplier in the national loss so the optimiser
# keeps it on target rather than letting it drift ~6% high.
POPULATION_LOSS_WEIGHT = 10.0

def load_weights(
    weight_file: Union[str, Path],
    dataset_key: str = "2025",
    n_areas: Optional[int] = None,
    n_records: Optional[int] = None,
) -> np.ndarray:
    """Load calibration weights from an h5 file and normalise their shape.

    Two calibration back-ends exist in this repo's history: the L2
    calibrator in `calibrate_local_areas` (this module) saves weights as a
    2D ``(n_areas, n_records)`` array, while the L0-regularised variant
    (when present) sometimes saves a flat 1D ``(n_records,)`` array under
    the same dataset key. Consumers that are not careful about axes can
    therefore silently read the wrong shape.

    This helper centralises loading and always returns a 2D
    ``(n_areas, n_records)`` array. A 1D input is reshaped to
    ``(1, n_records)`` so downstream ``.sum(axis=0)`` and matrix-multiply
    operations behave consistently. Optional ``n_areas`` / ``n_records``
    arguments raise a clear ``ValueError`` on shape mismatch instead of
    silently producing wrong answers.

    Args:
        weight_file: Path to the h5 file written by a calibrator. If the
            path is not absolute it is resolved relative to the package
            ``STORAGE_FOLDER``.
        dataset_key: H5 dataset key to read.
        n_areas: Optional expected number of areas (first axis). When
            provided, a 1D input is reshaped and its length checked; a 2D
            input has its first axis checked.
        n_records: Optional expected number of records (second axis).
            Checked against the final axis of the loaded array.

    Returns:
        A 2D ``(n_areas, n_records)`` numpy array.
    """
    path = Path(weight_file)
    if not path.is_absolute():
        path = STORAGE_FOLDER / path

    with h5py.File(path, "r") as f:
        if dataset_key not in f:
            available = ", ".join(sorted(f.keys()))
            raise KeyError(
                f"Dataset key {dataset_key!r} not found in {path}; "
                f"available keys: {available}"
            )
        arr = f[dataset_key][:]

    if arr.ndim == 1:
        # Flat (n_records,) layout — promote to (1, n_records) so callers
        # can treat all weights as a 2D matrix.
        arr = arr.reshape(1, -1)
    elif arr.ndim != 2:
        raise ValueError(
            f"Expected weights at {dataset_key!r} in {path} to be 1D or 2D; "
            f"got shape {arr.shape}"
        )

    if n_areas is not None and arr.shape[0] != n_areas:
        raise ValueError(
            f"Weights at {dataset_key!r} in {path} have {arr.shape[0]} areas, "
            f"expected {n_areas}"
        )
    if n_records is not None and arr.shape[-1] != n_records:
        raise ValueError(
            f"Weights at {dataset_key!r} in {path} have {arr.shape[-1]} "
            f"records, expected {n_records}"
        )
    return arr


def _build_national_target_weights(
    national_matrix,
    population_weight: float = POPULATION_LOSS_WEIGHT,
) -> np.ndarray:
    """Build per-target weight vector for the national loss.

    Every target gets weight 1.0 except ``ons/uk_population`` which gets
    ``population_weight``.  This ensures the optimiser treats population
    accuracy as a hard constraint rather than 1-of-N soft targets.
    """
    pop_col_name = "ons/uk_population"
    if hasattr(national_matrix, "columns"):
        n = len(national_matrix.columns)
        w = np.ones(n, dtype=np.float32)
        cols = list(national_matrix.columns)
        if pop_col_name in cols:
            w[cols.index(pop_col_name)] = population_weight
        return w
    # Fallback: no column names available — equal weights
    return np.ones(national_matrix.shape[1], dtype=np.float32)


def calibrate_local_areas(
    dataset: UKSingleYearDataset,
    matrix_fn,
    national_matrix_fn,
    area_count: int,
    weight_file: str,
    dataset_key: str = "2025",
    epochs: int = 512,
    excluded_training_targets=[],
    log_csv=None,
    verbose: bool = False,
    area_name: str = "area",
    get_performance=None,
    nested_progress=None,
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
    progress_tracker = ProcessingProgress() if verbose else None

    def track_stage(stage_name: str):
        if progress_tracker is None:
            return nullcontext()
        return progress_tracker.track_stage(stage_name)

    with track_stage(f"{area_name}: copy dataset"):
        dataset = dataset.copy()

    with track_stage(f"{area_name}: build local target matrix"):
        matrix, y, r = matrix_fn(dataset)
    m_c, y_c = matrix.copy(), y.copy()

    with track_stage(f"{area_name}: build national target matrix"):
        m_national, y_national = national_matrix_fn(dataset)
    m_n, y_n = m_national.copy(), y_national.copy()

    with track_stage(f"{area_name}: prepare tensors and optimizer"):
        # Weights - area_count x num_households
        # Use country-aware initialization: divide each household's weight by the
        # number of areas in its country, not the total area count. This ensures
        # households start at approximately correct weight for their country's targets.
        # The country_mask r[i,j]=1 iff household j is in same country as area i.
        areas_per_household = r.sum(
            axis=0
        )  # number of areas each household can contribute to
        areas_per_household = np.maximum(
            areas_per_household, 1
        )  # avoid division by zero
        original_weights = np.log(
            dataset.household.household_weight.values / areas_per_household
            + np.random.random(len(dataset.household.household_weight.values)) * 0.01
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
        y = torch.tensor(y.values if hasattr(y, "values") else y, dtype=torch.float32)
        matrix_national = torch.tensor(
            m_national.values if hasattr(m_national, "values") else m_national,
            dtype=torch.float32,
        )
        y_national = torch.tensor(
            y_national.values if hasattr(y_national, "values") else y_national,
            dtype=torch.float32,
        )
        r = torch.tensor(r, dtype=torch.float32)

    # Per-target weights for the national loss (population gets boosted)
    national_target_weights = torch.tensor(
        _build_national_target_weights(m_national),
        dtype=torch.float32,
    )

    def sre(x, y):
        one_way = ((1 + x) / (1 + y) - 1) ** 2
        other_way = ((1 + y) / (1 + x) - 1) ** 2
        return torch.min(one_way, other_way)

    def weighted_mean(values, weights):
        return (values * weights).sum() / weights.sum()

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
            mse_national = weighted_mean(
                sre(pred_national, y_national[mask]),
                national_target_weights[mask],
            )
        else:
            mse_national = weighted_mean(
                sre(pred_national, y_national),
                national_target_weights,
            )

        return mse_local + mse_national

    def pct_close(w, t=0.1, local=True, national=True):
        """Return the percentage of metrics that are within t% of the target"""
        numerator = 0
        denominator = 0

        if local:
            pred_local = (w.unsqueeze(-1) * metrics.unsqueeze(0)).sum(dim=1)
            e_local = torch.sum(torch.abs((pred_local / (1 + y) - 1)) < t).item()
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

    if verbose and progress_tracker:
        with progress_tracker.track_calibration(
            epochs, nested_progress
        ) as update_calibration:
            for epoch in range(epochs):
                update_calibration(epoch + 1, calculating_loss=True)

                optimizer.zero_grad()
                weights_ = torch.exp(dropout_weights(weights, 0.05)) * r
                loss_value = loss(weights_)
                loss_value.backward()
                optimizer.step()

                local_close = pct_close(weights_, local=True, national=False)
                national_close = pct_close(weights_, local=False, national=True)

                if dropout_targets:
                    validation_loss = loss(weights_, validation=True)
                    update_calibration(
                        epoch + 1,
                        loss_value=validation_loss.item(),
                        calculating_loss=False,
                    )
                else:
                    update_calibration(
                        epoch + 1,
                        loss_value=loss_value.item(),
                        calculating_loss=False,
                    )

                if epoch % 10 == 0:
                    final_weights = (torch.exp(weights) * r).detach().numpy()

                    # Log performance if requested and get_performance function is available
                    if log_csv and get_performance:
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
    else:
        for epoch in range(epochs):
            optimizer.zero_grad()
            weights_ = torch.exp(dropout_weights(weights, 0.05)) * r
            loss_value = loss(weights_)
            loss_value.backward()
            optimizer.step()

            local_close = pct_close(weights_, local=True, national=False)
            national_close = pct_close(weights_, local=False, national=True)

            if verbose and (epoch % 1 == 0):
                if dropout_targets:
                    validation_loss = loss(weights_, validation=True)
                    print(
                        f"Training loss: {loss_value.item():,.3f}, Validation loss: {validation_loss.item():,.3f}, Epoch: {epoch}, "
                        f"{area_name}<10%: {local_close:.1%}, National<10%: {national_close:.1%}"
                    )
                else:
                    print(
                        f"Loss: {loss_value.item()}, Epoch: {epoch}, {area_name}<10%: {local_close:.1%}, National<10%: {national_close:.1%}"
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
