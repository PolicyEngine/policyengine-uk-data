"""L0-regularised calibration for cloned FRS datasets.

Uses HardConcrete gates from the l0-python package to produce sparse
household weights — most clones get pruned to zero, keeping only the
best-fitting records per area.

This replaces the dense Adam reweighting in calibrate.py when the
dataset has been through clone-and-assign (Phase 2). The existing
calibrate.py is kept as a fallback.

US reference: policyengine-us-data PRs #364, #365.
"""

import logging

import h5py
import numpy as np
import pandas as pd
import torch
from scipy import sparse as sp

from policyengine_uk_data.storage import STORAGE_FOLDER

logger = logging.getLogger(__name__)


def _build_sparse_calibration_matrix(
    metrics: pd.DataFrame,
    targets: pd.DataFrame,
    country_mask: np.ndarray,
    national_metrics: pd.DataFrame,
    national_targets: pd.DataFrame,
):
    """Build a sparse (n_targets, n_records) matrix and target vector.

    Flattens the (area x metric) structure into rows, with country
    masking baked into the sparsity pattern (households that don't
    belong to an area's country have zero entries).

    Args:
        metrics: (n_records, n_metrics) household-level metric values.
        targets: (n_areas, n_metrics) target values per area.
        country_mask: (n_areas, n_records) binary mask.
        national_metrics: (n_records, n_national_metrics) national metrics.
        national_targets: (n_national_metrics,) national target values.

    Returns:
        M: scipy.sparse.csr_matrix of shape (n_total_targets, n_records)
        y: numpy array of shape (n_total_targets,)
        target_groups: numpy array of group IDs for equal weighting
    """
    metric_values = metrics.values if hasattr(metrics, "values") else np.array(metrics)
    target_values = targets.values if hasattr(targets, "values") else np.array(targets)

    n_areas, n_metrics = target_values.shape
    n_records = metric_values.shape[0]

    # Build sparse matrix: each row is (area_i, metric_j)
    # M[row, record] = metric_values[record, j] * country_mask[i, record]
    rows = []
    cols = []
    data = []
    y_list = []
    group_ids = []

    for j in range(n_metrics):
        metric_col = metric_values[:, j]
        for i in range(n_areas):
            target_val = target_values[i, j]
            # Skip zero/nan targets
            if np.isnan(target_val) or target_val == 0:
                continue

            # Records in this area's country
            mask = country_mask[i] > 0
            record_indices = np.where(mask)[0]
            values = metric_col[record_indices]

            # Skip if no non-zero contributions
            nonzero = values != 0
            if not nonzero.any():
                continue

            row_idx = len(y_list)
            record_indices = record_indices[nonzero]
            values = values[nonzero]

            rows.extend([row_idx] * len(record_indices))
            cols.extend(record_indices.tolist())
            data.extend(values.tolist())
            y_list.append(target_val)
            group_ids.append(j)  # Group by metric type

    # Add national targets
    national_metric_values = (
        national_metrics.values
        if hasattr(national_metrics, "values")
        else np.array(national_metrics)
    )
    national_target_values = (
        national_targets.values
        if hasattr(national_targets, "values")
        else np.array(national_targets)
    )

    n_national = len(national_target_values)
    national_group_start = n_metrics  # Offset group IDs

    for j in range(n_national):
        target_val = national_target_values[j]
        if np.isnan(target_val) or target_val == 0:
            continue

        metric_col = national_metric_values[:, j]
        nonzero = metric_col != 0
        if not nonzero.any():
            continue

        row_idx = len(y_list)
        record_indices = np.where(nonzero)[0]
        values = metric_col[record_indices]

        rows.extend([row_idx] * len(record_indices))
        cols.extend(record_indices.tolist())
        data.extend(values.tolist())
        y_list.append(target_val)
        group_ids.append(national_group_start + j)

    n_total_targets = len(y_list)
    M = sp.csr_matrix(
        (data, (rows, cols)),
        shape=(n_total_targets, n_records),
    )
    y = np.array(y_list, dtype=np.float64)
    group_ids = np.array(group_ids, dtype=np.int64)

    logger.info(
        "Sparse calibration matrix: %d targets x %d records, %.1f%% non-zero",
        n_total_targets,
        n_records,
        100 * M.nnz / (n_total_targets * n_records) if n_total_targets > 0 else 0,
    )

    return M, y, group_ids


def calibrate_l0(
    dataset,
    matrix_fn,
    national_matrix_fn,
    area_count: int,
    weight_file: str,
    dataset_key: str = "2025",
    epochs: int = 1000,
    lambda_l0: float = 0.01,
    lambda_l2: float = 1e-6,
    lr: float = 0.01,
    init_keep_prob: float = 0.5,
    excluded_training_targets=None,
    log_csv=None,
    verbose: bool = False,
    area_name: str = "area",
    get_performance=None,
    nested_progress=None,
):
    """Calibrate local area weights using L0-regularised optimisation.

    Uses HardConcrete gates to produce sparse weights — most cloned
    households are pruned to zero, keeping only a sparse subset per
    area.

    Args:
        dataset: UKSingleYearDataset (post clone-and-assign).
        matrix_fn: Returns (metrics, targets, country_mask) for local areas.
        national_matrix_fn: Returns (metrics, targets) for national totals.
        area_count: Number of areas (e.g. 650 constituencies).
        weight_file: HDF5 file name for saving weights.
        dataset_key: Key within HDF5 file.
        epochs: Training epochs.
        lambda_l0: L0 regularisation strength.
        lambda_l2: L2 regularisation strength.
        lr: Learning rate.
        init_keep_prob: Initial gate keep probability.
        excluded_training_targets: Targets to exclude from training.
        log_csv: CSV path for performance logging.
        verbose: Print progress.
        area_name: Area type name for logging.
        get_performance: Performance evaluation function.
        nested_progress: Progress tracker for nested display.

    Returns:
        dataset with calibrated household_weight.
    """
    from l0.calibration import SparseCalibrationWeights

    if excluded_training_targets is None:
        excluded_training_targets = []

    dataset = dataset.copy()

    # Build target matrices using existing functions
    metrics, targets, country_mask = matrix_fn(dataset)
    national_metrics, national_targets = national_matrix_fn(dataset)

    n_records = len(dataset.household)

    # Build sparse calibration matrix
    M, y, target_groups = _build_sparse_calibration_matrix(
        metrics=metrics,
        targets=targets,
        country_mask=country_mask,
        national_metrics=national_metrics,
        national_targets=national_targets,
    )

    # Initialise weights from household_weight
    init_weights = dataset.household.household_weight.values.astype(np.float64)

    logger.info(
        "L0 calibration: %d records, %d targets, "
        "lambda_l0=%.4f, lambda_l2=%.6f, epochs=%d",
        n_records,
        len(y),
        lambda_l0,
        lambda_l2,
        epochs,
    )

    # Create and fit the L0 model
    model = SparseCalibrationWeights(
        n_features=n_records,
        init_weights=init_weights,
        init_keep_prob=init_keep_prob,
        device="cpu",
    )

    model.fit(
        M=M,
        y=y,
        lambda_l0=lambda_l0,
        lambda_l2=lambda_l2,
        lr=lr,
        epochs=epochs,
        loss_type="relative",
        verbose=verbose,
        verbose_freq=50,
        target_groups=target_groups,
    )

    # Extract final weights
    with torch.no_grad():
        final_weights = model.get_weights(deterministic=True).numpy()

    sparsity = model.get_sparsity()
    active = model.get_active_weights()
    logger.info(
        "L0 calibration complete: %.1f%% sparse, %d/%d active records",
        sparsity * 100,
        active["count"],
        n_records,
    )

    # Save weights as flat vector (not area x household matrix)
    with h5py.File(STORAGE_FOLDER / weight_file, "w") as f:
        f.create_dataset(dataset_key, data=final_weights)
        f.create_dataset(f"{dataset_key}_sparsity", data=sparsity)

    # Update dataset weights
    dataset.household.household_weight = final_weights

    return dataset
