from contextlib import nullcontext
from inspect import signature
from pathlib import Path
from typing import Optional, Union

import torch
import pandas as pd
import numpy as np
import h5py
from policyengine_uk_data.storage import STORAGE_FOLDER
from policyengine_uk.data import UKSingleYearDataset
from policyengine_uk_data.datasets.frs_release import CURRENT_FRS_RELEASE
from policyengine_uk_data.utils.progress import ProcessingProgress


DEFAULT_ZERO_WEIGHT_PRIOR_TOTAL_SHARE = 0.5


def default_weight_dataset_key() -> str:
    return str(CURRENT_FRS_RELEASE.calibration_year)


def _call_matrix_fn(matrix_fn, dataset, time_period):
    if time_period is None:
        return matrix_fn(dataset)

    parameters = signature(matrix_fn).parameters
    accepts_time_period = "time_period" in parameters or any(
        p.kind == p.VAR_KEYWORD for p in parameters.values()
    )
    if accepts_time_period:
        return matrix_fn(dataset, time_period=time_period)
    return matrix_fn(dataset)


def load_weights(
    weight_file: Union[str, Path],
    dataset_key: str | None = None,
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
    if dataset_key is None:
        dataset_key = default_weight_dataset_key()

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


def initialize_weight_priors(
    original_weights: np.ndarray,
    zero_weight_total_share: float = DEFAULT_ZERO_WEIGHT_PRIOR_TOTAL_SHARE,
) -> np.ndarray:
    """Build deterministic positive household priors for calibration.

    SPI synthetic households enter the enhanced FRS with zero household
    weight. Giving those rows tiny random priors makes them technically
    positive but practically unavailable in log-space optimization. When
    zero-weight rows are present, preserve the relative distribution of
    positive survey weights while reserving a fixed share of total prior mass
    for the zero-weight rows.
    """
    weights = np.asarray(original_weights, dtype=np.float64)
    if weights.ndim != 1:
        raise ValueError("original_weights must be one-dimensional")
    if np.any(weights < 0):
        raise ValueError("original_weights must be non-negative")
    if weights.size == 0:
        return weights.copy()
    if not 0 < zero_weight_total_share < 1:
        raise ValueError("zero_weight_total_share must be between 0 and 1")

    positive_mask = weights > 0
    zero_mask = ~positive_mask
    if not zero_mask.any():
        return weights.copy()

    positive_total = float(weights[positive_mask].sum())
    if positive_total <= 0:
        return np.full_like(weights, 1.0, dtype=np.float64)

    priors = np.empty_like(weights, dtype=np.float64)
    priors[positive_mask] = weights[positive_mask] * (1 - zero_weight_total_share)
    priors[zero_mask] = positive_total * zero_weight_total_share / zero_mask.sum()
    return priors


def _as_bool_mask(series: pd.Series) -> np.ndarray:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).to_numpy(dtype=bool)
    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0).to_numpy(dtype=float) != 0
    return (
        series.fillna("").astype(str).str.lower().isin({"true", "1", "yes"}).to_numpy()
    )


def _weight_share(weight: float, total: float) -> float:
    return weight / total if total > 0 else 0.0


def _household_weight_diagnostics(
    dataset: UKSingleYearDataset,
    household_weights: np.ndarray,
    household_priors: np.ndarray,
) -> dict[str, float | int]:
    household_weights = np.asarray(household_weights, dtype=np.float64)
    household_priors = np.asarray(household_priors, dtype=np.float64)
    original_weights = dataset.household.household_weight.values.astype(np.float64)
    total_weight = float(household_weights.sum())
    total_prior = float(household_priors.sum())
    zero_initial = original_weights <= 0

    zero_weight = float(household_weights[zero_initial].sum())
    zero_prior = float(household_priors[zero_initial].sum())
    diagnostics: dict[str, float | int] = {
        "total_household_weight": total_weight,
        "prior_total_household_weight": total_prior,
        "initial_zero_weight_rows": int(zero_initial.sum()),
        "initial_zero_weight_household_weight": zero_weight,
        "initial_zero_weight_household_weight_share": _weight_share(
            zero_weight, total_weight
        ),
        "initial_zero_weight_prior_household_weight": zero_prior,
        "initial_zero_weight_prior_share": _weight_share(zero_prior, total_prior),
    }

    for column in dataset.household.columns:
        if not column.startswith("household_is_"):
            continue
        mask = _as_bool_mask(dataset.household[column])
        source_weight = float(household_weights[mask].sum())
        source_prior = float(household_priors[mask].sum())
        diagnostics[f"{column}_rows"] = int(mask.sum())
        diagnostics[f"{column}_positive_weight_rows"] = int(
            (household_weights[mask] > 0).sum()
        )
        diagnostics[f"{column}_household_weight"] = source_weight
        diagnostics[f"{column}_household_weight_share"] = _weight_share(
            source_weight, total_weight
        )
        diagnostics[f"{column}_prior_household_weight"] = source_prior
        diagnostics[f"{column}_prior_share"] = _weight_share(source_prior, total_prior)

    return diagnostics


def calibrate_local_areas(
    dataset: UKSingleYearDataset,
    matrix_fn,
    national_matrix_fn,
    area_count: int,
    weight_file: str,
    dataset_key: str | None = None,
    epochs: int = 512,
    excluded_training_targets=[],
    log_csv=None,
    verbose: bool = False,
    area_name: str = "area",
    get_performance=None,
    nested_progress=None,
    time_period: int | str | None = None,
    zero_weight_prior_total_share: float = DEFAULT_ZERO_WEIGHT_PRIOR_TOTAL_SHARE,
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
        zero_weight_prior_total_share: Share of prior household mass to reserve for
            rows whose incoming household_weight is zero.
    """
    if dataset_key is None:
        dataset_key = default_weight_dataset_key()
    if time_period is None and str(dataset_key).isdigit():
        time_period = dataset_key

    progress_tracker = ProcessingProgress() if verbose else None

    def track_stage(stage_name: str):
        if progress_tracker is None:
            return nullcontext()
        return progress_tracker.track_stage(stage_name)

    with track_stage(f"{area_name}: copy dataset"):
        dataset = dataset.copy()

    with track_stage(f"{area_name}: build local target matrix"):
        matrix, y, r = _call_matrix_fn(matrix_fn, dataset, time_period)
    m_c, y_c = matrix.copy(), y.copy()

    with track_stage(f"{area_name}: build national target matrix"):
        m_national, y_national = _call_matrix_fn(
            national_matrix_fn, dataset, time_period
        )
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
        household_prior_weights = initialize_weight_priors(
            dataset.household.household_weight.values,
            zero_weight_total_share=zero_weight_prior_total_share,
        )
        area_prior_weights = household_prior_weights / areas_per_household
        original_weights = np.log(np.clip(area_prior_weights, 1e-12, None))
        weights = torch.tensor(
            np.ones((area_count, len(original_weights))) * original_weights,
            dtype=torch.float32,
            requires_grad=True,
        )

        # Set up validation targets if specified
        validation_targets_local = (
            torch.tensor(
                matrix.columns.isin(excluded_training_targets),
                dtype=torch.bool,
            )
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
        y_values = y.values if hasattr(y, "values") else y
        local_target_available = torch.tensor(np.isfinite(y_values), dtype=torch.bool)
        y = torch.tensor(np.nan_to_num(y_values, nan=0.0), dtype=torch.float32)
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
        local_mask = local_target_available
        if dropout_targets and validation_targets_local is not None:
            if validation:
                column_mask = validation_targets_local
            else:
                column_mask = ~validation_targets_local
            local_mask = local_mask & column_mask.unsqueeze(0)

        if local_mask.any():
            mse_local = torch.mean(sre(pred_local[local_mask], y[local_mask]))
        else:
            mse_local = pred_local.sum() * 0

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
                (torch.abs((pred_local / (1 + y) - 1)) < t) & local_target_available
            ).item()
            c_local = torch.sum(local_target_available).item()
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

    def log_performance(
        epoch: int,
        final_weights: np.ndarray,
        training_loss: float,
        local_close: float,
        national_close: float,
        validation_loss: float | None = None,
    ):
        nonlocal performance

        if not log_csv or get_performance is None:
            return

        saved_weights_tensor = torch.tensor(final_weights, dtype=torch.float32)
        saved_weights_loss = float(loss(saved_weights_tensor).item())
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
        performance_step["training_loss"] = training_loss
        performance_step["saved_weights_loss"] = saved_weights_loss
        performance_step["validation_loss"] = validation_loss
        performance_step["local_pct_close_10pct"] = local_close
        performance_step["national_pct_close_10pct"] = national_close
        performance_step["target_name"] = [
            f"{area}/{metric}"
            for area, metric in zip(performance_step.name, performance_step.metric)
        ]

        diagnostics = _household_weight_diagnostics(
            dataset,
            final_weights.sum(axis=0),
            household_prior_weights,
        )
        for key, value in diagnostics.items():
            performance_step[key] = value

        performance = pd.concat([performance, performance_step], ignore_index=True)
        performance.to_csv(log_csv, index=False)

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
                    validation_loss_value = float(validation_loss.item())
                    update_calibration(
                        epoch + 1,
                        loss_value=validation_loss_value,
                        calculating_loss=False,
                    )
                else:
                    validation_loss_value = None
                    update_calibration(
                        epoch + 1,
                        loss_value=loss_value.item(),
                        calculating_loss=False,
                    )

                if epoch % 10 == 0:
                    final_weights = (torch.exp(weights) * r).detach().numpy()

                    log_performance(
                        epoch=epoch,
                        final_weights=final_weights,
                        training_loss=float(loss_value.item()),
                        validation_loss=validation_loss_value,
                        local_close=local_close,
                        national_close=national_close,
                    )

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

                log_performance(
                    epoch=epoch,
                    final_weights=final_weights,
                    training_loss=float(loss_value.item()),
                    validation_loss=None,
                    local_close=local_close,
                    national_close=national_close,
                )

                # Save weights
                with h5py.File(STORAGE_FOLDER / weight_file, "w") as f:
                    f.create_dataset(dataset_key, data=final_weights)

                dataset.household.household_weight = final_weights.sum(axis=0)

    return dataset
