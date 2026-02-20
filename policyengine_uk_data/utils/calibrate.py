import torch
from policyengine_uk import Microsimulation
import pandas as pd
import numpy as np
import h5py
from policyengine_uk_data.storage import STORAGE_FOLDER
from policyengine_uk.data import UKSingleYearDataset
from policyengine_uk_data.utils.progress import ProcessingProgress


def _run_optimisation(
    matrix_np: np.ndarray,
    y_np: np.ndarray,
    r_np: np.ndarray,
    matrix_national_np: np.ndarray,
    y_national_np: np.ndarray,
    weights_init_np: np.ndarray,
    epochs: int,
    device: torch.device,
    excluded_training_targets_local: np.ndarray | None = None,
    excluded_training_targets_national: np.ndarray | None = None,
    verbose: bool = False,
    area_name: str = "area",
    progress_tracker=None,
    nested_progress=None,
    log_csv: str | None = None,
    get_performance=None,
    m_c_orig=None,
    y_c_orig=None,
    m_n_orig=None,
    y_n_orig=None,
    weight_file: str | None = None,
    dataset_key: str = "2025",
    dataset=None,
) -> np.ndarray:
    """
    Pure optimisation loop (Adam, PyTorch). Device-agnostic — pass
    ``device=torch.device("cuda")`` for GPU or ``"cpu"`` for CPU.

    Returns the final weights array (area_count × n_households).
    """
    metrics = torch.tensor(matrix_np, dtype=torch.float32, device=device)
    y = torch.tensor(y_np, dtype=torch.float32, device=device)
    matrix_national = torch.tensor(
        matrix_national_np, dtype=torch.float32, device=device
    )
    y_national = torch.tensor(
        y_national_np, dtype=torch.float32, device=device
    )
    r = torch.tensor(r_np, dtype=torch.float32, device=device)

    weights = torch.tensor(
        weights_init_np,
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )

    dropout_targets = (
        excluded_training_targets_local is not None
        and excluded_training_targets_local.any()
    )

    def sre(x, y_t):
        one_way = ((1 + x) / (1 + y_t) - 1) ** 2
        other_way = ((1 + y_t) / (1 + x) - 1) ** 2
        return torch.min(one_way, other_way)

    def loss_fn(w, validation: bool = False):
        pred_local = (w.unsqueeze(-1) * metrics.unsqueeze(0)).sum(dim=1)
        if dropout_targets and excluded_training_targets_local is not None:
            mask = (
                excluded_training_targets_local
                if validation
                else ~excluded_training_targets_local
            )
            pred_local = pred_local[:, mask]
            mse_local = torch.mean(sre(pred_local, y[:, mask]))
        else:
            mse_local = torch.mean(sre(pred_local, y))

        pred_national = (w.sum(axis=0) * matrix_national.T).sum(axis=1)
        if dropout_targets and excluded_training_targets_national is not None:
            mask = (
                excluded_training_targets_national
                if validation
                else ~excluded_training_targets_national
            )
            pred_national = pred_national[mask]
            mse_national = torch.mean(sre(pred_national, y_national[mask]))
        else:
            mse_national = torch.mean(sre(pred_national, y_national))

        return mse_local + mse_national

    def pct_close(w, t=0.1, local=True, national=True):
        numerator = 0
        denominator = 0
        if local:
            pred_local = (w.unsqueeze(-1) * metrics.unsqueeze(0)).sum(dim=1)
            numerator += torch.sum(
                torch.abs((pred_local / (1 + y) - 1)) < t
            ).item()
            denominator += pred_local.shape[0] * pred_local.shape[1]
        if national:
            pred_national = (w.sum(axis=0) * matrix_national.T).sum(axis=1)
            numerator += torch.sum(
                torch.abs((pred_national / (1 + y_national) - 1)) < t
            ).item()
            denominator += pred_national.shape[0]
        return numerator / denominator

    def dropout_weights(w, p):
        if p == 0:
            return w
        mask = torch.rand_like(w) < p
        mean = w[~mask].mean()
        w2 = w.clone()
        w2[mask] = mean
        return w2

    optimizer = torch.optim.Adam([weights], lr=1e-1)
    final_weights = (torch.exp(weights) * r).detach().cpu().numpy()
    performance = pd.DataFrame()

    def _epoch_step(epoch):
        nonlocal final_weights, performance
        optimizer.zero_grad()
        weights_ = torch.exp(dropout_weights(weights, 0.05)) * r
        l = loss_fn(weights_)
        l.backward()
        optimizer.step()

        local_close = pct_close(weights_, local=True, national=False)
        national_close = pct_close(weights_, local=False, national=True)

        if epoch % 10 == 0:
            final_weights = (torch.exp(weights) * r).detach().cpu().numpy()

            if log_csv and get_performance and m_c_orig is not None:
                perf = get_performance(
                    final_weights,
                    m_c_orig,
                    y_c_orig,
                    m_n_orig,
                    y_n_orig,
                    [],
                )
                perf["epoch"] = epoch
                perf["loss"] = perf.rel_abs_error**2
                perf["target_name"] = [
                    f"{a}/{m}" for a, m in zip(perf.name, perf.metric)
                ]
                performance = pd.concat([performance, perf], ignore_index=True)
                performance.to_csv(log_csv, index=False)

            if weight_file:
                with h5py.File(STORAGE_FOLDER / weight_file, "w") as f:
                    f.create_dataset(dataset_key, data=final_weights)
            if dataset is not None:
                dataset.household.household_weight = final_weights.sum(axis=0)

        return l, local_close, national_close

    if verbose and progress_tracker is not None:
        with progress_tracker.track_calibration(
            epochs, nested_progress
        ) as update_calibration:
            for epoch in range(epochs):
                update_calibration(epoch + 1, calculating_loss=True)
                l, _, _ = _epoch_step(epoch)
                if dropout_targets:
                    weights_ = torch.exp(dropout_weights(weights, 0.05)) * r
                    val_loss = loss_fn(weights_, validation=True)
                    update_calibration(
                        epoch + 1,
                        loss_value=val_loss.item(),
                        calculating_loss=False,
                    )
                else:
                    update_calibration(
                        epoch + 1,
                        loss_value=l.item(),
                        calculating_loss=False,
                    )
    else:
        for epoch in range(epochs):
            _epoch_step(epoch)

    return final_weights


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
) -> UKSingleYearDataset:
    """
    Calibrate local-area weights on CPU using the extracted optimisation loop.
    """
    dataset = dataset.copy()
    matrix, y, r = matrix_fn(dataset)
    m_c, y_c = matrix.copy(), y.copy()
    m_national, y_national = national_matrix_fn(dataset)
    m_n, y_n = m_national.copy(), y_national.copy()

    areas_per_household = r.sum(axis=0)
    areas_per_household = np.maximum(areas_per_household, 1)
    original_weights = np.log(
        dataset.household.household_weight.values / areas_per_household
        + np.random.random(len(dataset.household.household_weight.values))
        * 0.01
    )
    weights_init = (
        np.ones((area_count, len(original_weights))) * original_weights
    )

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

    progress_tracker = ProcessingProgress() if verbose else None

    final_weights = _run_optimisation(
        matrix_np=matrix.values if hasattr(matrix, "values") else matrix,
        y_np=y.values if hasattr(y, "values") else y,
        r_np=r,
        matrix_national_np=(
            m_national.values if hasattr(m_national, "values") else m_national
        ),
        y_national_np=(
            y_national.values if hasattr(y_national, "values") else y_national
        ),
        weights_init_np=weights_init,
        epochs=epochs,
        device=torch.device("cpu"),
        excluded_training_targets_local=validation_targets_local,
        excluded_training_targets_national=validation_targets_national,
        verbose=verbose,
        area_name=area_name,
        progress_tracker=progress_tracker,
        nested_progress=nested_progress,
        log_csv=log_csv,
        get_performance=get_performance,
        m_c_orig=m_c,
        y_c_orig=y_c,
        m_n_orig=m_n,
        y_n_orig=y_n,
        weight_file=weight_file,
        dataset_key=dataset_key,
        dataset=dataset,
    )

    dataset.household.household_weight = final_weights.sum(axis=0)
    return dataset
