"""National household-weight calibration utilities for public datasets."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import lsq_linear

from policyengine_uk_data.targets.build_loss_matrix import create_target_matrix
from policyengine_uk_data.utils.loss import get_loss_results


@dataclass
class ReweightDiagnostics:
    target_count: int
    mean_abs_rel_error_before: float
    mean_abs_rel_error_after: float
    median_abs_rel_error_before: float
    median_abs_rel_error_after: float
    pct_within_10_before: float
    pct_within_10_after: float


def calibrate_household_weights(
    dataset,
    time_period: str,
    reform=None,
    min_weight: float = 1e-9,
    max_iter: int = 2000,
    compute_diagnostics: bool = True,
) -> tuple[np.ndarray, ReweightDiagnostics | None]:
    """Calibrate household weights to national/region/country targets.

    This fits non-negative household weights against the target matrix using a
    bounded least-squares solve on relative errors. It is intentionally simpler
    than the local-area calibration pipeline and is suitable for smaller public
    datasets such as the UK enhanced CPS.
    """

    matrix, targets = create_target_matrix(dataset, time_period, reform)
    household_count = len(dataset.household)
    if household_count == 0:
        raise ValueError("Dataset must contain at least one household")

    coefficients = matrix.values.T.astype(float)
    target_values = targets.values.astype(float)
    scaled_denominator = 1.0 + np.abs(target_values)

    weighted_coefficients = coefficients / scaled_denominator[:, None]
    weighted_targets = target_values / scaled_denominator

    result = lsq_linear(
        weighted_coefficients,
        weighted_targets,
        bounds=(min_weight, np.inf),
        tol=1e-4,
        lsmr_tol="auto",
        max_iter=max_iter,
    )
    if not result.success:
        raise RuntimeError(f"National calibration failed: {result.message}")

    calibrated_weights = result.x

    diagnostics = None
    if compute_diagnostics:
        before = get_loss_results(dataset, time_period, reform=reform)
        after = get_loss_results(
            dataset,
            time_period,
            reform=reform,
            household_weights=calibrated_weights,
        )

        diagnostics = ReweightDiagnostics(
            target_count=len(after),
            mean_abs_rel_error_before=float(before.abs_rel_error.mean()),
            mean_abs_rel_error_after=float(after.abs_rel_error.mean()),
            median_abs_rel_error_before=float(before.abs_rel_error.median()),
            median_abs_rel_error_after=float(after.abs_rel_error.median()),
            pct_within_10_before=float((before.abs_rel_error <= 0.10).mean()),
            pct_within_10_after=float((after.abs_rel_error <= 0.10).mean()),
        )
    return calibrated_weights, diagnostics
