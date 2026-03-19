"""Tests for L0-regularised calibration.

Validates that the L0 calibrator produces sparse weights that
match targets within reasonable tolerances.
"""

import numpy as np
import pandas as pd
import pytest
from scipy import sparse as sp

from policyengine_uk_data.utils.calibrate_l0 import (
    _build_sparse_calibration_matrix,
    calibrate_l0,
)


class TestBuildSparseMatrix:
    def test_output_shapes(self):
        n_records = 100
        n_metrics = 5
        n_areas = 3

        metrics = pd.DataFrame(
            np.random.rand(n_records, n_metrics),
            columns=[f"m{i}" for i in range(n_metrics)],
        )
        targets = pd.DataFrame(
            np.random.rand(n_areas, n_metrics) * 1000 + 1,
            columns=[f"m{i}" for i in range(n_metrics)],
        )
        country_mask = np.ones((n_areas, n_records))
        national_metrics = pd.DataFrame(
            np.random.rand(n_records, 2),
            columns=["n0", "n1"],
        )
        national_targets = pd.DataFrame({"n0": [5000.0], "n1": [3000.0]}).iloc[0]

        M, y, groups = _build_sparse_calibration_matrix(
            metrics, targets, country_mask, national_metrics, national_targets
        )

        assert isinstance(M, sp.csr_matrix)
        assert M.shape[1] == n_records
        assert len(y) == M.shape[0]
        assert len(groups) == M.shape[0]

    def test_country_masking_sparsity(self):
        """Records outside an area's country should have zero entries."""
        n_records = 20
        metrics = pd.DataFrame({"m0": np.ones(n_records)})
        targets = pd.DataFrame({"m0": [100.0, 200.0]})

        # Area 0 sees records 0-9, area 1 sees records 10-19
        country_mask = np.zeros((2, n_records))
        country_mask[0, :10] = 1
        country_mask[1, 10:] = 1

        national_metrics = pd.DataFrame({"n0": np.ones(n_records)})
        national_targets = pd.Series({"n0": 1000.0})

        M, y, groups = _build_sparse_calibration_matrix(
            metrics, targets, country_mask, national_metrics, national_targets
        )

        # Local target rows should only have entries for their country
        # Row 0 (area 0, metric 0) should have entries in cols 0-9 only
        # Row 1 (area 1, metric 0) should have entries in cols 10-19 only
        row0 = M.getrow(0).toarray().flatten()
        row1 = M.getrow(1).toarray().flatten()

        assert np.all(row0[10:] == 0), "Area 0 should not see records 10-19"
        assert np.all(row1[:10] == 0), "Area 1 should not see records 0-9"

    def test_zero_targets_skipped(self):
        """Zero target values should be omitted."""
        n_records = 10
        metrics = pd.DataFrame({"m0": np.ones(n_records), "m1": np.ones(n_records)})
        targets = pd.DataFrame({"m0": [100.0], "m1": [0.0]})
        country_mask = np.ones((1, n_records))
        national_metrics = pd.DataFrame({"n0": np.ones(n_records)})
        national_targets = pd.Series({"n0": 500.0})

        M, y, groups = _build_sparse_calibration_matrix(
            metrics, targets, country_mask, national_metrics, national_targets
        )

        # m1 has zero target, should be skipped
        # Should have: 1 local target (m0) + 1 national target
        assert len(y) == 2

    def test_group_ids_structure(self):
        """Group IDs should group by metric type."""
        n_records = 10
        metrics = pd.DataFrame({"m0": np.ones(n_records), "m1": np.ones(n_records)})
        targets = pd.DataFrame({"m0": [100.0, 200.0], "m1": [150.0, 250.0]})
        country_mask = np.ones((2, n_records))
        national_metrics = pd.DataFrame({"n0": np.ones(n_records)})
        national_targets = pd.Series({"n0": 500.0})

        M, y, groups = _build_sparse_calibration_matrix(
            metrics, targets, country_mask, national_metrics, national_targets
        )

        # Local targets: m0 has group 0, m1 has group 1
        # National target: n0 has group 2
        unique_groups = np.unique(groups)
        assert len(unique_groups) == 3


class TestL0CalibrationSmoke:
    """Smoke test with a tiny synthetic problem."""

    def test_l0_reduces_error(self):
        """L0 calibration should reduce target error vs uniform weights."""
        from l0.calibration import SparseCalibrationWeights

        n_records = 50
        n_targets = 5

        rng = np.random.default_rng(42)
        M_dense = rng.random((n_targets, n_records))
        M = sp.csr_matrix(M_dense)

        # True weights
        true_weights = rng.random(n_records) * 10
        y = M_dense @ true_weights

        # Fit L0 model
        model = SparseCalibrationWeights(
            n_features=n_records,
            init_weights=np.ones(n_records) * true_weights.mean(),
            init_keep_prob=0.9,
        )
        model.fit(
            M=M,
            y=y,
            lambda_l0=0.001,
            lambda_l2=0.0,
            lr=0.05,
            epochs=200,
            loss_type="relative",
        )

        # Check predictions
        import torch

        with torch.no_grad():
            y_pred = model.forward(M, deterministic=True).numpy()

        rel_errors = np.abs((y_pred - y) / (y + 1))
        mean_error = rel_errors.mean()

        # Should achieve reasonable accuracy (loose threshold for
        # stochastic optimisation across different platforms)
        assert mean_error < 0.2, f"Mean relative error {mean_error:.3f} too high"

    def test_sparsity_with_strong_l0(self):
        """Strong L0 penalty should produce sparse weights."""
        from l0.calibration import SparseCalibrationWeights

        n_records = 100
        n_targets = 3

        rng = np.random.default_rng(42)
        M = sp.csr_matrix(rng.random((n_targets, n_records)))
        y = rng.random(n_targets) * 1000

        model = SparseCalibrationWeights(
            n_features=n_records,
            init_keep_prob=0.5,
        )
        model.fit(
            M=M,
            y=y,
            lambda_l0=1.0,  # Strong L0
            epochs=500,
            loss_type="relative",
        )

        sparsity = model.get_sparsity()
        assert sparsity > 0.1, f"Sparsity {sparsity:.1%} too low with strong L0 penalty"
