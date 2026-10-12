"""Validation-only targets in calibrate_local_areas.

create_datasets passes VALIDATION_ONLY_LOCAL_TARGETS, which exist only in the
local matrices. The calibration must then train on the other targets, leave
the excluded one out of training, and log a finite validation loss (an empty
national validation set used to make it NaN).
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd
import pytest

if (
    importlib.util.find_spec("torch") is None
    or importlib.util.find_spec("policyengine_uk") is None
):
    pytest.skip(
        "torch/policyengine_uk not available in test environment",
        allow_module_level=True,
    )

from policyengine_uk_data.tests.test_calibrate_save import _StubDataset


def _performance(weights, _m_c, _y_c, m_n, y_n, _excluded_targets):
    estimate = float((weights.sum(axis=0) @ m_n).iloc[0])
    target = float(y_n.iloc[0])
    return pd.DataFrame(
        {
            "name": ["UK"],
            "metric": ["national_total"],
            "estimate": [estimate],
            "target": [target],
            "error": [estimate - target],
            "abs_error": [abs(estimate - target)],
            "rel_abs_error": [abs(estimate - target) / target],
            "validation": [False],
        }
    )


def _calibrate(tmp_path, monkeypatch, excluded, target_b):
    import h5py
    import torch

    from policyengine_uk_data.utils import calibrate as calibrate_module
    from policyengine_uk_data.utils.calibrate import calibrate_local_areas

    monkeypatch.setattr(calibrate_module, "STORAGE_FOLDER", tmp_path)
    # Household 0 feeds local target "a", household 1 feeds "b".
    matrix = pd.DataFrame({"a": [1.0, 0.0], "b": [0.0, 1.0]})
    local = pd.DataFrame({"a": [100.0], "b": [target_b]})
    national = pd.DataFrame({"national_total": [1.0, 1.0]})

    log_csv = tmp_path / f"log_{len(excluded)}_{target_b:.0f}.csv"
    # Weight dropout is random; seed it so runs differ only by their inputs.
    torch.manual_seed(0)
    calibrate_local_areas(
        dataset=_StubDataset(np.array([50.0, 50.0])),
        matrix_fn=lambda _d: (matrix.copy(), local.copy(), np.ones((1, 2))),
        national_matrix_fn=lambda _d: (national.copy(), pd.Series([200.0])),
        area_count=1,
        weight_file=f"weights_{len(excluded)}_{target_b:.0f}.h5",
        dataset_key="2025",
        epochs=21,
        excluded_training_targets=excluded,
        log_csv=log_csv,
        get_performance=_performance,
        verbose=True,
    )
    with h5py.File(tmp_path / f"weights_{len(excluded)}_{target_b:.0f}.h5") as f:
        weights = f["2025"][:]
    return weights, pd.read_csv(log_csv)


def test_local_only_validation_target_logs_finite_validation_loss(
    tmp_path, monkeypatch
):
    _, log = _calibrate(tmp_path, monkeypatch, ["b"], 100.0)
    assert np.isfinite(log["validation_loss"]).all()


def test_validation_only_target_does_not_train(tmp_path, monkeypatch):
    # Changing an excluded target's value leaves the weights unchanged;
    # changing it when it trains moves them.
    excluded_low, _ = _calibrate(tmp_path, monkeypatch, ["b"], 100.0)
    excluded_high, _ = _calibrate(tmp_path, monkeypatch, ["b"], 10_000.0)
    np.testing.assert_allclose(excluded_low, excluded_high)
    trained_low, _ = _calibrate(tmp_path, monkeypatch, [], 100.0)
    trained_high, _ = _calibrate(tmp_path, monkeypatch, [], 10_000.0)
    assert not np.allclose(trained_low, trained_high)
