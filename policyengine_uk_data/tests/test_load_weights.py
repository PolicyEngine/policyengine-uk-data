"""Tests for `policyengine_uk_data.utils.calibrate.load_weights`.

Adds a defensive loader that normalises shape across the two calibrator
back-ends that have lived in this module (2D L2 and flat L0), so downstream
consumers cannot silently read the wrong axis layout (bug-hunt finding U4).
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

if importlib.util.find_spec("h5py") is None:
    pytest.skip("h5py not installed", allow_module_level=True)

import h5py  # noqa: E402


def _write_h5(path, key: str, data: np.ndarray):
    with h5py.File(path, "w") as f:
        f.create_dataset(key, data=data)


def test_load_weights_returns_2d_for_2d_input(tmp_path):
    from policyengine_uk_data.utils.calibrate import load_weights

    weights = np.arange(6, dtype=float).reshape(2, 3)
    path = tmp_path / "w.h5"
    _write_h5(path, "2025", weights)

    out = load_weights(path, dataset_key="2025")
    assert out.shape == (2, 3)
    np.testing.assert_allclose(out, weights)


def test_load_weights_promotes_1d_input_to_2d(tmp_path):
    from policyengine_uk_data.utils.calibrate import load_weights

    flat = np.arange(4, dtype=float)
    path = tmp_path / "w.h5"
    _write_h5(path, "2025", flat)

    out = load_weights(path, dataset_key="2025")
    # Flat inputs become (1, n_records) so .sum(axis=0) still yields the
    # same vector and downstream matrix ops stay consistent.
    assert out.shape == (1, 4)
    np.testing.assert_allclose(out[0], flat)


def test_load_weights_checks_expected_shapes(tmp_path):
    from policyengine_uk_data.utils.calibrate import load_weights

    weights = np.ones((3, 5), dtype=float)
    path = tmp_path / "w.h5"
    _write_h5(path, "2025", weights)

    # Correct expected dims → no exception.
    load_weights(path, dataset_key="2025", n_areas=3, n_records=5)

    with pytest.raises(ValueError, match="areas"):
        load_weights(path, dataset_key="2025", n_areas=4, n_records=5)
    with pytest.raises(ValueError, match="records"):
        load_weights(path, dataset_key="2025", n_areas=3, n_records=999)


def test_load_weights_missing_key_raises(tmp_path):
    from policyengine_uk_data.utils.calibrate import load_weights

    weights = np.ones((2, 2), dtype=float)
    path = tmp_path / "w.h5"
    _write_h5(path, "2025", weights)

    with pytest.raises(KeyError, match="not found"):
        load_weights(path, dataset_key="2099")


def test_load_weights_rejects_higher_dim_input(tmp_path):
    from policyengine_uk_data.utils.calibrate import load_weights

    weights = np.ones((2, 2, 2), dtype=float)
    path = tmp_path / "w.h5"
    _write_h5(path, "2025", weights)

    with pytest.raises(ValueError, match="1D or 2D"):
        load_weights(path, dataset_key="2025")
