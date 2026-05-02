from __future__ import annotations

import io

import numpy as np
import pandas as pd

from policyengine_uk_data.datasets.create_datasets import (
    _modal_calibration_requested,
    _prepare_modal_calibration_payload,
)


class _DummyValues:
    def __init__(self, values):
        self.values = np.array(values, dtype=float)


class _DummyHousehold:
    def __init__(self, weights):
        self.household_weight = _DummyValues(weights)


class _DummyDataset:
    def __init__(self, weights):
        self.household = _DummyHousehold(weights)

    def copy(self):
        return _DummyDataset(self.household.household_weight.values.copy())


def _load_payload_array(payload, key):
    return np.load(io.BytesIO(payload[key]))


def test_modal_calibration_requested_requires_explicit_flag(monkeypatch):
    monkeypatch.delenv("MODAL_CALIBRATE", raising=False)
    assert not _modal_calibration_requested()

    monkeypatch.setenv("MODAL_CALIBRATE", "0")
    assert not _modal_calibration_requested()

    monkeypatch.setenv("MODAL_CALIBRATE", "1")
    assert _modal_calibration_requested()


def test_modal_payload_masks_nan_local_targets():
    dataset = _DummyDataset([10.0, 20.0, 30.0])
    national_matrix = np.ones((3, 1), dtype=float)
    national_targets = np.array([100.0], dtype=float)

    def matrix_fn(_dataset):
        matrix = pd.DataFrame(
            {
                "target_a": [1.0, 0.0, 1.0],
                "target_b": [0.0, 1.0, 1.0],
            }
        )
        targets = pd.DataFrame(
            {
                "target_a": [5.0, np.nan],
                "target_b": [7.0, 11.0],
            }
        )
        country_mask = np.ones((2, 3), dtype=float)
        return matrix, targets, country_mask

    payload, (_, logged_targets) = _prepare_modal_calibration_payload(
        dataset=dataset,
        matrix_fn=matrix_fn,
        area_count=2,
        m_national_bytes=_array_bytes(national_matrix),
        y_national_bytes=_array_bytes(national_targets),
    )

    local_targets = _load_payload_array(payload, "y")
    local_target_available = _load_payload_array(payload, "local_target_available")

    assert local_targets.tolist() == [[5.0, 7.0], [0.0, 11.0]]
    assert local_target_available.tolist() == [[True, True], [False, True]]
    assert np.isnan(logged_targets.iloc[1, 0])


def _array_bytes(value):
    buffer = io.BytesIO()
    np.save(buffer, value)
    return buffer.getvalue()
