import time

import numpy as np
import pandas as pd

from policyengine_uk_data.utils.calibrate import calibrate_local_areas


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


def test_calibrate_local_areas_logs_setup_stage_heartbeats_in_ci(
    monkeypatch, capsys, tmp_path
):
    monkeypatch.setenv("CI", "true")
    monkeypatch.setenv("POLICYENGINE_PROGRESS_HEARTBEAT_SECONDS", "0.01")
    monkeypatch.setattr(
        "policyengine_uk_data.utils.calibrate.STORAGE_FOLDER",
        tmp_path,
    )

    dataset = _DummyDataset([10.0, 20.0, 30.0])

    def matrix_fn(_dataset):
        time.sleep(0.03)
        matrix = pd.DataFrame({"metric": [1.0, 0.0, 1.0]})
        targets = pd.DataFrame({"metric": [2.0]})
        mask = np.ones((1, 3))
        return matrix, targets, mask

    def national_matrix_fn(_dataset):
        time.sleep(0.03)
        matrix = pd.DataFrame({"national_metric": [1.0, 1.0, 1.0]})
        targets = pd.Series({"national_metric": 3.0})
        return matrix, targets

    calibrate_local_areas(
        dataset=dataset,
        matrix_fn=matrix_fn,
        national_matrix_fn=national_matrix_fn,
        area_count=1,
        weight_file="weights.h5",
        epochs=1,
        verbose=True,
        area_name="Constituency",
    )

    output = capsys.readouterr().out
    assert "[calibration] starting: Constituency: build local target matrix" in output
    assert "[calibration] heartbeat: Constituency: build local target matrix" in output
    assert "[calibration] completed: Constituency: build local target matrix" in output
    assert (
        "[calibration] starting: Constituency: build national target matrix" in output
    )
    assert (
        "[calibration] heartbeat: Constituency: build national target matrix" in output
    )
    assert (
        "[calibration] completed: Constituency: build national target matrix" in output
    )
    assert "[calibration] epoch 1/1: calculating loss" in output
