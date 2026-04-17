"""Regression test for calibrate_local_areas weight saving.

Ensures that the non-verbose branch of `calibrate_local_areas` saves the
weight file to disk — previously the save block was indented outside the
for-loop, so on a typical 512-epoch run (511 % 10 = 1) no save ever ran
and the output .h5 file was never written.
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


def _make_toy_inputs(n_households: int = 4, area_count: int = 2, n_targets: int = 2):
    """Build minimal matrix/target/mask arrays for `calibrate_local_areas`.

    The calibration routine treats `matrix` as (n_households, n_targets),
    the local targets array as (n_areas, n_targets), the national matrix
    as (n_households, n_national_targets), and the country mask `r` as
    (n_areas, n_households). We construct small deterministic inputs so
    the routine can run for a handful of epochs with no real survey data.
    """

    # Household contribution matrix: each household contributes 1 to the
    # target with matching index (rest zero).
    matrix = pd.DataFrame(np.eye(n_households, n_targets, dtype=float))
    # Per-area local targets — identical rows are fine for a smoke test.
    local_targets = pd.DataFrame(
        np.tile([100.0, 200.0][:n_targets], (area_count, 1))
    )
    # Country mask: every area includes every household (simple case).
    country_mask = np.ones((area_count, n_households), dtype=float)

    national_matrix = pd.DataFrame(np.ones((n_households, 1), dtype=float))
    national_targets = pd.Series([1000.0])

    def matrix_fn(_dataset):
        return matrix.copy(), local_targets.copy(), country_mask.copy()

    def national_matrix_fn(_dataset):
        return national_matrix.copy(), national_targets.copy()

    return matrix_fn, national_matrix_fn


class _StubDataset:
    """Minimal stand-in for a UKSingleYearDataset.

    `calibrate_local_areas` only touches `.household.household_weight` and
    calls `.copy()`. A small dataframe-backed stub is enough for this
    regression test.
    """

    def __init__(self, weights: np.ndarray):
        self.household = pd.DataFrame({"household_weight": weights.astype(float)})

    def copy(self) -> "_StubDataset":
        copy = _StubDataset(self.household["household_weight"].to_numpy())
        return copy


def test_calibrate_local_areas_saves_weights_in_nonverbose_branch(
    tmp_path, monkeypatch
):
    """Non-verbose calibration must write the weights h5 file to disk.

    Regression: the `if epoch % 10 == 0` save block was indented outside the
    `for epoch in range(epochs):` loop, so for epochs=1 (or the default 512)
    the save block either ran once at the end with an unrelated `epoch`
    value or never ran at all. This test fails without the indentation fix.
    """

    import h5py

    from policyengine_uk_data.utils import calibrate as calibrate_module
    from policyengine_uk_data.utils.calibrate import calibrate_local_areas

    # Redirect STORAGE_FOLDER so the weight file lands in tmp_path rather
    # than the real package storage directory.
    monkeypatch.setattr(calibrate_module, "STORAGE_FOLDER", tmp_path)

    matrix_fn, national_matrix_fn = _make_toy_inputs(n_households=4, area_count=2)
    dataset = _StubDataset(np.array([1.0, 1.0, 1.0, 1.0]))

    weight_file = "toy_weights.h5"
    # epochs=5 is deliberate: with the broken indentation the save block
    # runs only once at the end with epoch=epochs-1=4, and 4 % 10 != 0 so
    # no save happens. With the fix in place the save fires at epoch=0
    # (inside the loop) and the file exists.
    calibrate_local_areas(
        dataset=dataset,
        matrix_fn=matrix_fn,
        national_matrix_fn=national_matrix_fn,
        area_count=2,
        weight_file=weight_file,
        dataset_key="2025",
        epochs=5,
        verbose=False,
    )

    assert (tmp_path / weight_file).exists(), (
        "calibrate_local_areas did not write the weight file — the save "
        "block is almost certainly outside the training loop again."
    )

    with h5py.File(tmp_path / weight_file, "r") as f:
        assert "2025" in f, "dataset_key not written to h5 file"
        weights = f["2025"][:]
        # Verify the saved weights have the area_count x n_households shape
        # produced by the calibrator.
        assert weights.shape == (2, 4)
