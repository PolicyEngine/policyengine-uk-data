"""Tests for calibrate_local_areas: weight priors, saving and logging.

Two save bugs are pinned here. The save block once sat outside the training
loop, so a 512-epoch run (511 % 10 = 1) never wrote the weight file. It then
saved only when epoch % 10 == 0, so a 512-epoch build shipped the state after
epoch 510 of 0-511 and dropped the final update (#456). A property test
checks, in both training loops, that the saved file, the returned weights and
the last logged row all describe the final optimiser state.
"""

from __future__ import annotations

import importlib.util
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st

if (
    importlib.util.find_spec("torch") is None
    or importlib.util.find_spec("policyengine_uk") is None
):
    pytest.skip(
        "torch/policyengine_uk not available in test environment",
        allow_module_level=True,
    )


def _make_toy_inputs(
    n_households: int = 4,
    area_count: int = 2,
    n_targets: int = 2,
    country_mask: np.ndarray | None = None,
):
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
    local_targets = pd.DataFrame(np.tile([100.0, 200.0][:n_targets], (area_count, 1)))
    # Country mask: by default every area includes every household.
    if country_mask is None:
        country_mask = np.ones((area_count, n_households), dtype=float)

    national_matrix = pd.DataFrame(np.ones((n_households, 1), dtype=float))
    national_targets = pd.Series([1000.0])

    def matrix_fn(_dataset):
        return matrix.copy(), local_targets.copy(), country_mask.copy()

    def national_matrix_fn(_dataset):
        return national_matrix.copy(), national_targets.copy()

    return matrix_fn, national_matrix_fn


# Two countries of two households each: area 0 holds households 0 and 1, area
# 1 holds 2 and 3. Its zeros make exp(parameters) differ from the saved
# weights, so a save that dropped the mask would fail.
TWO_COUNTRY_MASK = np.array([[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]])


def _national_performance(weights, _m_c, _y_c, m_n, y_n, _excluded_targets):
    """A one-row `get_performance` for the toy's single national target."""
    estimates = weights.sum(axis=0) @ m_n
    error = float(estimates.iloc[0] - y_n.iloc[0])
    return pd.DataFrame(
        {
            "name": ["UK"],
            "metric": ["national_total"],
            "estimate": [float(estimates.iloc[0])],
            "target": [float(y_n.iloc[0])],
            "error": [error],
            "abs_error": [abs(error)],
            "rel_abs_error": [abs(error) / float(y_n.iloc[0])],
            "validation": [False],
        }
    )


def _toy_loss(weights, matrix_fn, national_matrix_fn):
    """The calibrator's training loss without dropout, recomputed in float64."""

    def sre(x, y):
        return np.minimum(((1 + x) / (1 + y) - 1) ** 2, ((1 + y) / (1 + x) - 1) ** 2)

    weights = weights.astype(np.float64)
    matrix, local_targets, _ = matrix_fn(None)
    national_matrix, national_targets = national_matrix_fn(None)
    local = sre(weights @ matrix.to_numpy(), local_targets.to_numpy())
    national = sre(
        weights.sum(axis=0) @ national_matrix.to_numpy(), national_targets.to_numpy()
    )
    return local.mean() + national.mean()


class _StubDataset:
    """Minimal stand-in for a UKSingleYearDataset.

    `calibrate_local_areas` only touches `.household.household_weight` and
    calls `.copy()`. A small dataframe-backed stub is enough for this
    regression test.
    """

    def __init__(self, weights: np.ndarray, **household_columns):
        self.household = pd.DataFrame({"household_weight": weights.astype(float)})
        for column, values in household_columns.items():
            self.household[column] = values

    def copy(self) -> "_StubDataset":
        extra_columns = {
            column: self.household[column].to_numpy(copy=True)
            for column in self.household.columns
            if column != "household_weight"
        }
        copy = _StubDataset(
            self.household["household_weight"].to_numpy(),
            **extra_columns,
        )
        return copy


def test_initialize_weight_priors_gives_zero_weight_rows_balanced_mass():
    from policyengine_uk_data.utils.calibrate import initialize_weight_priors

    weights = np.array([1_500.0, 0.0, 625.0, 0.0], dtype=np.float64)

    priors = initialize_weight_priors(weights)

    assert np.all(priors > 0)
    assert priors.sum() == pytest.approx(weights.sum())
    assert priors[[0, 2]].sum() == pytest.approx(weights.sum() / 2)
    assert priors[[1, 3]].sum() == pytest.approx(weights.sum() / 2)
    assert priors[1] == pytest.approx(priors[3])
    assert priors[0] / priors[2] == pytest.approx(weights[0] / weights[2])


def test_initialize_weight_priors_preserves_positive_weights_exactly():
    from policyengine_uk_data.utils.calibrate import initialize_weight_priors

    weights = np.array([1_500.0, 400.0, 625.0], dtype=np.float64)

    priors = initialize_weight_priors(weights)

    np.testing.assert_array_equal(priors, weights)


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


def test_calibrate_local_areas_masks_nan_local_targets(tmp_path, monkeypatch):
    """Sparse local targets should be allowed.

    Local-authority sources are not available for every area/metric pair.
    A NaN target means "do not train on this cell", not "propagate NaN
    through the loss".
    """

    import h5py

    from policyengine_uk_data.utils import calibrate as calibrate_module
    from policyengine_uk_data.utils.calibrate import calibrate_local_areas

    monkeypatch.setattr(calibrate_module, "STORAGE_FOLDER", tmp_path)

    matrix_fn, national_matrix_fn = _make_toy_inputs(n_households=4, area_count=2)

    def sparse_matrix_fn(dataset):
        matrix, local_targets, country_mask = matrix_fn(dataset)
        local_targets.iloc[1, 0] = np.nan
        return matrix, local_targets, country_mask

    weight_file = "toy_sparse_weights.h5"
    calibrate_local_areas(
        dataset=_StubDataset(np.array([1.0, 1.0, 1.0, 1.0])),
        matrix_fn=sparse_matrix_fn,
        national_matrix_fn=national_matrix_fn,
        area_count=2,
        weight_file=weight_file,
        dataset_key="2025",
        epochs=5,
        verbose=False,
    )

    with h5py.File(tmp_path / weight_file, "r") as f:
        weights = f["2025"][:]
        assert np.isfinite(weights).all()


def test_calibrate_local_areas_logs_loss_targets_and_source_diagnostics(
    tmp_path, monkeypatch
):
    import h5py

    from policyengine_uk_data.utils import calibrate as calibrate_module
    from policyengine_uk_data.utils.calibrate import calibrate_local_areas

    monkeypatch.setattr(calibrate_module, "STORAGE_FOLDER", tmp_path)

    matrix_fn, national_matrix_fn = _make_toy_inputs(n_households=4, area_count=2)
    dataset = _StubDataset(
        np.array([4.0, 0.0, 4.0, 0.0]),
        household_is_spi_synthetic=[False, True, False, True],
    )

    weight_file = "toy_diagnostic_weights.h5"
    log_csv = tmp_path / "diagnostics.csv"
    calibrate_local_areas(
        dataset=dataset,
        matrix_fn=matrix_fn,
        national_matrix_fn=national_matrix_fn,
        area_count=2,
        weight_file=weight_file,
        dataset_key="2025",
        epochs=1,
        log_csv=log_csv,
        get_performance=_national_performance,
        verbose=False,
    )

    with h5py.File(tmp_path / weight_file, "r") as f:
        weights = f["2025"][:]
        assert weights[:, [1, 3]].sum() > 0

    diagnostics = pd.read_csv(log_csv)
    row = diagnostics.iloc[0]
    assert row["target_name"] == "UK/national_total"
    assert np.isfinite(row["loss"])
    assert np.isfinite(row["training_loss"])
    assert np.isfinite(row["saved_weights_loss"])
    assert row["initial_zero_weight_rows"] == 2
    assert row["initial_zero_weight_prior_share"] == pytest.approx(0.5)
    assert row["household_is_spi_synthetic_rows"] == 2
    assert row["household_is_spi_synthetic_prior_share"] == pytest.approx(0.5)
    assert row["household_is_spi_synthetic_household_weight"] > 0


def _final_state_run(inputs, epochs, verbose):
    """Calibrate the toy problem in a scratch folder.

    Returns the saved weights, the returned household weights, the final
    optimiser state (exp of the trained parameters times the country mask)
    and the performance log.
    """
    import h5py
    import torch

    from policyengine_uk_data.utils import calibrate as calibrate_module
    from policyengine_uk_data.utils.calibrate import calibrate_local_areas

    parameters = []

    class SpyAdam(torch.optim.Adam):
        def __init__(self, params, *args, **kwargs):
            params = list(params)
            parameters.extend(params)
            super().__init__(params, *args, **kwargs)

    matrix_fn, national_matrix_fn = inputs
    # Hypothesis reruns the test body, so it can't use pytest's
    # function-scoped tmp_path and monkeypatch fixtures.
    with (
        tempfile.TemporaryDirectory() as folder,
        pytest.MonkeyPatch.context() as monkeypatch,
    ):
        folder = Path(folder)
        monkeypatch.setattr(calibrate_module, "STORAGE_FOLDER", folder)
        monkeypatch.setattr(torch.optim, "Adam", SpyAdam)
        returned = calibrate_local_areas(
            dataset=_StubDataset(np.array([1.0, 2.0, 3.0, 4.0])),
            matrix_fn=matrix_fn,
            national_matrix_fn=national_matrix_fn,
            area_count=2,
            weight_file="final.h5",
            dataset_key="2025",
            epochs=epochs,
            log_csv=folder / "log.csv",
            get_performance=_national_performance,
            verbose=verbose,
        )
        with h5py.File(folder / "final.h5", "r") as f:
            saved = f["2025"][:]
        log = pd.read_csv(folder / "log.csv")
    (weights,) = parameters
    final = torch.exp(weights).detach().numpy() * TWO_COUNTRY_MASK
    return saved, returned.household["household_weight"].to_numpy(), final, log


@settings(max_examples=20, deadline=None)
@example(epochs=32, verbose=True)  # a TESTING build
@given(epochs=st.integers(1, 40), verbose=st.booleans())
def test_saved_returned_and_logged_weights_are_the_final_optimiser_state(
    epochs, verbose
):
    """#456: weights were saved only when epoch % 10 == 0, so a 512-epoch
    build shipped the state after epoch 510 of 0-511 and dropped the last
    update."""
    inputs = _make_toy_inputs(country_mask=TWO_COUNTRY_MASK)
    saved, returned, final, log = _final_state_run(inputs, epochs, verbose)
    np.testing.assert_array_equal(saved, final)
    np.testing.assert_allclose(returned, final.sum(axis=0), rtol=1e-6)
    # Every tenth epoch and the last are logged, so the last row is the
    # final epoch, and its saved_weights_loss is the saved weights' loss.
    assert log["epoch"].iloc[-1] == epochs - 1
    assert log["epoch"].tolist() == sorted({*range(0, epochs, 10), epochs - 1})
    assert log["saved_weights_loss"].iloc[-1] == pytest.approx(
        _toy_loss(saved, *inputs), rel=1e-5
    )
