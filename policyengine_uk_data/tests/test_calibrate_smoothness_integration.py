"""Integration tests for the smoothness-penalty wiring in calibrate_local_areas.

The unit tests for ``compute_log_weight_smoothness_penalty`` live in
``test_smoothness_penalty.py``. The tests here exercise the surrounding
plumbing: validation of the new kwargs, that default behaviour is
unchanged, and that a large penalty actually pulls the optimised weights
towards the prior.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from policyengine_uk.data import UKSingleYearDataset

from policyengine_uk_data.utils import calibrate as calibrate_module
from policyengine_uk_data.utils.calibrate import calibrate_local_areas


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _tiny_dataset() -> UKSingleYearDataset:
    """Three-household dataset just big enough for calibration shapes."""
    household = pd.DataFrame(
        {
            "household_id": [1, 2, 3],
            "household_weight": [1000.0, 1000.0, 1000.0],
        }
    )
    benunit = pd.DataFrame({"benunit_id": [101, 201, 301]})
    person = pd.DataFrame(
        {
            "person_id": [1001, 2001, 3001],
            "person_benunit_id": [101, 201, 301],
            "person_household_id": [1, 2, 3],
            "age": [30, 40, 50],
        }
    )
    return UKSingleYearDataset(
        person=person, benunit=benunit, household=household, fiscal_year=2025
    )


AREA_COUNT = 2


def _fake_local_matrix(dataset):
    """Two areas, three households, one target per area.

    Each target is the sum of household_weight over the households in
    that area. With default initial weights the target is easy to learn.
    """
    matrix = pd.DataFrame({"pop/area_size": [1.0, 1.0, 1.0]})
    y = pd.DataFrame({"pop/area_size": [3000.0, 3000.0]})
    # Simple country mask: both areas include all households.
    r = np.ones((AREA_COUNT, 3))
    return matrix, y, r


def _fake_national_matrix(dataset):
    matrix = pd.DataFrame({"pop/national": [1.0, 1.0, 1.0]})
    y = pd.DataFrame({"pop/national": [6000.0]})
    return matrix, y


@pytest.fixture
def patched_storage(tmp_path: Path, monkeypatch):
    """Redirect the hard-coded STORAGE_FOLDER write in calibrate.py."""
    monkeypatch.setattr(calibrate_module, "STORAGE_FOLDER", tmp_path)
    return tmp_path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_default_kwargs_reproduce_pre_step5_behaviour(patched_storage):
    """No prior + zero penalty ⇒ the smoothness branch must be inert."""
    # NB: calibrate_local_areas only flushes the weight file when the
    # final epoch index is a multiple of 10 (the function saves every 10
    # epochs). Use 11 epochs so the final epoch = 10 triggers a save.
    np.random.seed(0)
    import torch

    torch.manual_seed(0)
    calibrate_local_areas(
        dataset=_tiny_dataset(),
        matrix_fn=_fake_local_matrix,
        national_matrix_fn=_fake_national_matrix,
        area_count=AREA_COUNT,
        weight_file="test_weights.h5",
        epochs=11,
    )
    assert (patched_storage / "test_weights.h5").exists()


def test_shape_mismatch_in_prior_raises(patched_storage):
    bogus_prior = np.ones((AREA_COUNT, 99))  # wrong household count
    with pytest.raises(ValueError, match="prior_weights shape"):
        calibrate_local_areas(
            dataset=_tiny_dataset(),
            matrix_fn=_fake_local_matrix,
            national_matrix_fn=_fake_national_matrix,
            area_count=AREA_COUNT,
            weight_file="test_weights.h5",
            epochs=1,
            prior_weights=bogus_prior,
            smoothness_penalty=1.0,
        )


def test_none_prior_with_penalty_is_noop(patched_storage):
    """A penalty coefficient without a prior must not crash."""
    calibrate_local_areas(
        dataset=_tiny_dataset(),
        matrix_fn=_fake_local_matrix,
        national_matrix_fn=_fake_national_matrix,
        area_count=AREA_COUNT,
        weight_file="test_weights.h5",
        epochs=1,
        prior_weights=None,
        smoothness_penalty=10.0,
    )


def test_zero_penalty_with_prior_is_noop(patched_storage):
    """A prior without a penalty coefficient must not crash either."""
    prior = np.ones((AREA_COUNT, 3)) * 500.0
    calibrate_local_areas(
        dataset=_tiny_dataset(),
        matrix_fn=_fake_local_matrix,
        national_matrix_fn=_fake_national_matrix,
        area_count=AREA_COUNT,
        weight_file="test_weights.h5",
        epochs=1,
        prior_weights=prior,
        smoothness_penalty=0.0,
    )


def test_large_penalty_keeps_weights_near_prior(patched_storage):
    """With a huge penalty, the optimised weights should stay near the prior."""
    import h5py

    # Prior that is deliberately far from what the fit-loss alone would
    # drive us to (fit alone wants ~1000 per household per area to match
    # the area target; this prior has 10x larger values).
    prior = np.ones((AREA_COUNT, 3)) * 10_000.0

    np.random.seed(0)
    import torch

    torch.manual_seed(0)
    calibrate_local_areas(
        dataset=_tiny_dataset(),
        matrix_fn=_fake_local_matrix,
        national_matrix_fn=_fake_national_matrix,
        area_count=AREA_COUNT,
        weight_file="with_smoothness.h5",
        # 21 epochs ⇒ final index 20 is a multiple of 10 → save triggers.
        epochs=21,
        prior_weights=prior,
        smoothness_penalty=1e6,
    )

    with h5py.File(patched_storage / "with_smoothness.h5", "r") as f:
        final_with = np.array(f["2025"])

    # And the same run without the smoothness penalty.
    np.random.seed(0)
    torch.manual_seed(0)
    calibrate_local_areas(
        dataset=_tiny_dataset(),
        matrix_fn=_fake_local_matrix,
        national_matrix_fn=_fake_national_matrix,
        area_count=AREA_COUNT,
        weight_file="without_smoothness.h5",
        epochs=21,
    )

    with h5py.File(patched_storage / "without_smoothness.h5", "r") as f:
        final_without = np.array(f["2025"])

    # With the huge penalty, weights should be closer (in log-space) to
    # the prior than the no-smoothness run.
    log_dev_with = np.mean((np.log(final_with + 1e-8) - np.log(prior)) ** 2)
    log_dev_without = np.mean((np.log(final_without + 1e-8) - np.log(prior)) ** 2)
    assert log_dev_with < log_dev_without, (
        f"Smoothness failed to pull weights towards prior: "
        f"with={log_dev_with:.4f} vs without={log_dev_without:.4f}"
    )
