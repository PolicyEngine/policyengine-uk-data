"""Calibration gives the same weights for the same inputs and seed.

Invariants of ``calibrate_local_areas``:

- the saved weights are a function of the inputs and ``seed`` alone: not of
  the process, and not of whatever has consumed torch's global generator;
- calibration does not consume the global generator;
- ``seed`` is used: different seeds draw different dropout masks.

Before the dropout masks had their own generator, two builds of one commit
differed in every household weight, because torch seeds its global generator
differently in each process.
"""

from __future__ import annotations

import hashlib
import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

if (
    importlib.util.find_spec("torch") is None
    or importlib.util.find_spec("policyengine_uk") is None
):
    pytest.skip(
        "torch/policyengine_uk not available in test environment",
        allow_module_level=True,
    )

import h5py
import torch

from policyengine_uk_data.tests.test_calibrate_save import (
    _StubDataset,
    _make_toy_inputs,
)
from policyengine_uk_data.utils import calibrate as calibrate_module
from policyengine_uk_data.utils.calibrate import calibrate_local_areas


def _toy_weights(**kwargs) -> np.ndarray:
    """Area-by-household weights saved by a short toy calibration.

    120 weights and 21 epochs, so dropout (5% per weight per epoch) acts
    many times and the last save (epoch 20) reflects it.
    """
    matrix_fn, national_matrix_fn = _make_toy_inputs(n_households=40, area_count=3)
    dataset = _StubDataset(np.linspace(1.0, 5.0, 40))
    with tempfile.TemporaryDirectory() as folder:
        with patch.object(calibrate_module, "STORAGE_FOLDER", Path(folder)):
            calibrate_local_areas(
                dataset=dataset,
                matrix_fn=matrix_fn,
                national_matrix_fn=national_matrix_fn,
                area_count=3,
                weight_file="weights.h5",
                dataset_key="2025",
                epochs=21,
                verbose=False,
                **kwargs,
            )
        with h5py.File(Path(folder) / "weights.h5", "r") as f:
            return f["2025"][:]


def _toy_weights_digest() -> str:
    return hashlib.sha256(_toy_weights().tobytes()).hexdigest()


@pytest.mark.parametrize("seed", [0, 1, 2**31 - 1])
@pytest.mark.parametrize("global_state, other_global_state", [(0, 1), (7, 12345)])
def test_weights_depend_on_seed_not_on_the_global_generator(
    seed, global_state, other_global_state
):
    torch.manual_seed(global_state)
    first = _toy_weights(seed=seed)
    torch.manual_seed(other_global_state)
    torch.rand(7)
    second = _toy_weights(seed=seed)
    assert np.array_equal(first, second)


def test_calibration_does_not_consume_the_global_generator():
    torch.manual_seed(123)
    expected = torch.rand(5)
    torch.manual_seed(123)
    _toy_weights()
    assert torch.equal(torch.rand(5), expected)


def test_different_seeds_give_different_weights():
    assert not np.array_equal(_toy_weights(seed=0), _toy_weights(seed=1))


def test_fresh_processes_give_identical_weights():
    """The failure this guards against only shows across processes."""
    command = [
        sys.executable,
        "-c",
        "from policyengine_uk_data.tests.test_calibrate_determinism import "
        "_toy_weights_digest; print(_toy_weights_digest())",
    ]
    digests = [
        subprocess.run(command, capture_output=True, text=True, check=True)
        .stdout.strip()
        .splitlines()[-1]
        for _ in range(2)
    ]
    assert digests[0] == digests[1] == _toy_weights_digest()
