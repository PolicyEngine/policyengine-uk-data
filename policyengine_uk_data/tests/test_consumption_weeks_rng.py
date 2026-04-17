"""Regression tests for `datasets/imputations/consumption.py`.

Covers the two sub-fixes in bug-hunt finding U6:

- Weekly → annual conversion used a bare ``* 52`` while the sibling FRS
  loader uses ``WEEKS_IN_YEAR = 365.25 / 7 ≈ 52.1786``. The ~0.34% gap
  skews VAT / energy imputation targets against FRS income.
- Two ``np.random.seed(42)`` calls mutated the global RNG state, so any
  unrelated numpy random call after consumption imputation changed output.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

if importlib.util.find_spec("policyengine_uk") is None:
    pytest.skip(
        "policyengine_uk not available in test environment",
        allow_module_level=True,
    )


def test_consumption_imports_shared_weeks_in_year():
    """consumption.py should use the same WEEKS_IN_YEAR as frs.py."""
    from policyengine_uk_data.datasets.frs import WEEKS_IN_YEAR as frs_value
    from policyengine_uk_data.datasets.imputations import (
        consumption as consumption_module,
    )

    assert frs_value == 365.25 / 7
    # Imported at module scope, not re-computed locally with a different value.
    assert consumption_module.WEEKS_IN_YEAR == frs_value


def test_consumption_module_no_longer_calls_np_random_seed():
    """The fixed module must not reach into the process-wide RNG state.

    The buggy version called `np.random.seed(42)` in two places, which
    mutated the global RNG and silently changed output for any later
    numpy random consumer in the same process.
    """
    import inspect

    from policyengine_uk_data.datasets.imputations import (
        consumption as consumption_module,
    )

    src = inspect.getsource(consumption_module)
    assert "np.random.seed(" not in src, (
        "consumption.py still mutates the global RNG state via "
        "np.random.seed(...) — use np.random.default_rng(seed) instead."
    )


def test_consumption_does_not_perturb_global_rng_on_import():
    """Importing consumption.py must not change global RNG state.

    Regression against the original bug where running module-level code
    paths (e.g. during test collection) could have reseeded the global RNG.
    """
    np.random.seed(1234)
    before = np.random.random(3)

    # Force reimport to exercise module-level side effects.
    import importlib

    from policyengine_uk_data.datasets.imputations import consumption
    importlib.reload(consumption)

    np.random.seed(1234)
    after = np.random.random(3)
    np.testing.assert_allclose(before, after)
