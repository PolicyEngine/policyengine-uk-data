"""Unit tests for the rent/mortgage rescale factor helper in income.py.

Guards the zero-division bug reported in the bug hunt (finding U3):
`impute_over_incomes` computed ``new_income_total / original_income_total``
with no check for the degenerate case where the seed dataset had zero in
every imputation column — which is exactly the shape of the
`zero_weight_copy` branch inside `impute_income`.
"""

from __future__ import annotations

import math

import pytest


def test_safe_rescale_factor_with_zero_original_returns_one():
    from policyengine_uk_data.datasets.imputations.income import (
        _safe_rescale_factor,
    )

    # The bug: dividing by zero raised ZeroDivisionError (or produced inf).
    # The fix: leave housing costs untouched when we have no baseline.
    assert _safe_rescale_factor(0, 123_456) == 1.0
    assert _safe_rescale_factor(0.0, 0.0) == 1.0


def test_safe_rescale_factor_with_nonzero_original_returns_ratio():
    from policyengine_uk_data.datasets.imputations.income import (
        _safe_rescale_factor,
    )

    assert _safe_rescale_factor(1_000.0, 2_500.0) == pytest.approx(2.5)
    assert _safe_rescale_factor(42.0, 42.0) == pytest.approx(1.0)


def test_safe_rescale_factor_preserves_finiteness():
    from policyengine_uk_data.datasets.imputations.income import (
        _safe_rescale_factor,
    )

    # Non-zero inputs must still return finite floats.
    for original, new in [(1e9, 2e9), (1e-6, 1e-9), (100.0, 0.0)]:
        factor = _safe_rescale_factor(original, new)
        assert math.isfinite(factor), (original, new, factor)
