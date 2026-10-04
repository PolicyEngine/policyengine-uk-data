"""Calibration regularisation: household weight bounds and the prior-drift
penalty."""

import numpy as np
import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from policyengine_uk_data.utils.calibrate import (
    bound_household_weights,
    prior_drift,
)


@st.composite
def _case(draw):
    areas = draw(st.integers(1, 6))
    households = draw(st.integers(1, 8))
    size = areas * households
    log_weights = draw(st.lists(st.floats(-8, 8), min_size=size, max_size=size))
    mask = draw(st.lists(st.booleans(), min_size=size, max_size=size))
    lower = draw(st.lists(st.floats(-6, 6), min_size=households, max_size=households))
    width = draw(st.lists(st.floats(0, 6), min_size=households, max_size=households))
    return (
        torch.tensor(log_weights, dtype=torch.float64).reshape(areas, households),
        torch.tensor(mask, dtype=torch.float64).reshape(areas, households),
        torch.tensor(lower, dtype=torch.float64),
        torch.tensor(lower, dtype=torch.float64)
        + torch.tensor(width, dtype=torch.float64),
    )


def _totals(log_weights, mask):
    return (torch.exp(log_weights) * mask).sum(dim=0)


@settings(max_examples=300, deadline=None)
@given(_case())
def test_bounds_hold_and_keep_area_shares(case):
    log_weights, mask, log_lower, log_upper = case
    before = log_weights.clone()
    totals_before = _totals(before, mask)
    bound_household_weights(log_weights, mask, log_lower, log_upper)
    totals_after = _totals(log_weights, mask)
    lower, upper = torch.exp(log_lower), torch.exp(log_upper)
    member = mask.sum(dim=0) > 0

    # Every household with an area ends inside its bounds.
    assert torch.all(totals_after[member] <= upper[member] * (1 + 1e-9))
    assert torch.all(totals_after[member] >= lower[member] * (1 - 1e-9))
    # A household already inside is untouched, up to float rounding at an
    # exact bound (log(exp(x)) need not return x).
    inside = (totals_before >= lower) & (totals_before <= upper)
    torch.testing.assert_close(
        log_weights[:, inside], before[:, inside], rtol=0, atol=1e-12
    )
    # A rescaled household moves by one factor in every area.
    shift = log_weights - before
    torch.testing.assert_close(shift, shift[:1].expand_as(shift), rtol=0, atol=1e-9)


def test_either_bound_can_be_absent():
    log_weights = torch.log(torch.tensor([[5.0, 0.5]], dtype=torch.float64))
    mask = torch.ones_like(log_weights)
    bound_household_weights(log_weights, mask, None, torch.zeros(2))
    np.testing.assert_allclose(torch.exp(log_weights).numpy(), [[1.0, 0.5]])
    bound_household_weights(log_weights, mask, torch.zeros(2), None)
    np.testing.assert_allclose(torch.exp(log_weights).numpy(), [[1.0, 1.0]])


def test_masked_areas_do_not_count_towards_the_total():
    # Two areas; the household belongs only to the first.
    log_weights = torch.log(torch.tensor([[2.0], [1_000.0]], dtype=torch.float64))
    mask = torch.tensor([[1.0], [0.0]], dtype=torch.float64)
    bound_household_weights(log_weights, mask, None, torch.log(torch.tensor([1.0])))
    assert torch.exp(log_weights[0, 0]).item() == pytest.approx(1.0)


@settings(max_examples=200, deadline=None)
@given(
    st.lists(st.floats(0.01, 1e4), min_size=1, max_size=12),
    st.lists(st.floats(0.01, 1e4), min_size=12, max_size=12),
)
def test_prior_drift_matches_its_definition_and_falls_under_gradient(totals, priors):
    totals = np.array(totals)
    priors = np.array(priors[: len(totals)])
    weights = torch.tensor(totals[None, :], dtype=torch.float64, requires_grad=True)
    log_prior = torch.tensor(np.log(priors), dtype=torch.float64)
    drift = prior_drift(weights, log_prior)
    assert drift.item() == pytest.approx(np.mean(np.log(totals / priors) ** 2))
    # Zero at the prior.
    at_prior = torch.tensor(priors[None, :], dtype=torch.float64)
    assert prior_drift(at_prior, log_prior).item() == pytest.approx(0, abs=1e-12)
    # A small step against the gradient in log space lowers it.
    log_w = torch.log(
        torch.tensor(totals[None, :], dtype=torch.float64)
    ).requires_grad_()
    value = prior_drift(torch.exp(log_w), log_prior)
    value.backward()
    stepped = prior_drift(torch.exp(log_w - 0.01 * log_w.grad), log_prior)
    assert stepped.item() <= value.item() + 1e-12


def test_calibration_respects_bounds_and_rejects_inverted_ones(tmp_path, monkeypatch):
    from policyengine_uk_data.tests.test_calibrate_save import (
        _StubDataset,
        _make_toy_inputs,
    )
    from policyengine_uk_data.utils import calibrate as calibrate_module
    from policyengine_uk_data.utils.calibrate import calibrate_local_areas

    monkeypatch.setattr(calibrate_module, "STORAGE_FOLDER", tmp_path)
    matrix_fn, national_matrix_fn = _make_toy_inputs(n_households=4, area_count=2)
    priors = np.array([1.0, 1.0, 1.0, 1.0])
    kwargs = dict(
        matrix_fn=matrix_fn,
        national_matrix_fn=national_matrix_fn,
        area_count=2,
        weight_file="toy_weights.h5",
        epochs=50,
        prior_drift_penalty=0.1,
    )
    result = calibrate_local_areas(
        dataset=_StubDataset(priors),
        min_weight_ratio=0.5,
        max_weight_ratio=3.0,
        **kwargs,
    )
    totals = result.household.household_weight.to_numpy()
    # The toy targets want totals near 100-1000; the bounds keep each one
    # within 0.5-3x of its prior instead.
    assert np.all(totals <= 3.0 * priors * (1 + 1e-5))
    assert np.all(totals >= 0.5 * priors * (1 - 1e-5))
    with pytest.raises(ValueError):
        calibrate_local_areas(
            dataset=_StubDataset(priors),
            min_weight_ratio=2.0,
            max_weight_ratio=1.0,
            **kwargs,
        )
