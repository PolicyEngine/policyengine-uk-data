"""Tests for the cross-year smoothness penalty (step 5 of #345)."""

import pytest
import torch

from policyengine_uk_data.utils.calibrate import (
    compute_log_weight_smoothness_penalty,
)


def test_zero_when_log_weights_match_log_prior():
    """If current weights already equal the prior, the penalty is zero."""
    prior = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    log_weights = torch.log(prior)
    penalty = compute_log_weight_smoothness_penalty(log_weights, prior)
    assert penalty.item() == pytest.approx(0.0)


def test_penalty_scales_with_squared_log_deviation():
    """A log-ratio of ln(2) on every entry → penalty = (ln 2)**2."""
    prior = torch.ones(3, 4)
    # log_weights = log(2 * prior) = log(2)
    log_weights = torch.full((3, 4), float(torch.log(torch.tensor(2.0))))
    penalty = compute_log_weight_smoothness_penalty(log_weights, prior)
    assert penalty.item() == pytest.approx(
        float(torch.log(torch.tensor(2.0))) ** 2, rel=1e-6
    )


def test_zero_prior_entries_are_excluded_from_mean():
    """Households outside an area's country (prior == 0) must not inflate the penalty."""
    prior = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    log_weights = torch.zeros_like(prior)  # log(1) on the valid entries
    penalty = compute_log_weight_smoothness_penalty(log_weights, prior)
    # Only two entries are valid and both match the prior → penalty is zero.
    assert penalty.item() == pytest.approx(0.0)


def test_zero_prior_entries_do_not_pull_gradient():
    """Gradient w.r.t. a masked-out entry must be exactly zero."""
    prior = torch.tensor([[0.0, 2.0]])
    log_weights = torch.tensor([[100.0, 0.0]], requires_grad=True)
    penalty = compute_log_weight_smoothness_penalty(log_weights, prior)
    penalty.backward()
    # First entry is masked out → grad should be zero regardless of value.
    assert log_weights.grad[0, 0].item() == pytest.approx(0.0)
    # Second entry pulled towards log(2).
    assert log_weights.grad[0, 1].item() != 0.0


def test_all_zero_prior_returns_zero_without_nan():
    """No valid entries → zero, not NaN."""
    prior = torch.zeros(2, 2)
    log_weights = torch.randn(2, 2)
    penalty = compute_log_weight_smoothness_penalty(log_weights, prior)
    assert penalty.item() == 0.0
    assert not torch.isnan(penalty)


def test_shape_mismatch_raises_valueerror():
    prior = torch.ones(3, 4)
    log_weights = torch.zeros(3, 5)
    with pytest.raises(ValueError, match="shape"):
        compute_log_weight_smoothness_penalty(log_weights, prior)


def test_symmetric_log_deviation():
    """Doubling the prior and halving it produce the same penalty magnitude."""
    prior = torch.ones(2, 2)
    log_weights_double = torch.full((2, 2), float(torch.log(torch.tensor(2.0))))
    log_weights_half = torch.full((2, 2), -float(torch.log(torch.tensor(2.0))))
    a = compute_log_weight_smoothness_penalty(log_weights_double, prior)
    b = compute_log_weight_smoothness_penalty(log_weights_half, prior)
    assert a.item() == pytest.approx(b.item())


def test_penalty_is_differentiable():
    """The result must carry a grad so Adam can actually use it."""
    prior = torch.ones(2, 3)
    log_weights = torch.randn(2, 3, requires_grad=True)
    penalty = compute_log_weight_smoothness_penalty(log_weights, prior)
    assert penalty.requires_grad
    penalty.backward()
    assert log_weights.grad is not None
    # Some entry must see a non-zero gradient for a non-trivial prior.
    assert torch.any(log_weights.grad != 0)


def test_device_and_dtype_round_trip():
    """The output dtype matches the log_weights dtype (not the prior's)."""
    prior = torch.ones(2, 2, dtype=torch.float32)
    log_weights = torch.zeros(2, 2, dtype=torch.float64)
    penalty = compute_log_weight_smoothness_penalty(log_weights, prior)
    assert penalty.dtype == torch.float64


def test_heterogeneous_mask_and_values():
    """Explicit hand-computed example to lock in the arithmetic."""
    # prior = [[1, 0], [4, e]]  ⇒  valid entries are (0,0), (1,0), (1,1).
    e = float(torch.e)
    prior = torch.tensor([[1.0, 0.0], [4.0, e]])
    # log_weights = [[0, any], [0, 0]]  ⇒  deviations on valid entries
    # are: (0 - log 1)=0,  (0 - log 4)=-2 log 2,  (0 - log e)=-1.
    log_weights = torch.tensor([[0.0, 999.0], [0.0, 0.0]])
    penalty = compute_log_weight_smoothness_penalty(log_weights, prior)
    expected = (0.0**2 + (2 * torch.log(torch.tensor(2.0))).item() ** 2 + 1.0**2) / 3
    assert penalty.item() == pytest.approx(expected, rel=1e-5)
