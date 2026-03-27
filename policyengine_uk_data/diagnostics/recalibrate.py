"""Recalibration with weight regularisation.

Extends the existing calibration pipeline to add entropy
regularisation (penalising weight distributions that diverge from
a prior) and optional hard weight capping.  This prevents the
calibration from concentrating weight on a few records, even when
the expanded dataset provides alternatives.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from policyengine_uk.data import UKSingleYearDataset

logger = logging.getLogger(__name__)


def recalibrate_with_regularisation(
    dataset,
    time_period: str = "2025",
    entropy_lambda: float = 0.01,
    weight_cap: float | None = 5_000.0,
    epochs: int = 256,
    lr: float = 0.05,
) -> UKSingleYearDataset:
    """Recalibrate dataset weights with entropy regularisation.

    Minimises:
        sum_t (hat_T_t(w) - T_t)^2  +  lambda * sum_i w_i * log(w_i / w0_i)

    where T_t are population targets, hat_T_t are weighted estimates,
    w0_i are prior weights (uniform for offspring, design weights for
    originals), and lambda controls regularisation strength.

    Args:
        dataset: UKSingleYearDataset to recalibrate.
        time_period: calendar year as string.
        entropy_lambda: entropy regularisation strength.
        weight_cap: optional hard upper bound on any weight.
        epochs: optimisation epochs.
        lr: learning rate.

    Returns:
        Recalibrated UKSingleYearDataset (copy with updated weights).
    """
    from policyengine_uk_data.targets.build_loss_matrix import (
        create_target_matrix,
    )

    dataset = dataset.copy()

    matrix, targets = create_target_matrix(dataset, time_period=time_period)

    if matrix.empty:
        logger.warning("No targets available, returning unmodified dataset")
        return dataset

    initial_weights = dataset.household.household_weight.values.astype(float)
    # Prior weights: original weights normalised to sum to population
    w0 = np.maximum(initial_weights, 1.0)

    # Tensors
    log_w = torch.tensor(
        np.log(np.maximum(initial_weights, 1e-6)),
        dtype=torch.float32,
        requires_grad=True,
    )
    M = torch.tensor(matrix.values, dtype=torch.float32)  # (n_households, n_targets)
    T = torch.tensor(targets.values, dtype=torch.float32)  # (n_targets,)
    w0_t = torch.tensor(w0, dtype=torch.float32)

    optimizer = torch.optim.Adam([log_w], lr=lr)

    def loss_fn():
        w = torch.exp(log_w)

        # Apply weight cap via soft clamping
        if weight_cap is not None:
            w = torch.clamp(w, max=weight_cap)

        # Target matching: symmetric relative error
        pred = (w.unsqueeze(1) * M).sum(dim=0)
        sre = torch.min(
            ((1 + pred) / (1 + T) - 1) ** 2,
            ((1 + T) / (1 + pred) - 1) ** 2,
        )
        target_loss = sre.mean()

        # Entropy regularisation: KL divergence from prior
        w_normed = w / w.sum()
        w0_normed = w0_t / w0_t.sum()
        # Avoid log(0) with small epsilon
        kl = (w_normed * torch.log((w_normed + 1e-10) / (w0_normed + 1e-10))).sum()

        return target_loss + entropy_lambda * kl

    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = loss_fn()
        loss.backward()
        optimizer.step()

        if epoch % 50 == 0:
            w_current = torch.exp(log_w).detach().numpy()
            if weight_cap is not None:
                w_current = np.clip(w_current, 0, weight_cap)
            logger.info(
                "Epoch %d: loss=%.6f, max_weight=%.0f, n_nonzero=%d",
                epoch,
                loss.item(),
                w_current.max(),
                (w_current > 1).sum(),
            )

    # Final weights
    final_weights = torch.exp(log_w).detach().numpy()
    if weight_cap is not None:
        final_weights = np.clip(final_weights, 0, weight_cap)

    dataset.household["household_weight"] = final_weights
    return dataset


def prune_zero_weight_records(
    dataset,
    epsilon: float = 1.0,
) -> UKSingleYearDataset:
    """Remove records with near-zero weight after recalibration.

    Args:
        dataset: UKSingleYearDataset with calibrated weights.
        epsilon: weight threshold below which records are removed.

    Returns:
        Pruned UKSingleYearDataset.
    """
    from policyengine_uk.data import UKSingleYearDataset

    keep_mask = dataset.household.household_weight > epsilon
    keep_hh_ids = dataset.household.household_id[keep_mask].values

    person_keep = dataset.person.person_household_id.isin(keep_hh_ids)
    keep_bu_ids = dataset.person[person_keep].person_benunit_id.unique()
    benunit_keep = dataset.benunit.benunit_id.isin(keep_bu_ids)

    n_removed = (~keep_mask).sum()
    logger.info(
        "Pruned %d zero-weight records (%.1f%%), %d remain",
        n_removed,
        100 * n_removed / len(keep_mask),
        keep_mask.sum(),
    )

    return UKSingleYearDataset(
        person=dataset.person[person_keep].reset_index(drop=True),
        benunit=dataset.benunit[benunit_keep].reset_index(drop=True),
        household=dataset.household[keep_mask].reset_index(drop=True),
        fiscal_year=int(dataset.time_period),
    )
