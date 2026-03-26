"""A/B comparison: old min-of-ratios SRE vs new log-ratio SRE.

Runs a short calibration (256 epochs) with each loss function and
compares resulting population totals against the ONS target.
"""

import sys
import numpy as np
import torch
from policyengine_uk.data import UKSingleYearDataset
from policyengine_uk_data.datasets.local_areas.constituencies.loss import (
    create_constituency_target_matrix,
)
from policyengine_uk_data.targets.build_loss_matrix import (
    create_target_matrix as create_national_target_matrix,
)

EPOCHS = 256
AREA_COUNT = 650
ONS_TARGET_M = 69.5


def log(msg):
    print(msg, flush=True)


def run_calibration(sre_fn, label, dataset, matrix, y, r, m_national, y_national):
    """Run a short calibration and return final nat_pop in millions."""
    areas_per_household = r.sum(axis=0)
    areas_per_household = np.maximum(areas_per_household, 1)

    np.random.seed(42)
    original_weights = np.log(
        dataset.household.household_weight.values / areas_per_household
        + np.random.random(len(dataset.household.household_weight.values)) * 0.01
    )
    weights = torch.tensor(
        np.ones((AREA_COUNT, len(original_weights))) * original_weights,
        dtype=torch.float32,
        requires_grad=True,
    )

    metrics_t = torch.tensor(matrix.values, dtype=torch.float32)
    y_t = torch.tensor(y.values, dtype=torch.float32)
    mn_t = torch.tensor(m_national.values, dtype=torch.float32)
    yn_t = torch.tensor(y_national.values, dtype=torch.float32)
    r_t = torch.tensor(r, dtype=torch.float32)

    pop_col = None
    if hasattr(m_national, "columns"):
        cols = list(m_national.columns)
        if "ons/uk_population" in cols:
            pop_col = cols.index("ons/uk_population")

    def loss(w):
        pred_local = (w.unsqueeze(-1) * metrics_t.unsqueeze(0)).sum(dim=1)
        mse_local = torch.mean(sre_fn(pred_local, y_t))
        pred_national = (w.sum(axis=0) * mn_t.T).sum(axis=1)
        mse_national = torch.mean(sre_fn(pred_national, yn_t))
        return mse_local + mse_national

    optimizer = torch.optim.Adam([weights], lr=1e-1)

    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        mask = torch.rand_like(weights) < 0.05
        mean_w = weights[~mask].mean()
        w_dropped = weights.clone()
        w_dropped[mask] = mean_w
        w = torch.exp(w_dropped) * r_t

        l = loss(w)
        l.backward()
        optimizer.step()

        if epoch % 32 == 0 or epoch == EPOCHS - 1:
            with torch.no_grad():
                final_w = torch.exp(weights) * r_t
                pred_nat = (final_w.sum(axis=0) * mn_t.T).sum(axis=1)
                nat_pop = pred_nat[pop_col].item() / 1e6 if pop_col is not None else float("nan")
                pct_off = (nat_pop / ONS_TARGET_M - 1) * 100
                log(f"  [{label}] epoch {epoch:3d}: loss={l.item():.4f}  pop={nat_pop:.2f}M  deviation={pct_off:+.2f}%")

    with torch.no_grad():
        final_w = torch.exp(weights) * r_t
        pred_nat = (final_w.sum(axis=0) * mn_t.T).sum(axis=1)
        return pred_nat[pop_col].item() / 1e6 if pop_col is not None else float("nan")


def sre_old(x, y):
    one_way = ((1 + x) / (1 + y) - 1) ** 2
    other_way = ((1 + y) / (1 + x) - 1) ** 2
    return torch.min(one_way, other_way)


def sre_new(x, y):
    return torch.log((1 + x) / (1 + y)) ** 2


if __name__ == "__main__":
    from policyengine_uk_data.storage import STORAGE_FOLDER

    log("Loading dataset and building matrices...")
    dataset = UKSingleYearDataset(STORAGE_FOLDER / "enhanced_frs_2023_24.h5")
    ds_copy = dataset.copy()
    matrix, y, r = create_constituency_target_matrix(ds_copy)
    m_national, y_national = create_national_target_matrix(ds_copy)
    log("Matrices built.")

    log(f"\nONS population target: {ONS_TARGET_M}M")
    log(f"Running {EPOCHS} epochs with each loss function...\n")

    log("=== OLD loss (min-of-ratios, asymmetric) ===")
    pop_old = run_calibration(sre_old, "OLD", ds_copy, matrix, y, r, m_national, y_national)

    log("\n=== NEW loss (log-ratio, symmetric) ===")
    pop_new = run_calibration(sre_new, "NEW", ds_copy, matrix, y, r, m_national, y_national)

    log(f"\n{'='*50}")
    log(f"OLD loss final population: {pop_old:.2f}M ({(pop_old/ONS_TARGET_M - 1)*100:+.2f}%)")
    log(f"NEW loss final population: {pop_new:.2f}M ({(pop_new/ONS_TARGET_M - 1)*100:+.2f}%)")
    log(f"ONS target:               {ONS_TARGET_M}M")
