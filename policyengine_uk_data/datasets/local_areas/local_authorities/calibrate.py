import torch
from policyengine_uk import Microsimulation
import pandas as pd
import numpy as np
from tqdm import tqdm
import h5py
import os
from policyengine_uk_data.storage import STORAGE_FOLDER


from policyengine_uk_data.datasets.local_areas.local_authorities.loss import (
    create_local_authority_target_matrix,
    create_national_target_matrix,
)
from policyengine_uk.data import UKDataset

DEVICE = "cpu"


def calibrate(
    dataset: UKDataset,
    verbose: bool = False,
):
    dataset = dataset.copy()
    matrix, y, r = create_local_authority_target_matrix(
        dataset, dataset.time_period
    )

    m_national, y_national = create_national_target_matrix(
        dataset, dataset.time_period
    )

    sim = Microsimulation(dataset=dataset)
    sim.default_calculation_period = dataset.time_period

    count_local_authority = 360

    # Weights - 360 x 100180
    original_weights = np.log(
        sim.calculate("household_weight").values / count_local_authority
        + np.random.random(len(sim.calculate("household_weight").values))
        * 0.01
    )
    weights = torch.tensor(
        np.ones((count_local_authority, len(original_weights)))
        * original_weights,
        dtype=torch.float32,
        device=DEVICE,
        requires_grad=True,
    )
    metrics = torch.tensor(matrix.values, dtype=torch.float32, device=DEVICE)
    y = torch.tensor(y.values, dtype=torch.float32, device=DEVICE)
    matrix_national = torch.tensor(
        m_national.values, dtype=torch.float32, device=DEVICE
    )
    y_national = torch.tensor(
        y_national.values, dtype=torch.float32, device=DEVICE
    )
    r = torch.tensor(r, dtype=torch.float32, device=DEVICE)

    def loss(w):
        pred_c = (w.unsqueeze(-1) * metrics.unsqueeze(0)).sum(dim=1)
        mse_c = torch.mean((pred_c / (1 + y) - 1) ** 2)

        pred_n = (w.sum(axis=0) * matrix_national.T).sum(axis=1)
        mse_n = torch.mean((pred_n / (1 + y_national) - 1) ** 2)

        return mse_c + mse_n

    def pct_close(w, t=0.1, la=True, national=True):
        # Return the percentage of metrics that are within t% of the target
        numerator = 0
        denominator = 0
        pred_la = (w.unsqueeze(-1) * metrics.unsqueeze(0)).sum(dim=1)
        e_la = torch.sum(torch.abs((pred_la / (1 + y) - 1)) < t).item()
        c_la = pred_la.shape[0] * pred_la.shape[1]

        if la:
            numerator += e_la
            denominator += c_la

        pred_n = (w.sum(axis=0) * matrix_national.T).sum(axis=1)
        e_n = torch.sum(torch.abs((pred_n / (1 + y_national) - 1)) < t).item()
        c_n = pred_n.shape[0]

        if national:
            numerator += e_n
            denominator += c_n

        return numerator / denominator

    def dropout_weights(weights, p):
        if p == 0:
            return weights
        # Replace p% of the weights with the mean value of the rest of them
        mask = torch.rand_like(weights) < p
        mean = weights[~mask].mean()
        masked_weights = weights.clone()
        masked_weights[mask] = mean
        return masked_weights

    optimizer = torch.optim.Adam([weights], lr=1e-1)

    desc = range(32) if os.environ.get("DATA_LITE") else range(128)

    for epoch in desc:
        optimizer.zero_grad()
        weights_ = torch.exp(dropout_weights(weights, 0.05)) * r
        l = loss(weights_)
        l.backward()
        optimizer.step()
        c_close = pct_close(weights_, la=True, national=False)
        n_close = pct_close(weights_, la=False, national=True)
        if verbose and (epoch % 1 == 0):
            print(
                f"Loss: {l.item()}, Epoch: {epoch}, Local Authority<10%: {c_close:.1%}, National<10%: {n_close:.1%}"
            )
        if epoch % 10 == 0:
            final_weights = (torch.exp(weights) * r).detach().cpu().numpy()

            with h5py.File(
                STORAGE_FOLDER / "local_authority_weights.h5", "w"
            ) as f:
                f.create_dataset("2025", data=final_weights)

        dataset.household.household_weight = final_weights.sum(axis=0)

    return dataset


if __name__ == "__main__":
    calibrate()
