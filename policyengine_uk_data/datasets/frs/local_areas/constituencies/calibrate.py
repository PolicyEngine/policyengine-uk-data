import torch
from policyengine_uk import Microsimulation
import pandas as pd
import numpy as np
from tqdm import tqdm
import h5py
import os

# Fill in missing constituencies with average column values
import pandas as pd
import numpy as np

from policyengine_uk_data.datasets.frs.local_areas.constituencies.loss import (
    create_constituency_target_matrix,
    create_national_target_matrix,
)
from policyengine_uk_data.datasets.frs.local_areas.constituencies.boundary_changes.mapping_matrix import (
    mapping_matrix,
)
from pathlib import Path
from policyengine_uk_data.storage import STORAGE_FOLDER
from typing import List

FOLDER = Path(__file__).parent

DEVICE = "cpu"


def calibrate(
    exclude_targets: List[str] | None = None,
    epochs: int = 256,
    analytics = None,
):
    matrix, y, country_mask = create_constituency_target_matrix(
        "enhanced_frs_2022_23", 2025
    )

    m_national, y_national = create_national_target_matrix(
        "enhanced_frs_2022_23", 2025
    )

    if exclude_targets is not None:
        matrix = matrix.drop(columns=exclude_targets)
        y = y.drop(columns=exclude_targets)

    sim = Microsimulation(dataset="enhanced_frs_2022_23")

    COUNT_CONSTITUENCIES = 650

    # Weights - 650 x 100180
    original_weights = np.log(
        sim.calculate("household_weight", 2025).values / COUNT_CONSTITUENCIES
    )
    weights = torch.tensor(
        np.ones((COUNT_CONSTITUENCIES, len(original_weights)))
        * original_weights,
        dtype=torch.float32,
        requires_grad=True,
        device=DEVICE,
    )
    metrics = torch.tensor(matrix.values, dtype=torch.float32, device=DEVICE)
    y = torch.tensor(y.values, dtype=torch.float32, device=DEVICE)
    matrix_national = torch.tensor(m_national.values, dtype=torch.float32, device=DEVICE)
    y_national = torch.tensor(y_national.values, dtype=torch.float32, device=DEVICE)
    r = torch.tensor(country_mask, dtype=torch.float32, device=DEVICE)

    def loss(w):
        pred_c = (w.unsqueeze(-1) * metrics.unsqueeze(0)).sum(dim=1)
        mse_c = torch.mean((pred_c / (1 + y) - 1) ** 2)

        pred_n = (w.sum(axis=0) * matrix_national.T).sum(axis=1)
        mse_n = torch.mean((pred_n / (1 + y_national) - 1) ** 2)

        return mse_c + mse_n

    def pct_close(w, t=0.1, constituency=True, national=True):
        # Return the percentage of metrics that are within t% of the target
        numerator = 0
        denominator = 0
        pred_c = (w.unsqueeze(-1) * metrics.unsqueeze(0)).sum(dim=1)
        e_c = torch.sum(torch.abs((pred_c / (1 + y) - 1)) < t).item()
        c_c = pred_c.shape[0] * pred_c.shape[1]

        if constituency:
            numerator += e_c
            denominator += c_c

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

    optimizer = torch.optim.Adam([weights], lr=0.15)

    desc = range(32) if os.environ.get("DATA_LITE") else range(epochs)

    analytics_values = []

    for epoch in desc:
        optimizer.zero_grad()
        weights_ = torch.exp(dropout_weights(weights, 0.05)) * r
        l = loss(weights_)
        l.backward()
        optimizer.step()
        c_close = pct_close(weights_, constituency=True, national=False)
        n_close = pct_close(weights_, constituency=False, national=True)
        if epoch % 1 == 0:
            print(
                f"Loss: {l.item()}, Epoch: {epoch}, Constituency<10%: {c_close:.1%}, National<10%: {n_close:.1%}"
            )
        if epoch % 10 == 0:
            if analytics is not None:
                analytics_values.append(
                    analytics(
                        weights_.detach().cpu().numpy(),
                    )
                )
                print(analytics_values[-1])
            final_weights = (torch.exp(weights) * r).detach().cpu().numpy()

            with h5py.File(
                STORAGE_FOLDER / "parliamentary_constituency_weights.h5", "w"
            ) as f:
                f.create_dataset("2025", data=final_weights)

            with h5py.File(
                STORAGE_FOLDER / "enhanced_frs_2022_23.h5", "r+"
            ) as f:
                if "household_weight/2025" in f:
                    del f["household_weight/2025"]
                f.create_dataset(
                    "household_weight/2025", data=final_weights.sum(axis=0)
                )

    return final_weights, analytics_values


if __name__ == "__main__":
    calibrate()
