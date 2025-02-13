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

FOLDER = Path(__file__).parent


def calibrate(
    map_to_2024_boundaries: bool = True,
    epochs: int = 256,
):
    matrix, y = create_constituency_target_matrix("enhanced_frs_2022_23", 2025)

    m_national, y_national = create_national_target_matrix(
        "enhanced_frs_2022_23", 2025
    )

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
    )
    metrics = torch.tensor(matrix.values, dtype=torch.float32)
    y = torch.tensor(y.values, dtype=torch.float32)
    matrix_national = torch.tensor(m_national.values, dtype=torch.float32)
    y_national = torch.tensor(y_national.values, dtype=torch.float32)

    def loss(w):
        pred_c = (w.unsqueeze(-1) * metrics.unsqueeze(0)).sum(dim=1)
        mse_c = torch.mean((pred_c / (1 + y) - 1) ** 2)

        pred_n = (w.sum(axis=0) * matrix_national.T).sum(axis=1)
        mse_n = torch.mean((pred_n / (1 + y_national) - 1) ** 2)

        return mse_c + mse_n

    def pct_close(w, t=0.1):
        # Return the percentage of metrics that are within t% of the target
        pred_c = (w.unsqueeze(-1) * metrics.unsqueeze(0)).sum(dim=1)
        e_c = torch.sum(torch.abs((pred_c / (1 + y) - 1)) < t)
        c_c = pred_c.shape[0] * pred_c.shape[1]

        pred_n = (w.sum(axis=0) * matrix_national.T).sum(axis=1)
        e_n = torch.sum(torch.abs((pred_n / (1 + y_national) - 1)) < t)
        c_n = pred_n.shape[0]

        return (e_c + e_n) / (c_c + c_n)

    def dropout_weights(weights, p):
        if p == 0:
            return weights
        # Replace p% of the weights with the mean value of the rest of them
        mask = torch.rand_like(weights) < p
        mean = weights[~mask].mean()
        masked_weights = weights.clone()
        masked_weights[mask] = mean
        return masked_weights

    optimizer = torch.optim.Adam([weights], lr=0.1)

    desc = range(32) if os.environ.get("DATA_LITE") else range(epochs)

    for epoch in desc:
        optimizer.zero_grad()
        weights_ = torch.exp(dropout_weights(weights, 0.05))
        l = loss(weights_)
        l.backward()
        optimizer.step()
        close = pct_close(weights_)
        if epoch % 10 == 0:
            print(f"Loss: {l.item()}, Epoch: {epoch}, Within 10%: {close:.2%}")

    final_weights = torch.exp(weights).detach().numpy()

    if map_to_2024_boundaries:
        final_weights = mapping_matrix @ final_weights

    with h5py.File(
        STORAGE_FOLDER / "parliamentary_constituency_weights.h5", "w"
    ) as f:
        f.create_dataset("2025", data=final_weights)

    # Override national weights in 2025 with the sum of the constituency weights

    with h5py.File(
        STORAGE_FOLDER / "enhanced_frs_2022_23.h5",
        "r+",
    ) as f:
        national_weights = final_weights.sum(axis=0)
        f.create_dataset("household_weight/2025", data=national_weights)

    return final_weights


if __name__ == "__main__":
    calibrate()
