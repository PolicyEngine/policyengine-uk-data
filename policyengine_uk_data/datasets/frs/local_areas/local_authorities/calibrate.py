import torch
from policyengine_uk import Microsimulation
import pandas as pd
import numpy as np
from tqdm import tqdm
import h5py
from policyengine_uk_data.storage import STORAGE_FOLDER


from loss import (
    create_local_authority_target_matrix,
    create_national_target_matrix,
)


def calibrate():
    matrix, y = create_local_authority_target_matrix(
        "enhanced_frs_2022_23", 2025
    )

    m_national, y_national = create_national_target_matrix(
        "enhanced_frs_2022_23", 2025
    )

    sim = Microsimulation(dataset="enhanced_frs_2022_23")

    count_local_authority = 360

    # Weights - 360 x 100180
    original_weights = np.log(
        sim.calculate("household_weight", 2025).values / count_local_authority
    )
    weights = torch.tensor(
        np.ones((count_local_authority, len(original_weights)))
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

    desc = range(512)

    for epoch in desc:
        optimizer.zero_grad()
        weights_ = dropout_weights(weights, 0.05)
        l = loss(torch.exp(weights_))
        l.backward()
        optimizer.step()
        if epoch % 50 == 0:
            print(f"Loss: {l.item()}, Epoch: {epoch}")

        if epoch % 100 == 0:
            final_weights = torch.exp(weights).detach().numpy()

            with h5py.File(
                STORAGE_FOLDER / "local_authority_weights.h5", "w"
            ) as f:
                f.create_dataset("2025", data=final_weights)


if __name__ == "__main__":
    calibrate()
