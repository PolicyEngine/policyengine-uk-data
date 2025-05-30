import numpy as np
import torch
import os


def reweight(
    original_weights,
    loss_matrix,
    targets_array,
    dropout_rate=0.05,
):
    target_names = np.array(loss_matrix.columns)
    loss_matrix = torch.tensor(loss_matrix.values, dtype=torch.float32)
    targets_array = torch.tensor(targets_array, dtype=torch.float32)
    weights = torch.tensor(
        np.log(original_weights), requires_grad=True, dtype=torch.float32
    )

    # TODO: replace this with a call to the python reweight.py package.
    def loss(weights):
        # Check for Nans in either the weights or the loss matrix
        if torch.isnan(weights).any():
            raise ValueError("Weights contain NaNs")
        if torch.isnan(loss_matrix).any():
            raise ValueError("Loss matrix contains NaNs")
        estimate = weights @ loss_matrix
        if torch.isnan(estimate).any():
            raise ValueError("Estimate contains NaNs")
        rel_error = (
            ((estimate - targets_array) + 1) / (targets_array + 1)
        ) ** 2
        if torch.isnan(rel_error).any():
            raise ValueError("Relative error contains NaNs")
        return rel_error.mean()

    def pct_close(weights, t=0.1):
        # Return the percentage of metrics that are within t% of the target
        estimate = weights @ loss_matrix
        abs_error = torch.abs((estimate - targets_array) / (1 + targets_array))
        return (abs_error < t).sum() / abs_error.numel()

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
    from tqdm import trange

    start_loss = None

    iterator = range(32) if os.environ.get("DATA_LITE") else range(2048)
    for i in iterator:
        optimizer.zero_grad()
        weights_ = dropout_weights(weights, dropout_rate)
        l = loss(torch.exp(weights_))
        close = pct_close(torch.exp(weights_))
        if start_loss is None:
            start_loss = l.item()
        loss_rel_change = (l.item() - start_loss) / start_loss
        l.backward()
        if i % 100 == 0:
            print(
                f"Loss: {l.item()}, Rel change: {loss_rel_change}, Epoch: {i}, Within 10%: {close:.2%}"
            )
        optimizer.step()

    return torch.exp(weights).detach().numpy()
