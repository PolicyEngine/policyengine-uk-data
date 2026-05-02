import io

import modal
import numpy as np

app = modal.App("policyengine-uk-calibration")

image_gpu = modal.Image.debian_slim().pip_install("numpy", "torch")


def _load_array(array_bytes: bytes):
    return np.load(io.BytesIO(array_bytes))


@app.function(gpu="A10G", image=image_gpu, timeout=3600, serialized=True)
def run_calibration(
    matrix: bytes,
    y: bytes,
    local_target_available: bytes,
    r: bytes,
    matrix_national: bytes,
    y_national: bytes,
    weights_init: bytes,
    epochs: int,
):
    """Run the local-area Adam calibration loop on a Modal GPU."""
    import io

    import numpy as np
    import torch

    def load(array_bytes: bytes):
        return np.load(io.BytesIO(array_bytes))

    device = torch.device("cuda")
    metrics = torch.tensor(load(matrix), dtype=torch.float32, device=device)
    y_local = torch.tensor(load(y), dtype=torch.float32, device=device)
    local_mask = torch.tensor(
        load(local_target_available), dtype=torch.bool, device=device
    )
    r_tensor = torch.tensor(load(r), dtype=torch.float32, device=device)
    metrics_national = torch.tensor(
        load(matrix_national), dtype=torch.float32, device=device
    )
    y_nat = torch.tensor(load(y_national), dtype=torch.float32, device=device)
    weights = torch.tensor(
        load(weights_init),
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )

    def sre(x, y_ref):
        one_way = ((1 + x) / (1 + y_ref) - 1) ** 2
        other_way = ((1 + y_ref) / (1 + x) - 1) ** 2
        return torch.min(one_way, other_way)

    def loss_fn(w):
        pred_local = (w.unsqueeze(-1) * metrics.unsqueeze(0)).sum(dim=1)
        if local_mask.any():
            mse_local = torch.mean(sre(pred_local[local_mask], y_local[local_mask]))
        else:
            mse_local = pred_local.sum() * 0

        pred_national = (w.sum(axis=0) * metrics_national.T).sum(axis=1)
        mse_national = torch.mean(sre(pred_national, y_nat))
        return mse_local + mse_national

    def dropout_weights(w, p):
        if p == 0:
            return w
        mask = torch.rand_like(w) < p
        mean = w[~mask].mean()
        masked_weights = w.clone()
        masked_weights[mask] = mean
        return masked_weights

    optimizer = torch.optim.Adam([weights], lr=1e-1)
    checkpoints = []

    for epoch in range(epochs):
        optimizer.zero_grad()
        weights_with_dropout = torch.exp(dropout_weights(weights, 0.05)) * r_tensor
        loss = loss_fn(weights_with_dropout)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            final_weights = (torch.exp(weights) * r_tensor).detach().cpu().numpy()
            buffer = io.BytesIO()
            np.save(buffer, final_weights)
            checkpoints.append((epoch, buffer.getvalue()))

    return checkpoints
