import io
import modal
import numpy as np

app = modal.App("policyengine-uk-calibration")

image_gpu = modal.Image.debian_slim().pip_install(
    "torch", "numpy", "h5py", "pandas"
)

image_cpu = (
    modal.Image.debian_slim(python_version="3.13")
    .apt_install("libhdf5-dev", "pkg-config", "gcc")
    .run_commands("pip install uv")
    .run_commands(
        "uv pip install --system torch --index-url https://download.pytorch.org/whl/cpu",
        "uv pip install --system policyengine-uk tables microimpute",
    )
    .add_local_dir("policyengine_uk_data", "/root/policyengine_uk_data")
)


@app.function(
    cpu=8,
    memory=16384,
    image=image_cpu,
    timeout=3600,
    serialized=True,
)
def run_imputation(frs_bytes: bytes, year: int = 2023) -> bytes:
    """
    Run the full imputation pipeline on a high-CPU container.

    Accepts and returns the FRS dataset serialised as h5 bytes.
    """
    import io
    import tempfile
    from policyengine_uk.data import UKSingleYearDataset
    from policyengine_uk_data.datasets.imputations import (
        impute_consumption,
        impute_wealth,
        impute_vat,
        impute_income,
        impute_capital_gains,
        impute_services,
        impute_salary_sacrifice,
        impute_student_loan_plan,
    )
    from policyengine_uk_data.utils.uprating import uprate_dataset

    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
        f.write(frs_bytes)
        frs_path = f.name

    frs = UKSingleYearDataset(file_path=frs_path)

    frs = impute_wealth(frs)
    frs = impute_consumption(frs)
    frs = impute_vat(frs)
    frs = impute_services(frs)
    frs = impute_income(frs)
    frs = impute_capital_gains(frs)
    frs = impute_salary_sacrifice(frs)
    frs = impute_student_loan_plan(frs, year=2025)
    frs = uprate_dataset(frs, 2025)

    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
        out_path = f.name
    frs.save(out_path)
    return open(out_path, "rb").read()


@app.function(gpu="T4", image=image_gpu, timeout=3600, serialized=True)
def run_calibration(
    matrix: bytes,
    y: bytes,
    r: bytes,
    matrix_national: bytes,
    y_national: bytes,
    weights_init: bytes,
    epochs: int,
) -> bytes:
    """
    Run the Adam calibration loop on a GPU container. All arrays are
    serialised with ``np.save`` / deserialised with ``np.load``.

    Returns the final weights (area_count × n_households) as np.save bytes.
    """
    import io
    import numpy as np
    import torch

    # Inline _run_optimisation to keep the Modal image dependency-free
    # (no policyengine_uk_data import needed inside the container).

    def load(b):
        return np.load(io.BytesIO(b))

    matrix_np = load(matrix)
    y_np = load(y)
    r_np = load(r)
    matrix_national_np = load(matrix_national)
    y_national_np = load(y_national)
    weights_init_np = load(weights_init)

    device = torch.device("cuda")

    metrics = torch.tensor(matrix_np, dtype=torch.float32, device=device)
    y_t = torch.tensor(y_np, dtype=torch.float32, device=device)
    m_national = torch.tensor(
        matrix_national_np, dtype=torch.float32, device=device
    )
    y_nat = torch.tensor(y_national_np, dtype=torch.float32, device=device)
    r_t = torch.tensor(r_np, dtype=torch.float32, device=device)

    weights = torch.tensor(
        weights_init_np,
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
        mse_local = torch.mean(sre(pred_local, y_t))
        pred_national = (w.sum(axis=0) * m_national.T).sum(axis=1)
        mse_national = torch.mean(sre(pred_national, y_nat))
        return mse_local + mse_national

    def dropout_weights(w, p):
        if p == 0:
            return w
        mask = torch.rand_like(w) < p
        mean = w[~mask].mean()
        w2 = w.clone()
        w2[mask] = mean
        return w2

    optimizer = torch.optim.Adam([weights], lr=1e-1)
    checkpoints = []

    for epoch in range(epochs):
        optimizer.zero_grad()
        weights_ = torch.exp(dropout_weights(weights, 0.05)) * r_t
        l = loss_fn(weights_)
        l.backward()
        optimizer.step()

        if epoch % 10 == 0:
            w = (torch.exp(weights) * r_t).detach().cpu().numpy()
            buf = io.BytesIO()
            np.save(buf, w)
            checkpoints.append((epoch, buf.getvalue()))

    return checkpoints
