"""Check land value calibration targets against latest HuggingFace data."""

import h5py
import numpy as np
from huggingface_hub import hf_hub_download

# ONS National Balance Sheet 2025 targets
_ONS_2020_HOUSEHOLD = 4.31e12
_ONS_2020_CORPORATE = 1.76e12
_ONS_2020_TOTAL = _ONS_2020_HOUSEHOLD + _ONS_2020_CORPORATE
_ONS_2024_TOTAL = 7.1e12
_SCALE = _ONS_2024_TOTAL / _ONS_2020_TOTAL

TARGETS = {
    "Household land (owned_land)": {
        "column": "owned_land",
        "target": _ONS_2020_HOUSEHOLD * _SCALE,
    },
    "Corporate wealth": {
        "column": "corporate_wealth",
        "target": _ONS_2020_CORPORATE * _SCALE,
    },
    "Property wealth": {
        "column": "property_wealth",
        "target": _ONS_2024_TOTAL,
    },
}

path = hf_hub_download(
    "policyengine/policyengine-uk-data-private",
    "enhanced_frs_2023_24.h5",
    force_download=True,
)

with h5py.File(path, "r") as f:
    tbl = f["household/table"][:]
    weights = tbl["household_weight"]

    rows = []
    for name, info in TARGETS.items():
        estimate = np.sum(weights * tbl[info["column"]])
        target = info["target"]
        rel_error = estimate / target - 1
        if abs(rel_error) > 0.5:
            status = "Way off"
        else:
            direction = "over" if rel_error > 0 else "under"
            status = f"~{abs(rel_error):.0%} {direction}"
        rows.append((name, estimate, target, status))

# Print table
col_w = [max(len(r[0]) for r in rows), 7, 10, max(len(r[3]) for r in rows)]
col_w[0] = max(col_w[0], len("Variable"))
col_w[3] = max(col_w[3], len("Status"))

hdr = ["Variable", "v1.44.0", "ONS Target", "Status"]


def fmt_row(vals):
    cells = []
    for i, v in enumerate(vals):
        cells.append(f" {v:<{col_w[i]}} ")
    return "│" + "│".join(cells) + "│"


def sep(left, mid, right):
    parts = [left]
    for i, w in enumerate(col_w):
        parts.append("─" * (w + 2))
        parts.append(mid if i < len(col_w) - 1 else right)
    return "".join(parts)


print(sep("┌", "┬", "┐"))
print(fmt_row(hdr))
for row in rows:
    print(sep("├", "┼", "┤"))
    name, est, tgt, status = row
    print(fmt_row([name, f"£{est/1e12:.2f}tn", f"£{tgt/1e12:.2f}tn", status]))
print(sep("└", "┴", "┘"))
