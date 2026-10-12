"""Hub finisher: income tax and household net income, aggregate output only.

Run only after the hub approves the queued builds; one microsimulation at a
time, in the same locked environment used for all four builds.
Usage: python totals.py <enhanced_frs.h5> <output.json>
"""

import hashlib
import importlib.metadata
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
from policyengine_uk import Microsimulation


def main():
    dataset, output = Path(sys.argv[1]), Path(sys.argv[2])
    for variable in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        if os.environ.get(variable) != "1":
            raise SystemExit(f"Export {variable}=1 before measuring totals")
    memory = subprocess.check_output(["vm_stat"], text=True)
    page_size = int(re.search(r"page size of (\d+) bytes", memory).group(1))
    pages = int(re.search(r"Pages occupied by compressor:\s+(\d+)\.", memory).group(1))
    if pages * page_size >= 30e9:
        raise SystemExit("Compressor is at least 30 GB; wait before microsimulation")
    with dataset.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    result = {
        "dataset_sha256": digest,
        "policyengine_uk_version": importlib.metadata.version("policyengine-uk"),
        "policyengine_core_version": importlib.metadata.version("policyengine-core"),
    }
    if result["policyengine_uk_version"] != "2.122.2":
        raise SystemExit("Use the frozen policyengine-uk 2.122.2 build environment")
    if result["policyengine_core_version"] != "3.32.13":
        raise SystemExit("Use the frozen policyengine-core 3.32.13 build environment")
    sim = Microsimulation(dataset=str(dataset))
    for year in (2025, 2026):
        result[str(year)] = {}
        for variable, weight in (
            ("income_tax", "person_weight"),
            ("household_net_income", "household_weight"),
        ):
            values = np.asarray(sim.calculate(variable, year).values, dtype=np.float64)
            weights = np.asarray(sim.calculate(weight, year).values, dtype=np.float64)
            result[str(year)][variable] = {
                "total_bn": float(np.sum(values * weights, dtype=np.float64) / 1e9)
            }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
