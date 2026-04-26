# PolicyEngine UK Data

PolicyEngine's project to build accurate UK household survey data.

## Local dataset builds

For a full local dataset build:

1. Ensure the private prerequisite folders exist under `policyengine_uk_data/storage/`.
2. Use Python 3.13. Python 3.14 currently fails while loading PyTables/Blosc2 in this repo.
3. Prefer the sibling `policyengine-uk` checkout when building locally, because the published wheel in your active environment may not expose all variables required by the data pipeline.

If `../policyengine-uk` exists, you can run:

```sh
make data-local
```

## Public UK calibrated transfer dataset

This repo now also includes a public calibrated microdata file:

- `policyengine_uk_data/storage/enhanced_cps_2025.h5`
- source manifest: `policyengine_uk_data/storage/enhanced_cps_source_2025.csv`
- artifact manifest: `policyengine_uk_data/storage/enhanced_cps_manifest_2025.json`

The public UK calibrated transfer dataset starts from a public export of eligible households from
PolicyEngine-US Enhanced CPS. In the current build that source manifest contains
`28,532` households, not `1,000`. The pipeline maps those records into a
`UKSingleYearDataset`, aligns core UK-facing inputs such as council tax bands,
vehicle ownership, pensions, disability/PIP, consumption, and capital gains,
and then recalibrates the household weights against the UK national/region/country
target registry used by the loss pipeline.

The checked-in 2025 artifact uses a pinned `0.759` USD-to-GBP conversion rate
from the IRS 2025 yearly average exchange-rate table. The builder intentionally
does not call a live foreign-exchange API, because the committed H5 should be
reproducible from versioned code and source data. Pass `exchange_rate=...` to
`create_enhanced_cps` or `save_enhanced_cps` when rebuilding with a different
conversion assumption.

The calibration step is tested against the native 2025 loss matrix and should
reduce mean absolute relative error relative to the raw transfer weights. Report
full-artifact loss values from a named release artifact rather than copying them
into this README.

This is a public calibrated dataset, not a replacement for the FRS or enhanced
FRS. It is intended as the first step in a broader cross-country public-microdata
strategy.

The legacy `policybench_transfer_2025.h5` and
`policybench_transfer_source_2025.csv` files remain the original
1,000-household proof-of-method artifacts. The Python
`create_policybench_transfer` and `save_policybench_transfer` entry points are
backward-compatible aliases for the current `enhanced_cps` builder, not a
request to regenerate the legacy 1,000-household files.

Programmatic entrypoints:

- `policyengine_uk_data.datasets.create_enhanced_cps`
- `policyengine_uk_data.datasets.export_enhanced_cps_source`
- `policyengine_uk_data.datasets.save_enhanced_cps`
- `make enhanced-cps-manifest`
- `make upload-public-transfer`

Backward-compatible aliases remain available:

- `policyengine_uk_data.datasets.create_policybench_transfer`
- `policyengine_uk_data.datasets.save_policybench_transfer`
