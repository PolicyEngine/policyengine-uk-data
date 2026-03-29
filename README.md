# PolicyEngine UK Data

PolicyEngine's project to build accurate UK household survey data.

## Public enhanced CPS

This repo now also includes a public calibrated microdata file:

- `policyengine_uk_data/storage/enhanced_cps_2025.h5`
- source manifest: `policyengine_uk_data/storage/enhanced_cps_source_2025.csv`

The UK enhanced CPS starts from PolicyBench's public 1,000-household CPS-derived
sample, maps those records into a `UKSingleYearDataset`, and recalibrates the
household weights against the UK national/region/country target registry used by
the loss pipeline.

On the native 2025 loss matrix, that reweighting step cuts mean absolute
relative error from roughly `3.81` on the raw transfer weights to roughly
`0.66` on the calibrated dataset.

This is a public calibrated dataset, not a replacement for the FRS or enhanced
FRS. It is intended as the first step in a broader cross-country public-microdata
strategy.

Programmatic entrypoints:

- `policyengine_uk_data.datasets.create_enhanced_cps`
- `policyengine_uk_data.datasets.save_enhanced_cps`

Backward-compatible aliases remain available:

- `policyengine_uk_data.datasets.create_policybench_transfer`
- `policyengine_uk_data.datasets.save_policybench_transfer`
