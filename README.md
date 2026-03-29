# PolicyEngine UK Data

PolicyEngine's project to build accurate UK household survey data.

## Public transfer dataset

This repo now also includes a public synthetic transfer dataset:

- `policyengine_uk_data/storage/policybench_transfer_2025.h5`
- source manifest: `policyengine_uk_data/storage/policybench_transfer_source_2025.csv`

It maps PolicyBench's public 1,000-household US benchmark sample into a
`UKSingleYearDataset`, preserving source household weights and assigning UK
geography and tenure synthetically. This is useful for cross-system transfer
benchmarking on shared households.

It is not a representative UK baseline like the FRS or enhanced FRS.

Programmatic entrypoints:

- `policyengine_uk_data.datasets.create_policybench_transfer`
- `policyengine_uk_data.datasets.save_policybench_transfer`
