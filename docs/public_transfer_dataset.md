# Public UK transfer dataset

The public UK transfer dataset is an openly distributable benchmark artifact.
It is not native UK survey microdata.

The 2025 artifact starts from a public export of benchmark-compatible
PolicyEngine US Enhanced CPS households. The builder maps those households into
UK-facing PolicyEngine inputs, assigns synthetic UK geography, populates input
leaves such as council tax bands, vehicle ownership, pensions, disability/PIP,
consumption, and capital gains, and recalibrates household weights to selected
UK national, regional, and country targets.

The current public artifact is:

- `policyengine_uk_data/storage/enhanced_cps_2025.h5`
- `policyengine_uk_data/storage/enhanced_cps_source_2025.csv`
- `policyengine_uk_data/storage/enhanced_cps_manifest_2025.json`

The artifact manifest is the source of record for row counts, checksums, build
assumptions, weight diagnostics, and loss diagnostics. The checked-in 2025
manifest reports 28,532 households in both the source CSV and H5 file, 58,848
people in the H5 file, an effective sample size of about 11,197 households, and
a top-10 household-weight share of about 0.52%.

## Intended use

Use this dataset for public demos, reproducible examples, and public benchmark
analysis where restricted UK microdata cannot be redistributed.

Do not use this dataset as a substitute for FRS or enhanced FRS, as evidence of
the UK joint household distribution, or as administrative ground truth. Aggregate
calibration can improve target fit without recovering the native UK joint
distribution.

## Versioning

The public artifact should be cited by path, manifest, package version, and
checksum. The 2025 artifact uses a pinned USD-to-GBP exchange rate of 0.759 from
the IRS 2025 yearly average exchange-rate table. The builder deliberately does
not call a live foreign-exchange API.

## Legacy files

The `policybench_transfer_2025.h5` and `policybench_transfer_source_2025.csv`
files are retained as the original 1,000-household proof-of-method artifacts.
Current Python entry points named `create_policybench_transfer` and
`save_policybench_transfer` are aliases for the current 28,532-household
`enhanced_cps` builder.
