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

## Building the FRS from your own UK Data Service download

If you hold a UK Data Service (UKDS) licence for the Family Resources Survey,
you can build PolicyEngine's base FRS dataset from the raw tables yourself.

**Supported release.** `main` builds FRS 2024-25 only (UKDS SN 9563, pinned as
`CURRENT_FRS_RELEASE` in `policyengine_uk_data/datasets/frs_release.py`). The
build selects columns and thresholds reported benefits by survey year, and no
test runs it on another release.

**Install from GitHub, not PyPI.** At the time of writing the
`policyengine-uk-data` package on PyPI is version 1.11.1, from December 2024.
It knows FRS releases up to 2022-23 only and does not map codes that later
releases use. Clone this repository instead:

```sh
git clone https://github.com/PolicyEngine/policyengine-uk-data
cd policyengine-uk-data
uv sync --extra dev  # Python 3.13
```

**Build.**

1. Download the FRS 2024-25 TAB-format zip from the UKDS (SN 9563).
2. Check it is the release the build pins: `shasum -a 256` of the zip should
   equal `ukds_tab_zip_sha256` in `frs_release.py`. A different digest means a
   re-issued release, so check the "NEW FOR" sheet of its variable listing
   before relying on the mappings.
3. Extract the `.tab` files under `UKDA-9563-tab/tab/` into
   `policyengine_uk_data/storage/frs_2024_25/`. The folder name sets the survey
   year, and `create_frs` refuses a `year` that does not match it.
4. Run `uv run python -m policyengine_uk_data.datasets.frs`. This writes
   `policyengine_uk_data/storage/frs_2024_25.h5`. From Python:

   ```python
   from policyengine_uk_data.datasets.frs import create_frs

   frs = create_frs(
       "policyengine_uk_data/storage/frs_2024_25", year=2024, policy_year=2025
   )
   frs.save("frs_2024_25.h5")
   ```

**What you get.** The base FRS with DWP's grossing weights. It has none of the
enhanced FRS imputations (wealth, consumption, VAT, top incomes from the SPI)
and no calibration to administrative totals, so its aggregates differ from the
figures PolicyEngine publishes, which use the enhanced FRS and Microcosm
datasets. UKDS licence holders can request access to those through the
[policyengine-uk getting-started guide](https://github.com/PolicyEngine/policyengine-uk/blob/main/docs/book/usage/getting-started.md#running-microsimulation-analyses).

**Running it.** A local file bypasses `policyengine.py`'s certified release
bundle, so pass `allow_unmanaged=True`:

```python
import policyengine as pe

datasets = pe.uk.ensure_datasets(
    datasets=["/path/to/frs_2024_25.h5"],
    years=[2026],
    data_folder="./data",
    allow_unmanaged=True,
)
```

**FRS codes that changed since 2022-23.** The UKDS variable listings for
2022-23 (SN 9252), 2023-24 (SN 9367) and 2024-25 (SN 9563) give the same codes
for region (GVTREGNO), tenure (PTENTYP2), accommodation type (TYPEACC), marital
status (MARITAL), ILO employment status (EMPSTATI) and school type (TYPEED2).
The differences that matter here:

- **Council tax band (CTBAND).** From 2024-25 DWP takes the band from Valuation
  Office Agency and Scottish Assessors Association records rather than asking
  the respondent. Code 10, now "Information not found" (before 2024-25
  "Household not valued separately"), gives no band, and the build assigns
  band D, the policyengine-uk default. Code 9 is band I, now labelled "Wales
  only".
- **Benefit codes (BENEFITS.BENEFIT).** Codes 69 and 70, recoveries and Social
  Fund repayments, are new in 2024-25 and are not read. Scottish Adult and
  Child Disability Payment (117, 118, 121, 122) are not read yet: see
  [#500](https://github.com/PolicyEngine/policyengine-uk-data/issues/500).
- **Benefit receipt.** DWP now bases receipt of the main benefits on
  administrative data, so reported receipt is higher than in earlier releases
  ([FRS 2024-25 background information and methodology](https://www.gov.uk/government/statistics/family-resources-survey-financial-year-2024-to-2025/family-resources-survey-background-information-and-methodology)).
- **Highest qualification (EDUCQUAL).** Its labels were reworded in 2024-25
  with the same codes. The mapping to `highest_education` does not yet follow
  the data dictionary: see
  [#459](https://github.com/PolicyEngine/policyengine-uk-data/issues/459).
- **Scottish sewerage charge (CSEWAMT).** The column is empty in 2024-25, so
  the build sets Scottish water and sewerage charges to zero: see
  [#467](https://github.com/PolicyEngine/policyengine-uk-data/issues/467).

## Public UK calibrated transfer dataset

This repo now also includes a public calibrated microdata file:

- `policyengine_uk_data/storage/enhanced_cps_2025.h5`
- source manifest: `policyengine_uk_data/storage/enhanced_cps_source_2025.csv`

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

Programmatic entrypoints:

- `policyengine_uk_data.datasets.create_enhanced_cps`
- `policyengine_uk_data.datasets.export_enhanced_cps_source`
- `policyengine_uk_data.datasets.save_enhanced_cps`

Backward-compatible aliases remain available:

- `policyengine_uk_data.datasets.create_policybench_transfer`
- `policyengine_uk_data.datasets.save_policybench_transfer`
