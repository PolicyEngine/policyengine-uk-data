# Calibration target coverage by year

Per-year calibration (step 4 of [#345](https://github.com/PolicyEngine/policyengine-uk-data/issues/345)) depends on target values being available for every year we want to calibrate against. This document is the snapshot of where coverage is today so it is easy to see where future data-sourcing work is needed.

The pipeline resolves a requested year against each target's available years using `resolve_target_value` in `policyengine_uk_data/targets/build_loss_matrix.py`. The policy is: exact match → nearest past year within three years → `None`. There is no backwards extrapolation and no forwards extrapolation beyond three years — if the tolerance is exceeded the target is silently skipped.

## National and country-level targets

| Category | Source | Year range in repo | Year-keyed? | Notes |
| --- | --- | --- | --- | --- |
| Population by sex × age band | ONS mid-year population estimates | 2022-2029 | Yes (`ons_demographics.py`) | Downloaded as multi-year at registry build time |
| Regional population by age | ONS subnational estimates | 2018-2029 | Yes (`demographics.csv`) | Multi-year CSV keyed by `year` column |
| UK total population | ONS | 2018-2029 | Yes (`demographics.csv`) | Used by `resolve_target_value` for VOA scaling |
| Scotland demographics (children, babies, 3+ child households) | ONS / Scottish Government | 2025 + some 2029 | Partial | Missing 2026-2028 |
| Income by band (SPI) — national | HMRC SPI | 2022-2029 | Yes (`incomes_projection.csv`) | Projected from 2021 SPI via microsimulation |
| Income tax, NICs, VAT, CGT, SDLT, fuel duty totals | OBR Economic and Fiscal Outlook | 2024-2030 | Yes (`obr.py`) | Live download, multi-year |
| Council tax totals | OBR | 2024-2030 | Yes | OBR line items |
| Council tax band counts | VOA | 2024 | No | Population-scaled by `resolve_target_value` for adjacent years |
| Housing totals (mortgage, private rent, social rent) | ONS / EHS | 2025 | No | Single-year only — needs 2026+ refresh |
| Tenure totals | ONS / EHS | 2025 | No | Single-year only |
| Savings interest | ONS | 2025 | No | Single-year only |
| Land values (household, corporate, total) | ONS National Balance Sheet | 2025 | No | Single-year only |
| Regional household land values | MHCLG | 2025 | No | Single-year only |
| DWP benefit caseloads (UC, ESA, PIP, JSA, benefit cap, UC by children / family type) | DWP Stat-Xplore / benefit statistics | 2025 (a few 2026) | Mostly no | **Primary gap**: 2026+ needs DWP forecasts or policy extrapolation |
| Salary sacrifice (IT relief, contributions, NI relief) | HMRC / OBR | 2025 | No | OBR has 2024-2030 on some items |
| Salary sacrifice headcount | OBR | 2024-2030 | Yes | Multi-year |
| UC jobseeker splits, UC outside cap | OBR | 2024-2030 | Yes | Multi-year |
| Two-child limit | DWP | 2025 | No | Single-year only |
| Student loan plan borrower counts | SLC | 2025 | No | Single-year only |
| Student loan repayment | SLC | 2025 | No | Single-year only |
| NTS vehicle ownership | DfT National Travel Survey | 2024 | No | Single-year only |
| TV licence | OBR | 2024 + 3% pa extrapolation | Implicit | Hard-coded extrapolation in `obr.py` |

## Constituency- and LA-level targets

| Category | Source | Year | Notes |
| --- | --- | --- | --- |
| Age bands per constituency / LA | ONS subnational population estimates | Snapshot (no year column) | `age.csv` files under `datasets/local_areas/*/targets/`; need annual refresh |
| Income by area (employment, self-employment; count + amount) | HMRC SPI table 3.15 | Snapshot | `spi_by_constituency.csv`, `spi_by_la.csv`; HMRC publishes annually |
| UC household counts by area | DWP Stat-Xplore | November 2023 | Scaled to 2025 national totals via `_scaled_uc_children_by_country` |
| UC households by number of children (area level) | DWP Stat-Xplore | November 2023 base + 2025 scaling | In `local_uc.py` |
| ONS small-area income estimates (LA only) | ONS | FYE 2020 + uprating | Uprated per-year via `get_ons_income_uprating_factors(year)` |
| Tenure by LA | English Housing Survey 2023 | Snapshot | `la_tenure.xlsx` |
| Private rent median by LA | VOA / ONS | Snapshot | `la_private_rents_median.xlsx` |

## Known gaps — what blocks full per-year calibration

1. **DWP benefit caseloads for 2026+**. The DWP statistical releases publish mostly current-year snapshots; forecasts are internal. Getting these requires coordination with the policy team or an agreed extrapolation policy from 2025 onwards.
2. **Local-area CSVs (age, SPI, UC)**. Single-year snapshots stored as CSVs without a `year` column. For panel calibration these need an annual refresh process and a filename convention that includes the source year (e.g. `spi_by_constituency_2024.csv`).
3. **Small-scale single-year sources**. NTS vehicles (2024), housing totals, SLC student loans, land values — each individually small but collectively relevant.

## Related code

- `policyengine_uk_data/targets/build_loss_matrix.py` — `resolve_target_value` and the national loss matrix builder.
- `policyengine_uk_data/datasets/local_areas/constituencies/loss.py` — constituency loss matrix; now honours `time_period`.
- `policyengine_uk_data/datasets/local_areas/local_authorities/loss.py` — LA loss matrix; now honours `time_period` (previously read weights at hard-coded 2025).
- `policyengine_uk_data/targets/sources/` — individual target modules. The multi-year ones (`obr.py`, `ons_demographics.py`) are the template for converting the rest.
