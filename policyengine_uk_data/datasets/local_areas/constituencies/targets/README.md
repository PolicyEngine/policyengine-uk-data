# Data

* Age data is from the [House of Commons Library](https://commonslibrary.parliament.uk/constituency-statistics-population-by-age/), with single-year age counts for each parliamentary constituency. The data is from 2020. The raw download should be saved as `raw_age.csv`; running `fill_missing_age_demographics.py` fills in missing constituencies and writes the processed output to `age.csv`.
  * The raw data covers **England, Wales and Scotland** constituencies. Northern Ireland constituencies are missing from this source.
  * Missing constituencies are filled using **country-specific mean age profiles** (not UK-wide mean). For NI, the age distribution is derived from [ONS subnational population projections](https://www.ons.gov.uk/peoplepopulationandcommunity/populationandmigration/populationprojections/datasets/tablea21principalprojectionukpopulationinagegroups) via `demographics.csv`.
  * NI constituency-level age data is also available from [NISRA mid-year estimates (small areas)](https://www.nisra.gov.uk/publications/2024-mid-year-population-estimates-small-geographical-areas-within-northern-ireland) if per-constituency precision is needed in future.
* Employment incomes are from Nomis ASHE workplace analysis, covering all workers (part-time and full-time), from 2023.
* HMRC total income is from 2021.
