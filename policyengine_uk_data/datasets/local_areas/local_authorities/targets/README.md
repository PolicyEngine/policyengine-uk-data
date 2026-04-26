# Data

* Age data is from the [ONS mid-year population estimates](https://www.nomisweb.co.uk/datasets/pestsyoala), with single-year age counts for each local authority district. The raw download should be saved as `raw_age.csv`; running `fill_missing_age_demographics.py` fills missing values and intersects with the income file, writing the processed output to `age.csv`.
  * The Nomis dataset covers all UK local authorities including the 11 NI local government districts. However, some NI LGDs may have missing age breakdowns in the raw download.
  * Missing values are filled using **country-specific means** (not UK-wide mean), so NI areas get NI-shaped age profiles. The NI national age distribution comes from [ONS subnational population projections](https://www.ons.gov.uk/peoplepopulationandcommunity/populationandmigration/populationprojections/datasets/tablea21principalprojectionukpopulationinagegroups) via `demographics.csv`.
  * NI LGD-level data is also available directly from [NISRA mid-year estimates](https://www.nisra.gov.uk/publications/2024-mid-year-population-estimates-northern-ireland-and-estimates-population-aged-85).
* Employment incomes are from Nomis ASHE workplace analysis, covering all workers (part-time and full-time), from 2023.
* HMRC total income is from 2021.
