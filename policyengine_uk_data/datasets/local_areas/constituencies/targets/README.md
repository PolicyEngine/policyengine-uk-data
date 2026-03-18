# Data

* Age data is from the [House of Commons Library](https://commonslibrary.parliament.uk/constituency-statistics-population-by-age/), with single-year age counts for each parliamentary constituency (2010). The data is from 2020. The raw download should be saved as `raw_age.csv`; running `fill_missing_age_demographics.py` fills in missing Scotland and NI constituencies using mean age profiles and writes the processed output to `age.csv`.
* Employment incomes are from Nomis ASHE workplace analysis, covering all workers (part-time and full-time), from 2023.
* HMRC total income is from 2021.
