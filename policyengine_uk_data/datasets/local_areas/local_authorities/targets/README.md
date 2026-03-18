# Data

* Age data is from the [ONS mid-year population estimates](https://www.nomisweb.co.uk/datasets/pestsyoala), with single-year age counts for each local authority district. The raw download should be saved as `raw_age.csv`; running `fill_missing_age_demographics.py` fills missing values and intersects with the income file, writing the processed output to `age.csv`.
* Employment incomes are from Nomis, and are from 2023.
* HMRC total income is from 2021.
