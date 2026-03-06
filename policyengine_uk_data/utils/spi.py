# This script generates SPI-based statistics on income types from public SPI publications.
# There are two tables needed:
# Profit, employment and pension income: https://www.gov.uk/government/statistics/earned-income-2010-to-2011
# Property, interest and dividend income: https://www.gov.uk/government/statistics/investment-income-2010-to-2011
# Minor step to avoid an ods dependency: put the ODS files into a CSV file with the header being the table header, then point the below references to those CSVs.

# Python script to extract income statistics from SPI .ods files.
import pandas as pd
from policyengine_uk_data.storage import STORAGE_FOLDER

earned_income = pd.read_csv("~/Downloads/spi_earned_income.csv")
unearned_income = pd.read_csv("~/Downloads/spi_unearned_income.csv")

RENAMED_EARNED_INCOME_COLUMNS = [
    "income_range",
    "self_employment_income_count",
    "self_employment_income_amount",
    "self_employment_income_mean",
    "employment_income_count",
    "employment_income_amount",
    "employment_income_mean",
    "state_pension_count",
    "state_pension_amount",
    "state_pension_mean",
    "private_pension_income_count",
    "private_pension_income_amount",
    "private_pension_income_mean",
]
RENAMED_UNEARNED_INCOME_COLUMNS = [
    "income_range",
    "property_income_count",
    "property_income_amount",
    "property_income_mean",
    "savings_interest_income_count",
    "savings_interest_income_amount",
    "savings_interest_income_mean",
    "dividend_income_count",
    "dividend_income_amount",
    "dividend_income_mean",
    # Other income is ignored here- difficult to reconcile with FRS data and it's only £3bn.
]

earned_income = earned_income[
    earned_income.columns[: len(RENAMED_EARNED_INCOME_COLUMNS)]
]
earned_income.columns = RENAMED_EARNED_INCOME_COLUMNS
unearned_income = unearned_income[
    unearned_income.columns[: len(RENAMED_UNEARNED_INCOME_COLUMNS)]
]
unearned_income.columns = RENAMED_UNEARNED_INCOME_COLUMNS

# Join on income range
income = pd.merge(earned_income, unearned_income, on="income_range").dropna()


def parse_value(value):
    if value == "All ranges":
        return "All ranges"
    return int(value.replace(",", ""))


income = income[[col for col in income.columns if "_mean" not in col]]

for column in income:
    income[column] = income[column].apply(parse_value)
    if "_amount" in column:
        income[column] = income[column] * 1e6
    elif "_count" in column:
        income[column] = income[column] * 1e3

import numpy as np

income["total_income_lower_bound"] = list(income["income_range"][:-1]) + [12_570]
income["total_income_upper_bound"] = list(income["income_range"][1:-1]) + [np.inf] * 2
# Order the income bound columns first
income = income[
    [
        "total_income_lower_bound",
        "total_income_upper_bound",
    ]
    + [col for col in income.columns if "income_range" not in col]
]

income.to_csv(STORAGE_FOLDER / "incomes.csv", index=False)
