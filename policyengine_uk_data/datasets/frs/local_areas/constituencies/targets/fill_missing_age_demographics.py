import pandas as pd
import numpy as np

ages = pd.read_csv("age.csv")
incomes = pd.read_csv("total_income.csv")

ENGLAND_CONSTITUENCY = "E14"
NI_CONSTITUENCY = "N06"
SCOTLAND_CONSTITUENCY = "S14"
WALES_CONSTITUENCY = "W07"

incomes = incomes[
    np.any(
        [
            incomes["code"].str.contains(country_code)
            for country_code in [
                ENGLAND_CONSTITUENCY,
                NI_CONSTITUENCY,
                SCOTLAND_CONSTITUENCY,
                WALES_CONSTITUENCY,
            ]
        ],
        axis=0,
    )
]

full_constituencies = incomes.code
missing_constituencies = pd.Series(list(set(incomes.code) - set(ages.code)))
missing_constituencies = pd.DataFrame(
    {
        "code": missing_constituencies.values,
        "name": incomes.set_index("code")
        .loc[missing_constituencies]
        .name.values,
    }
)
for col in ages.columns[2:]:
    # We only have England and Wales demographics- fill in the remaining with the average age profiles among the rest of the UK.
    missing_constituencies[col] = ages[col].mean()

ages = pd.concat([ages, missing_constituencies])
ages.to_csv("age.csv", index=False)
