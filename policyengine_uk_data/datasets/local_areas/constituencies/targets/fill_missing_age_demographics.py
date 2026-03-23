import pandas as pd
import numpy as np

ages = pd.read_csv("raw_age.csv")
incomes = pd.read_csv("total_income.csv")

ENGLAND_CONSTITUENCY = "E14"
NI_CONSTITUENCY = "N06"
SCOTLAND_CONSTITUENCY = "S14"
WALES_CONSTITUENCY = "W07"

COUNTRY_PREFIXES = {
    "E": ENGLAND_CONSTITUENCY,
    "N": NI_CONSTITUENCY,
    "S": SCOTLAND_CONSTITUENCY,
    "W": WALES_CONSTITUENCY,
}

incomes = incomes[
    np.any(
        [
            incomes["code"].str.contains(country_code)
            for country_code in COUNTRY_PREFIXES.values()
        ],
        axis=0,
    )
]

missing_codes = set(incomes.code) - set(ages.code)
missing_constituencies = pd.DataFrame(
    {
        "code": list(missing_codes),
        "name": incomes.set_index("code").loc[list(missing_codes)].name.values,
    }
)

age_cols = ages.columns[2:]

# Use country-specific mean profiles instead of UK-wide mean (#64).
# For each missing constituency, find areas in the same country and
# use their mean age profile. Falls back to UK-wide mean if no areas
# exist for that country.
for _, row in missing_constituencies.iterrows():
    country_letter = row["code"][0]
    same_country = ages[ages["code"].str.startswith(country_letter)]
    if len(same_country) > 0:
        for col in age_cols:
            missing_constituencies.loc[
                missing_constituencies["code"] == row["code"], col
            ] = same_country[col].mean()
    else:
        for col in age_cols:
            missing_constituencies.loc[
                missing_constituencies["code"] == row["code"], col
            ] = ages[col].mean()

ages = pd.concat([ages, missing_constituencies])
ages.to_csv("age.csv", index=False)
