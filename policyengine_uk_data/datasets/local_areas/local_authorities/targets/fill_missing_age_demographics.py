import pandas as pd

# Read the files
df_age = pd.read_csv("raw_age.csv", skiprows=6, thousands=",")

# Rename age columns
columns = ["name", "code"]
age_columns = [str(i) for i in range(90)] + ["90+"]
df_age.columns = columns + age_columns

# Convert age columns to numeric
for col in age_columns:
    df_age[col] = pd.to_numeric(df_age[col], errors="coerce")

# Delete last row from age file before any processing
df_age = df_age.iloc[:-1]

# Read total_income file
df_income = pd.read_csv("total_income.csv")

# Convert income columns to numeric if they exist
if "total_income_count" in df_income.columns:
    df_income["total_income_count"] = pd.to_numeric(
        df_income["total_income_count"], errors="coerce"
    )

if "total_income_amount" in df_income.columns:
    df_income["total_income_amount"] = pd.to_numeric(
        df_income["total_income_amount"], errors="coerce"
    )

# Find common codes between both files
common_codes = set(df_age["code"]).intersection(set(df_income["code"]))

# Keep only common codes in both dataframes
df_age = df_age[df_age["code"].isin(common_codes)]
df_income = df_income[df_income["code"].isin(common_codes)]

# Fill missing age values using country-specific means instead of UK-wide
# mean, so Scotland and NI areas get their own age profiles (#64).
for col in age_columns:
    missing_mask = df_age[col].isna()
    if missing_mask.any():
        for _, row in df_age[missing_mask].iterrows():
            country_letter = row["code"][0]
            same_country = df_age[
                (df_age["code"].str.startswith(country_letter)) & ~df_age[col].isna()
            ]
            if len(same_country) > 0:
                fill_value = same_country[col].mean()
            else:
                fill_value = df_age[col].mean()
            df_age.loc[df_age["code"] == row["code"], col] = fill_value

# Calculate 'all' as sum of all age columns after filling missing values
df_age["all"] = df_age[age_columns].sum(axis=1)

# Reorder columns to match desired format
final_columns = ["code", "name", "all"] + age_columns
df_age = df_age[final_columns]

df_age = df_age.sort_values("code")
df_income = df_income.sort_values("code")

# Save files
df_age.to_csv("age.csv", index=False)
df_income.to_csv("total_income.csv", index=False)
