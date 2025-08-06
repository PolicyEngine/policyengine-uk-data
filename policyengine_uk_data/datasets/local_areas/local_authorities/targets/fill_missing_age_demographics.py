import pandas as pd
import numpy as np

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

# Calculate mean of each age column for non-missing values and fill missing values
for col in age_columns:
    df_age[col] = df_age[col].fillna(df_age[col].mean())

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
