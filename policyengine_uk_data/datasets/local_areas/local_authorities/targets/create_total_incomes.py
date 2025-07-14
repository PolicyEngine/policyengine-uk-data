import pandas as pd
import numpy as np

df = pd.read_csv("spi_la_raw.csv", encoding="utf-8")
age = pd.read_csv("age.csv", encoding="utf-8")

# Table 3.15 of the SPI, with columns renamed to code,name,self_employment_income_count,self_employment_income_mean,...

df = df.dropna()
df = df[df.code.apply(lambda code: code in age.code.values)]
df = df.sort_values("code")

for code in age.code.values:
    if code not in df.code.values:
        df = pd.concat(
            [
                df,
                pd.DataFrame(
                    {
                        "code": [code],
                        "name": [age[age.code == code].name.values[0]],
                        **{
                            col: [np.nan]
                            for col in df.columns
                            if col not in ["code", "name"]
                        },
                    }
                ),
            ]
        )

for col in df.columns[2:]:
    df[col] = (
        df[col]
        .astype(str)
        .str.replace(",", "")
        .str.replace("£", "")
        .str.replace("-", "0")
        .replace("[Not available]", np.nan)
    )
    avg = df[col].dropna().astype(float).mean()
    df[col] = df[col].fillna(avg).astype(float)

    if "_count" in col:
        df[col] = df[col] * 1_000

    elif "_mean" in col:
        income = col.split("_mean")[0]
        df[income + "_amount"] = df[col] * df[income + "_count"]

df.sort_values("code").to_csv("spi_by_la.csv", index=False)
