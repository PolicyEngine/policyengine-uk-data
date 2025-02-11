import pandas as pd

df = pd.read_csv("spi_constituency_raw.csv", encoding="utf-8")

# Table 3.15 of the SPI, with columns renamed to code,name,self_employment_income_count,self_employment_income_mean,...

df = df.dropna()
df = df[~df.code.str.contains("E1200000")]
df = df[~df.code.str.contains("W9200000")]
df = df[~df.code.str.contains("S9200000")]
df = df[~df.code.str.contains("N9200000")]
df = df.sort_values("code")

for col in df.columns[2:]:
    df[col] = df[col].astype(str).str.replace(",", "").str.replace("£", "").str.replace("-", "0").astype(float)

    if "_count" in col:
        df[col] = df[col] * 1_000
    
    elif "_mean" in col:
        income = col.split("_mean")[0]
        df[income + "_amount"] = df[col] * df[income + "_count"]

df.to_csv("spi_by_constituency.csv", index=False)