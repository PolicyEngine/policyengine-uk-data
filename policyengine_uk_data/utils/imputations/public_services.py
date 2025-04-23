from policyengine_uk import Microsimulation
import pandas as pd
import numpy as np
from policyengine_uk_data.utils.qrf import QRF

sim = Microsimulation(
    dataset="hf://policyengine/policyengine-uk-data/enhanced_frs_2022_23.h5"
)

df = sim.calculate_dataframe(
    [
        "household_weight",
        "household_id",
        "is_adult",
        "is_child",
        "is_SP_age",
        "dla",
        "pip",
        "hbai_household_net_income",
    ],
    period=2025,
)

education = sim.calculate("current_education", period=2025)
df["count_primary_education"] = sim.map_result(
    education == "PRIMARY", "person", "household"
)
df["count_secondary_education"] = sim.map_result(
    education == "LOWER_SECONDARY", "person", "household"
)
df["count_further_education"] = sim.map_result(
    education.isin(["UPPER_SECONDARY", "TERTIARY"]), "person", "household"
)


etb = pd.read_csv(
    "~/Downloads/UKDA-8856-tab 2/tab/householdv2_1977-2021.tab", delimiter="\t"
)
etb = etb[etb.year == etb.year.max()]
etb = etb.replace(" ", np.nan)

etb = etb[
    [
        "adults",
        "childs",
        "disinc",
        "benk",
        "educ",
        "totnhs",
        "rail",
        "bussub",
        "hsub",
        "hhold_adj_weight",
        "noretd",
        "primed",
        "secoed",
        "wagern",
        "welf",
        "furted",
        "disliv",
        "pips",
    ]
]
etb = etb.dropna().astype(float)
model = QRF()

WEEKS_IN_YEAR = 52


train = pd.DataFrame()
train["is_adult"] = etb.adults
train["is_child"] = etb.childs
train["hbai_household_net_income"] = etb.disinc * WEEKS_IN_YEAR
train["is_SP_age"] = etb.noretd
train["count_primary_education"] = etb.primed
train["count_secondary_education"] = etb.secoed
train["count_further_education"] = etb.furted
train["dla"] = etb.disliv
train["pip"] = etb.pips
train["public_service_in_kind_value"] = etb.benk * WEEKS_IN_YEAR
train["education_service_in_kind_value"] = etb.educ * WEEKS_IN_YEAR
train["nhs_in_kind_value"] = etb.totnhs * WEEKS_IN_YEAR
train["rail_subsidy_in_kind_value"] = etb.rail * WEEKS_IN_YEAR
train["bus_subsidy_in_kind_value"] = etb.bussub * WEEKS_IN_YEAR

PREDICTORS = [
    "is_adult",
    "is_child",
    "is_SP_age",
    "count_primary_education",
    "count_secondary_education",
    "count_further_education",
    "dla",
    "pip",
    "hbai_household_net_income",
]
OUTPUTS = [
    "public_service_in_kind_value",
    "education_service_in_kind_value",
    "nhs_in_kind_value",
    "rail_subsidy_in_kind_value",
    "bus_subsidy_in_kind_value",
]

model.fit(X=train[PREDICTORS], y=train[OUTPUTS])

outputs = model.predict(df[PREDICTORS])
