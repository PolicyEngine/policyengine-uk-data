"""
Consumption imputation using Living Costs and Food Survey data.

This module imputes household consumption patterns (including fuel spending)
using QRF models trained on LCFS data, with vehicle ownership information
from the Wealth and Assets Survey to improve fuel spending predictions.

Key innovation: We impute `has_fuel_consumption` to WAS based on vehicle
ownership, then use this to bridge WAS and LCFS for fuel spending imputation.
This addresses the issue that LCFS 2-week diaries undercount fuel purchases
(58% have any fuel) vs actual vehicle ownership (78% per NTS 2024).
"""

import pandas as pd
from pathlib import Path
import numpy as np
import yaml
from policyengine_uk_data.storage import STORAGE_FOLDER
from policyengine_uk.data import UKSingleYearDataset
from policyengine_uk import Microsimulation
from policyengine_uk_data.utils.stack import stack_datasets

LCFS_TAB_FOLDER = STORAGE_FOLDER / "lcfs_2021_22"

# EV/ICE vehicle mix from NTS 2024
# Source: https://www.gov.uk/government/statistics/national-travel-survey-2024
# "Around 59% of cars people owned were petrol, 30% were diesel, 6% hybrid,
#  4% battery electric and 2% plug-in hybrid."
# ICE share = 59% + 30% = 89%, plus hybrids still use some fuel
# We use 90% as the probability a vehicle owner buys petrol/diesel
NTS_2024_ICE_VEHICLE_SHARE = 0.90

REGIONS = {
    1: "NORTH_EAST",
    2: "NORTH_WEST",
    3: "YORKSHIRE",
    4: "EAST_MIDLANDS",
    5: "WEST_MIDLANDS",
    6: "EAST_OF_ENGLAND",
    7: "LONDON",
    8: "SOUTH_EAST",
    9: "SOUTH_WEST",
    10: "WALES",
    11: "SCOTLAND",
    12: "NORTHERN_IRELAND",
}

HOUSEHOLD_LCF_RENAMES = {
    "G018": "is_adult",
    "G019": "is_child",
    "Gorx": "region",
    "P389p": "household_net_income",
    "weighta": "household_weight",
}
PERSON_LCF_RENAMES = {
    "B303p": "employment_income",
    "B3262p": "self_employment_income",
    "B3381": "state_pension",
    "P049p": "private_pension_income",
}

CONSUMPTION_VARIABLE_RENAMES = {
    "P601": "food_and_non_alcoholic_beverages_consumption",
    "P602": "alcohol_and_tobacco_consumption",
    "P603": "clothing_and_footwear_consumption",
    "P604": "housing_water_and_electricity_consumption",
    "P605": "household_furnishings_consumption",
    "P606": "health_consumption",
    "P607": "transport_consumption",
    "P608": "communication_consumption",
    "P609": "recreation_consumption",
    "P610": "education_consumption",
    "P611": "restaurants_and_hotels_consumption",
    "P612": "miscellaneous_consumption",
    "C72211": "petrol_spending",
    "C72212": "diesel_spending",
    "P537": "domestic_energy_consumption",
}


PREDICTOR_VARIABLES = [
    "is_adult",
    "is_child",
    "region",
    "employment_income",
    "self_employment_income",
    "private_pension_income",
    "household_net_income",
    "has_fuel_consumption",  # Imputed from WAS vehicle ownership
]

IMPUTATIONS = [
    "food_and_non_alcoholic_beverages_consumption",
    "alcohol_and_tobacco_consumption",
    "clothing_and_footwear_consumption",
    "housing_water_and_electricity_consumption",
    "household_furnishings_consumption",
    "health_consumption",
    "transport_consumption",
    "communication_consumption",
    "recreation_consumption",
    "education_consumption",
    "restaurants_and_hotels_consumption",
    "miscellaneous_consumption",
    "petrol_spending",
    "diesel_spending",
    "domestic_energy_consumption",
]


def create_has_fuel_model():
    """
    Train a model to predict has_fuel_consumption from demographics.

    Uses WAS vehicle ownership to create has_fuel_consumption:
    - Households with vehicles have ~90% chance of fuel consumption (ICE vehicles)
    - Households without vehicles have ~0% chance

    This bridges the gap between:
    - LCFS: 58% of households recorded fuel in 2-week diary
    - NTS 2024: 78% of households have vehicles

    Sources:
    - NTS 2024 vehicle ownership: https://www.gov.uk/government/statistics/
      national-travel-survey-2024/nts-2024-household-car-availability-and-trends
      "22% of households had no vehicle, 44% one vehicle, 34% two or more"
    - NTS 2024 fuel type: "59% petrol, 30% diesel, 6% hybrid, 4% BEV, 2% PHEV"
      So ~90% of vehicle owners use petrol/diesel (ICE + hybrids)

    Returns:
        QRF model predicting has_fuel_consumption from demographics.
    """
    from policyengine_uk_data.utils.qrf import QRF
    from policyengine_uk_data.datasets.imputations.wealth import (
        WAS_TAB_FOLDER,
        REGIONS,
    )

    model_path = STORAGE_FOLDER / "has_fuel_model.pkl"
    if model_path.exists():
        return QRF(file_path=model_path)

    # Load WAS with vehicle ownership
    was = pd.read_csv(
        WAS_TAB_FOLDER / "was_round_7_hhold_eul_march_2022.tab",
        sep="\t",
        low_memory=False,
    )
    was.columns = [c.lower() for c in was.columns]

    # Create has_fuel_consumption from vehicle ownership
    # Vehicle owners have 90% chance (ICE vehicles), non-owners have 0%
    num_vehicles = was["vcarnr7"].fillna(0).clip(lower=0)
    has_vehicle = num_vehicles > 0

    # Randomly assign fuel consumption based on ICE share
    # This simulates that ~10% of vehicle owners have EVs/PHEVs
    np.random.seed(42)  # Reproducibility
    is_ice_vehicle = np.random.random(len(was)) < NTS_2024_ICE_VEHICLE_SHARE
    has_fuel = (has_vehicle & is_ice_vehicle).astype(float)

    # Build training DataFrame with predictors available in LCFS
    was_df = pd.DataFrame(
        {
            "household_net_income": was["dvtotinc_bhcr7"],
            "num_adults": was["numadultr7"],
            "num_children": was["numch18r7"],
            "private_pension_income": was["dvgippenr7_aggr"],
            "employment_income": was["dvgiempr7_aggr"],
            "self_employment_income": was["dvgiser7_aggr"],
            "region": was["gorr7"].map(REGIONS),
            "has_fuel_consumption": has_fuel,
        }
    ).dropna()

    predictors = [
        "household_net_income",
        "num_adults",
        "num_children",
        "private_pension_income",
        "employment_income",
        "self_employment_income",
        "region",
    ]

    model = QRF()
    model.fit(was_df[predictors], was_df[["has_fuel_consumption"]])
    model.save(model_path)
    return model


def impute_has_fuel_to_lcfs(household: pd.DataFrame) -> pd.DataFrame:
    """
    Impute has_fuel_consumption to LCFS households using WAS-trained model.

    This provides a consistent fuel consumption indicator based on vehicle
    ownership patterns, rather than relying on the LCFS 2-week diary which
    underestimates fuel purchasers (58% vs 78% vehicle ownership).
    """
    model = create_has_fuel_model()

    input_df = pd.DataFrame(
        {
            "household_net_income": household["household_net_income"],
            "num_adults": household["is_adult"],
            "num_children": household["is_child"],
            "private_pension_income": household["private_pension_income"],
            "employment_income": household["employment_income"],
            "self_employment_income": household["self_employment_income"],
            "region": household["region"],
        }
    )

    output_df = model.predict(input_df)
    # Clip to [0, 1] as it's a probability
    household["has_fuel_consumption"] = output_df[
        "has_fuel_consumption"
    ].values.clip(0, 1)

    return household


def generate_lcfs_table(
    lcfs_person: pd.DataFrame, lcfs_household: pd.DataFrame
):
    """
    Generate LCFS training table for consumption imputation.

    Processes raw LCFS data and imputes has_fuel_consumption from WAS
    vehicle ownership patterns to improve fuel spending predictions.
    """
    person = lcfs_person.rename(columns=PERSON_LCF_RENAMES)
    household = lcfs_household.rename(columns=HOUSEHOLD_LCF_RENAMES)
    household["region"] = household["region"].map(REGIONS)
    household = household.rename(columns=CONSUMPTION_VARIABLE_RENAMES)
    for variable in list(CONSUMPTION_VARIABLE_RENAMES.values()) + [
        "household_net_income"
    ]:
        household[variable] = household[variable] * 52
    for variable in PERSON_LCF_RENAMES.values():
        household[variable] = (
            person[variable].groupby(person.case).sum()[household.case] * 52
        )
    household.household_weight *= 1_000

    # Impute has_fuel_consumption from WAS vehicle ownership model
    # This bridges WAS (has vehicles) to LCFS (has fuel spending)
    household = impute_has_fuel_to_lcfs(household)

    return household[
        PREDICTOR_VARIABLES + IMPUTATIONS + ["household_weight"]
    ].dropna()


def uprate_lcfs_table(
    household: pd.DataFrame, time_period: str
) -> pd.DataFrame:
    from policyengine_uk.system import system

    start_period = 2021
    fuel_uprating = 1.3
    household["petrol_spending"] *= fuel_uprating
    household["diesel_spending"] *= fuel_uprating

    cpi = (
        system.parameters.gov.economic_assumptions.indices.obr.consumer_price_index
    )
    cpi_uprating = cpi(time_period) / cpi(start_period)

    for variable in IMPUTATIONS:
        if variable not in ["petrol_spending", "diesel_spending"]:
            household[variable] *= cpi_uprating
    return household


def save_imputation_models():
    from policyengine_uk_data.utils.qrf import QRF

    consumption = QRF()
    lcfs_household = pd.read_csv(
        LCFS_TAB_FOLDER / "lcfs_2021_dvhh_ukanon.tab",
        delimiter="\t",
        low_memory=False,
    )
    lcfs_person = pd.read_csv(
        LCFS_TAB_FOLDER / "lcfs_2021_dvper_ukanon202122.tab", delimiter="\t"
    )
    household = generate_lcfs_table(lcfs_person, lcfs_household)
    household = uprate_lcfs_table(household, "2024")
    consumption.fit(
        household[PREDICTOR_VARIABLES],
        household[IMPUTATIONS],
    )
    consumption.save(
        STORAGE_FOLDER / "consumption.pkl",
    )
    return consumption


def create_consumption_model(overwrite_existing: bool = False):
    from policyengine_uk_data.utils.qrf import QRF

    if (
        STORAGE_FOLDER / "consumption.pkl"
    ).exists() and not overwrite_existing:
        return QRF(file_path=STORAGE_FOLDER / "consumption.pkl")
    return save_imputation_models()


def impute_consumption(dataset: UKSingleYearDataset) -> UKSingleYearDataset:
    """
    Impute consumption variables using LCFS-trained model.

    Requires num_vehicles to be present in the dataset (from wealth imputation)
    to compute has_fuel_consumption.
    """
    dataset = dataset.copy()

    # First, compute has_fuel_consumption from num_vehicles
    # This uses the same logic as the WAS training data:
    # - Vehicle owners have 90% chance of fuel consumption (ICE vehicles)
    # - Non-owners have 0% chance
    sim = Microsimulation(dataset=dataset)
    num_vehicles = sim.calculate("num_vehicles", map_to="household").values

    np.random.seed(42)  # Match training data randomness
    has_vehicle = num_vehicles > 0
    is_ice = np.random.random(len(num_vehicles)) < NTS_2024_ICE_VEHICLE_SHARE
    has_fuel_consumption = (has_vehicle & is_ice).astype(float)
    dataset.household["has_fuel_consumption"] = has_fuel_consumption

    # Now run the consumption model with has_fuel_consumption as predictor
    model = create_consumption_model()
    predictors = model.input_columns

    input_df = sim.calculate_dataframe(
        [p for p in predictors if p != "has_fuel_consumption"],
        map_to="household",
    )
    input_df["has_fuel_consumption"] = has_fuel_consumption

    output_df = model.predict(input_df)

    for column in output_df.columns:
        dataset.household[column] = output_df[column].values

    dataset.validate()

    return dataset
