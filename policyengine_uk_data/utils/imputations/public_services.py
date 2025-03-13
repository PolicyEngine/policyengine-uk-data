"""
Imputation model for public services received by households.

This module creates a quantile regression forest model to predict the value of
public services received by households based on demographic characteristics.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from policyengine_uk import Microsimulation
from policyengine_uk_data.utils.qrf import QRF
from policyengine_uk_data.storage import STORAGE_FOLDER

# Constants
WEEKS_IN_YEAR = 52

# Variables used to predict public service receipt
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

# Public service variables to impute
OUTPUTS = [
    "public_service_in_kind_value",
    "education_service_in_kind_value",
    "nhs_in_kind_value",
    "rail_subsidy_in_kind_value",
    "bus_subsidy_in_kind_value",
]


def create_inference_df(sim: Microsimulation, period: int = 2025) -> pd.DataFrame:
    """
    Create a dataframe with predictors needed for public service imputation.
    
    Args:
        sim: A PolicyEngine UK microsimulation.
        period: The year to calculate variables for.
        
    Returns:
        DataFrame with household-level predictors.
    """
    # Calculate basic household characteristics
    df = sim.calculate_dataframe([
        "household_weight", "household_id", "is_adult", "is_child", 
        "is_SP_age", "dla", "pip", "hbai_household_net_income"
    ], period=period)
    
    # Count people in education by level
    education = sim.calculate("current_education", period=period)
    df["count_primary_education"] = sim.map_result(education == "PRIMARY", "person", "household")
    df["count_secondary_education"] = sim.map_result(education == "LOWER_SECONDARY", "person", "household")
    df["count_further_education"] = sim.map_result(education.isin(["UPPER_SECONDARY", "TERTIARY"]), "person", "household")
    
    return df


def create_public_services_model(overwrite_existing: bool = False) -> None:
    """
    Create and save a model for imputing public service receipt values.
    
    Args:
        overwrite_existing: Whether to overwrite an existing model file.
    """
    # Check if model already exists and we're not overwriting
    if (STORAGE_FOLDER / "public_services.pkl").exists() and not overwrite_existing:
        return
    
    # Load Effects of Taxes and Benefits (ETB) dataset
    etb = pd.read_csv("~/Downloads/UKDA-8856-tab 2/tab/householdv2_1977-2021.tab", delimiter="\t")
    etb = etb[etb.year == etb.year.max()]  # Use most recent year
    etb = etb.replace(" ", np.nan)
    
    # Select relevant columns
    etb = etb[[
        "adults", "childs", "disinc", "benk", "educ", "totnhs", "rail", "bussub",
        "hsub", "hhold_adj_weight", "noretd", "primed", "secoed", "wagern", 
        "welf", "furted", "disliv", "pips"
    ]]
    etb = etb.dropna().astype(float)
    
    # Prepare training data
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
    
    # Output variables (annualized)
    train["public_service_in_kind_value"] = etb.benk * WEEKS_IN_YEAR
    train["education_service_in_kind_value"] = etb.educ * WEEKS_IN_YEAR
    train["nhs_in_kind_value"] = etb.totnhs * WEEKS_IN_YEAR
    train["rail_subsidy_in_kind_value"] = etb.rail * WEEKS_IN_YEAR
    train["bus_subsidy_in_kind_value"] = etb.bussub * WEEKS_IN_YEAR
    
    # Train model
    model = QRF()
    model.fit(X=train[PREDICTORS], y=train[OUTPUTS])
    
    # Save model
    model.save(STORAGE_FOLDER / "public_services.pkl")


if __name__ == "__main__":
    create_public_services_model()