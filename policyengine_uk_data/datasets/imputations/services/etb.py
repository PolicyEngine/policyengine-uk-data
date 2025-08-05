"""
Imputation model for public services received by households.

This module creates a quantile regression forest model to predict the value of
public services received by households based on demographic characteristics.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from policyengine_uk import Microsimulation
from huggingface_hub import hf_hub_download
import os
from policyengine_uk_data.storage import STORAGE_FOLDER
from policyengine_uk_data.utils.qrf import QRF
from policyengine_uk.data import UKSingleYearDataset

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
    "dfe_education_spending",
    "rail_subsidy_spending",
    "bus_subsidy_spending",
]


def create_public_services_model(overwrite_existing: bool = False) -> None:
    """
    Create and save a model for imputing public service receipt values.

    Args:
        overwrite_existing: Whether to overwrite an existing model file.
    """
    # Check if model already exists and we're not overwriting
    if (
        STORAGE_FOLDER / "public_services.pkl"
    ).exists() and not overwrite_existing:
        return

    etb_path = STORAGE_FOLDER / "etb_1977_21" / "householdv2_1977-2021.tab"

    # Load Effects of Taxes and Benefits (ETB) dataset
    etb = pd.read_csv(etb_path, delimiter="\t")
    etb = etb[etb.year == etb.year.max()]  # Use most recent year
    etb = etb.replace(" ", np.nan)

    # Select relevant columns
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
    train["dfe_education_spending"] = etb.educ * WEEKS_IN_YEAR
    train["rail_subsidy_spending"] = etb.rail * WEEKS_IN_YEAR
    train["bus_subsidy_spending"] = etb.bussub * WEEKS_IN_YEAR

    # Train model
    model = QRF()
    model.fit(X=train[PREDICTORS], y=train[OUTPUTS])

    return model


def impute_public_services(efrs: pd.DataFrame) -> pd.DataFrame:
    """
    Impute public services received by households.

    Args:
        efrs: DataFrame containing household data.

    Returns:
        DataFrame with imputed public service values.
    """
    # Create model if it doesn't exist
    model = create_public_services_model()

    efrs_h = efrs.groupby("household_id").sum()
    household_count_person = efrs_h.is_adult.values + efrs_h.is_child.values

    # Impute public services
    efrs_h[OUTPUTS] = model.predict(efrs_h[PREDICTORS]).values

    for output in OUTPUTS:
        efrs_h[output] /= household_count_person
        efrs[output] = efrs_h[output].loc[efrs["household_id"].values].values

    return efrs


def create_efrs_input_dataset(dataset: UKSingleYearDataset) -> pd.DataFrame:
    sim = Microsimulation(
        dataset=dataset,
    )

    variables = [
        "age",
        "gender",
        "household_weight",
        "region",
        "household_id",
        "is_adult",
        "is_child",
        "is_SP_age",
        "dla",
        "pip",
        "household_count_people",
        "hbai_household_net_income",
        "equiv_hbai_household_net_income",
    ]
    education = sim.calculate("current_education")

    df = sim.calculate_dataframe(variables)

    df["count_primary_education"] = education == "PRIMARY"
    df["count_secondary_education"] = education == "LOWER_SECONDARY"
    df["count_further_education"] = education.isin(
        ["UPPER_SECONDARY", "TERTIARY"]
    )
    df["hbai_household_net_income"] = (
        df["hbai_household_net_income"] / df["household_count_people"]
    )
    df["equiv_hbai_household_net_income"] = (
        df["equiv_hbai_household_net_income"] / df["household_count_people"]
    )

    data = pd.DataFrame(df)
    return data
