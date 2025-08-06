"""
VAT expenditure imputation using Effects of Taxes and Benefits data.

This module imputes household VAT expenditure rates based on demographic
characteristics using machine learning models trained on ETB survey data.
"""

import pandas as pd
from pathlib import Path
import numpy as np
from policyengine_uk_data.storage import STORAGE_FOLDER
from policyengine_uk.data import UKSingleYearDataset
from policyengine_uk import Microsimulation

ETB_TAB_FOLDER = STORAGE_FOLDER / "etb_1977_21"

CONSUMPTION_PCT_REDUCED_RATE = 0.03  # From OBR's VAT page
CURRENT_VAT_RATE = 0.2

PREDICTORS = ["is_adult", "is_child", "is_SP_age", "household_net_income"]
IMPUTATIONS = ["full_rate_vat_expenditure_rate"]


def generate_etb_table(etb: pd.DataFrame):
    """
    Clean and transform ETB data for VAT imputation model training.

    Args:
        etb: Raw ETB survey data DataFrame.

    Returns:
        Cleaned DataFrame with VAT expenditure rates calculated.
    """
    etb_2020 = etb[etb.year == 2020].dropna()
    for col in etb_2020:
        etb_2020[col] = pd.to_numeric(etb_2020[col], errors="coerce")

    etb_2020_df = pd.DataFrame()
    etb_2020_df["is_adult"] = etb_2020.adults
    etb_2020_df["is_child"] = etb_2020.childs
    etb_2020_df["is_SP_age"] = etb_2020.noretd
    etb_2020_df["household_net_income"] = etb_2020.disinc * 52
    etb_2020_df["full_rate_vat_expenditure_rate"] = (
        etb_2020.totvat * (1 - CONSUMPTION_PCT_REDUCED_RATE) / CURRENT_VAT_RATE
    ) / (etb_2020.expdis - etb_2020.totvat)
    return etb_2020_df[~etb_2020_df.full_rate_vat_expenditure_rate.isna()]


def save_imputation_models():
    """
    Train and save VAT imputation model.

    Returns:
        Trained QRF model for VAT imputation.
    """
    from policyengine_uk_data.utils.qrf import QRF

    vat = QRF()
    etb = pd.read_csv(
        ETB_TAB_FOLDER / "householdv2_1977-2021.tab",
        delimiter="\t",
        low_memory=False,
    )
    etb = generate_etb_table(etb)
    etb = etb[PREDICTORS + IMPUTATIONS]
    vat.fit(etb[PREDICTORS], etb[IMPUTATIONS])
    vat.save(STORAGE_FOLDER / "vat.pkl")
    return vat


def create_vat_model(overwrite_existing: bool = False):
    """
    Create or load VAT imputation model.

    Args:
        overwrite_existing: Whether to retrain model if it exists.

    Returns:
        QRF model for VAT expenditure imputation.
    """
    from policyengine_uk_data.utils.qrf import QRF

    if (STORAGE_FOLDER / "vat.pkl").exists() and not overwrite_existing:
        return QRF(file_path=STORAGE_FOLDER / "vat.pkl")
    return save_imputation_models()


def impute_vat(dataset: UKSingleYearDataset) -> UKSingleYearDataset:
    """
    Impute household VAT expenditure rates using trained model.

    Uses ETB-trained models to predict VAT expenditure rates for households
    based on demographic composition and income.

    Args:
        dataset: PolicyEngine UK dataset to augment with VAT data.

    Returns:
        Dataset with imputed VAT expenditure variables added to household table.
    """
    # Impute wealth, assuming same time period as trained data
    dataset = dataset.copy()

    model = create_vat_model()
    sim = Microsimulation(dataset=dataset)
    predictors = model.input_columns

    input_df = sim.calculate_dataframe(predictors, map_to="household")

    output_df = model.predict(input_df)

    for column in output_df.columns:
        dataset.household[column] = output_df[column].values

    dataset.validate()

    return dataset
