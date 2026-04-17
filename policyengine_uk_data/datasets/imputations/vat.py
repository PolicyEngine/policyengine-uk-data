"""
VAT expenditure imputation using Effects of Taxes and Benefits data.

This module imputes household VAT expenditure rates based on demographic
characteristics using machine learning models trained on ETB survey data.

The ETB VAT columns report the standard-rate VAT actually paid plus a
reduced-rate share of expenditure. To back out the underlying
full-rate-taxable expenditure we divide by the statutory VAT standard
rate and subtract an OBR-published reduced-rate share of consumption.
Both are parameterised per-year so later years (or forthcoming rate
changes) don't need a code edit.
"""

import pandas as pd
from pathlib import Path
import numpy as np
from policyengine_uk_data.storage import STORAGE_FOLDER
from policyengine_uk.data import UKSingleYearDataset
from policyengine_uk import Microsimulation

ETB_TAB_FOLDER = STORAGE_FOLDER / "etb_1977_21"

# Default ETB vintage used when training the imputation model. Kept at 2020
# for backward compatibility with the checked-in vat.pkl fingerprint, but
# exposed as a module constant rather than an inline magic number so later
# updates require only a one-line change (not scattered `etb.year == 2020`
# checks).
DEFAULT_ETB_YEAR = 2020

# Fallback VAT parameters used when `policyengine_uk` is unavailable (e.g.
# unit-test environments). Values match the 2020-21 UK statutory position.
_FALLBACK_VAT_STANDARD_RATE = 0.2
_FALLBACK_REDUCED_RATE_SHARE = 0.03

# Manual year → (standard rate, reduced rate share) override used when
# `policyengine_uk` parameters are not available. Kept intentionally short:
# extend only if the team agrees that a VAT code change warrants a hardcoded
# value until the parameter file is updated upstream.
VAT_RATE_BY_YEAR: dict[int, tuple[float, float]] = {
    2020: (0.2, 0.03),
    2021: (0.2, 0.03),
}

PREDICTORS = ["is_adult", "is_child", "is_SP_age", "household_net_income"]
IMPUTATIONS = ["full_rate_vat_expenditure_rate"]


def _get_vat_parameters(year: int) -> tuple[float, float]:
    """Return ``(standard_rate, reduced_rate_share)`` for the given calendar year.

    Prefers live `policyengine_uk` parameters (``gov.hmrc.vat.standard_rate``
    and ``gov.hmrc.vat.reduced_rate_share``). Falls back to the module-level
    ``VAT_RATE_BY_YEAR`` dict, and finally to the 2020-21 statutory values so
    callers never silently get wrong numbers.
    """
    try:
        from policyengine_uk.system import system

        standard_rate = float(
            system.parameters.gov.hmrc.vat.standard_rate(str(year))
        )
        reduced_rate_share = float(
            system.parameters.gov.hmrc.vat.reduced_rate_share(str(year))
        )
        return standard_rate, reduced_rate_share
    except Exception:
        if year in VAT_RATE_BY_YEAR:
            return VAT_RATE_BY_YEAR[year]
        return _FALLBACK_VAT_STANDARD_RATE, _FALLBACK_REDUCED_RATE_SHARE


def generate_etb_table(
    etb: pd.DataFrame, year: int = DEFAULT_ETB_YEAR
) -> pd.DataFrame:
    """
    Clean and transform ETB data for VAT imputation model training.

    Args:
        etb: Raw ETB survey data DataFrame.
        year: ETB survey year to filter to. Defaults to ``DEFAULT_ETB_YEAR``.

    Returns:
        Cleaned DataFrame with VAT expenditure rates calculated.
    """
    standard_rate, reduced_rate_share = _get_vat_parameters(year)

    etb_year = etb[etb.year == year].dropna()
    for col in etb_year:
        etb_year[col] = pd.to_numeric(etb_year[col], errors="coerce")

    etb_year_df = pd.DataFrame()
    etb_year_df["is_adult"] = etb_year.adults
    etb_year_df["is_child"] = etb_year.childs
    etb_year_df["is_SP_age"] = etb_year.noretd
    etb_year_df["household_net_income"] = etb_year.disinc * 52
    etb_year_df["full_rate_vat_expenditure_rate"] = (
        etb_year.totvat * (1 - reduced_rate_share) / standard_rate
    ) / (etb_year.expdis - etb_year.totvat)
    return etb_year_df[~etb_year_df.full_rate_vat_expenditure_rate.isna()]


def save_imputation_models(year: int = DEFAULT_ETB_YEAR):
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
    etb = generate_etb_table(etb, year=year)
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
