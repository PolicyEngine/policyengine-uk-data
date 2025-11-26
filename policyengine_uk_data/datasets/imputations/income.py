"""
Income imputation using Survey of Personal Incomes data.

This module imputes detailed income components (employment, self-employment,
pensions, property, savings interest, dividends) using machine learning
models trained on HMRC Survey of Personal Incomes (SPI) data.
"""

import pandas as pd
from pathlib import Path
import numpy as np
from policyengine_uk_data.storage import STORAGE_FOLDER
from policyengine_uk.data import UKSingleYearDataset
from policyengine_uk import Microsimulation
from policyengine_uk_data.utils.stack import stack_datasets
from policyengine_uk_data.utils.subsample import subsample_dataset

SPI_TAB_FOLDER = STORAGE_FOLDER / "spi_2020_21"
SPI_RENAMES = dict(
    private_pension_income="PENSION",
    self_employment_income="PROFITS",
    property_income="INCPROP",
    savings_interest_income="INCBBS",
    dividend_income="DIVIDENDS",
    blind_persons_allowance="BPADUE",
    married_couples_allowance="MCAS",
    gift_aid="GIFTAID",
    capital_allowances="CAPALL",
    deficiency_relief="DEFICIEN",
    covenanted_payments="COVNTS",
    charitable_investment_gifts="GIFTINV",
    employment_expenses="EPB",
    other_deductions="MOTHDED",
    person_weight="FACT",
    benunit_weight="FACT",
    household_weight="FACT",
    state_pension="SRP",
)


def generate_spi_table(spi: pd.DataFrame):
    """
    Clean and transform SPI data for income imputation model training.

    Args:
        spi: Raw SPI survey data DataFrame.

    Returns:
        Cleaned DataFrame with age and region mappings applied.
    """
    LOWER = np.array([0, 16, 25, 35, 45, 55, 65, 75])
    UPPER = np.array([16, 25, 35, 45, 55, 65, 75, 80])
    age_range = spi.AGERANGE
    spi["age"] = LOWER[age_range] + np.random.rand(len(spi)) * (
        UPPER[age_range] - LOWER[age_range]
    )

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

    spi["region"] = np.array([REGIONS.get(x, "LONDON") for x in spi.GORCODE])

    spi["gender"] = np.where(spi.SEX == 1, "MALE", "FEMALE")

    for rename in SPI_RENAMES:
        spi[rename] = spi[SPI_RENAMES[rename]]

    spi["employment_income"] = spi[["PAY", "EPB", "TAXTERM"]].sum(axis=1)

    spi = pd.concat(
        [
            spi.sample(100_000, weights=spi.person_weight),
        ]
    )

    return spi


PREDICTORS = [
    "age",
    "gender",
    "region",
]

IMPUTATIONS = [
    "employment_income",
    "self_employment_income",
    "savings_interest_income",
    "dividend_income",
    "private_pension_income",
    "property_income",
]


def save_imputation_models():
    """
    Train and save income imputation model.

    Returns:
        Trained QRF model for income imputation.
    """
    from policyengine_uk_data.utils import QRF

    income = QRF()
    spi = pd.read_csv(SPI_TAB_FOLDER / "put2021uk.tab", delimiter="\t")
    spi = generate_spi_table(spi)
    spi = spi[PREDICTORS + IMPUTATIONS]
    income.fit(spi[PREDICTORS], spi[IMPUTATIONS])
    income.save(STORAGE_FOLDER / "income.pkl")
    return income


def create_income_model(overwrite_existing: bool = False):
    """
    Create or load income imputation model.

    Args:
        overwrite_existing: Whether to retrain model if it exists.

    Returns:
        QRF model for income imputation.
    """
    from policyengine_uk_data.utils.qrf import QRF

    if (STORAGE_FOLDER / "income.pkl").exists() and not overwrite_existing:
        return QRF(file_path=STORAGE_FOLDER / "income.pkl")
    return save_imputation_models()


def impute_over_incomes(
    dataset: UKSingleYearDataset, model, output_variables: list[str]
) -> pd.DataFrame:
    """
    Impute specified income components using trained model.

    Args:
        dataset: PolicyEngine UK dataset to augment with income data.
        output_variables: List of income components to impute.

    Returns:
        DataFrame with imputed income components.
    """
    dataset = dataset.copy()
    input_df = Microsimulation(dataset=dataset).calculate_dataframe(
        ["age", "gender", "region"]
    )
    output_df = model.predict(input_df)

    for column in output_variables:
        dataset.person[column] = output_df[column].fillna(0).values

    return dataset


def impute_income(dataset: UKSingleYearDataset) -> UKSingleYearDataset:
    """
    Impute detailed income components using trained model.

    Uses SPI-trained models to predict various income sources for individuals
    based on age, gender, and region. Creates a synthetic population with
    the imputed income data.

    Args:
        dataset: PolicyEngine UK dataset to augment with income data.

    Returns:
        Combined dataset with original data plus synthetic high-income individuals.
    """
    # Impute wealth, assuming same time period as trained data
    dataset = dataset.copy()
    zero_weight_copy = dataset.copy()
    zero_weight_copy.household.household_weight = 0
    zero_weight_copy = subsample_dataset(zero_weight_copy, 10_000)

    model = create_income_model()

    # Impute just dividends on the original, full variable set on the copy

    zero_weight_copy = impute_over_incomes(
        zero_weight_copy,
        model,
        IMPUTATIONS,
    )

    dataset = impute_over_incomes(
        dataset,
        model,
        ["dividend_income"],
    )

    zero_weight_copy.validate()
    dataset.validate()

    data = stack_datasets(
        dataset,
        zero_weight_copy,
    )

    return data
