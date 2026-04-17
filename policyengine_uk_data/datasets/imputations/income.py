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
            spi.sample(100_000, weights=spi.person_weight, replace=True),
        ]
    )

    return spi


PREDICTORS = [
    "age",
    "gender",
    "region",
]

INCOME_COMPONENTS = [
    "employment_income",
    "self_employment_income",
    "savings_interest_income",
    "dividend_income",
    "private_pension_income",
    "property_income",
]

# Gift Aid is in SPI but isn't in FRS — without it in the model outputs,
# the zero-weight SPI-donor rows carry a middle-income FRS donor's (always
# zero) Gift Aid, missing the £1-1.5bn/yr Gift Aid higher-rate relief flow.
# Including it here means the multi-output QRF draws gift_aid jointly with
# income components, so high-earner donors get plausibly non-zero Gift Aid.
# We keep it separate from INCOME_COMPONENTS because the rent/mortgage
# adjustment factor downstream is built from income sums, and Gift Aid is
# an expenditure, not income.
IMPUTATIONS = INCOME_COMPONENTS + ["gift_aid"]


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
    income.save(STORAGE_FOLDER / "income_v2.pkl")
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

    if (STORAGE_FOLDER / "income_v2.pkl").exists() and not overwrite_existing:
        return QRF(file_path=STORAGE_FOLDER / "income_v2.pkl")
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
    sim = Microsimulation(dataset=dataset)
    input_df = sim.calculate_dataframe(["age", "gender", "region"])
    original_income_total = dataset.person[INCOME_COMPONENTS].copy().sum().sum()
    output_df = model.predict(input_df)

    for column in output_variables:
        dataset.person[column] = output_df[column].fillna(0).values

    new_income_total = dataset.person[INCOME_COMPONENTS].sum().sum()
    adjustment_factor = new_income_total / original_income_total
    # Adjust rent and mortgage interest and capital repayments proportionally
    dataset.household["rent"] = dataset.household["rent"] * adjustment_factor
    dataset.household["mortgage_interest_repayment"] = (
        dataset.household["mortgage_interest_repayment"] * adjustment_factor
    )
    dataset.household["mortgage_capital_repayment"] = (
        dataset.household["mortgage_capital_repayment"] * adjustment_factor
    )

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
    # gift_aid is in IMPUTATIONS but is not a column on the raw FRS build, so
    # initialise it to zero everywhere before imputation. Without this, the
    # full-FRS half stays NaN for gift_aid (it's never touched by the dividend-
    # only impute_over_incomes call below), and the eventual stacked dataset
    # fails validate() on the gift_aid column.
    if "gift_aid" not in dataset.person.columns:
        dataset.person["gift_aid"] = 0.0
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
