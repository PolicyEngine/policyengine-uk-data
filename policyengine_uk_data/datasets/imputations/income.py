import pandas as pd
from pathlib import Path
import numpy as np
from policyengine_uk_data.storage import STORAGE_FOLDER
from policyengine_uk.data import UKDataset
from policyengine_uk import Microsimulation
from policyengine_uk_data.utils.stack import stack_datasets

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

    spi = spi[spi.TI > 100_000]

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
    "employment_expenses",
    "property_income",
    "gift_aid",
]


def save_imputation_models():
    from policyengine_uk_data.utils import QRF

    income = QRF()
    spi = pd.read_csv(SPI_TAB_FOLDER / "put2021uk.tab", delimiter="\t")
    spi = generate_spi_table(spi)
    spi = spi[PREDICTORS + IMPUTATIONS]
    income.fit(spi[PREDICTORS], spi[IMPUTATIONS])
    income.save(STORAGE_FOLDER / "income.pkl")
    return income


def create_income_model(overwrite_existing: bool = False):
    from policyengine_uk_data.utils.qrf import QRF

    if (STORAGE_FOLDER / "income.pkl").exists() and not overwrite_existing:
        return QRF(file_path=STORAGE_FOLDER / "income.pkl")
    return save_imputation_models()


def impute_income(dataset: UKDataset) -> UKDataset:
    # Impute wealth, assuming same time period as trained data
    dataset = dataset.copy()
    zero_weight_copy = dataset.copy()
    zero_weight_copy.household.household_weight = 0
    data = stack_datasets(
        dataset,
        zero_weight_copy,
    )

    model = create_income_model()
    sim = Microsimulation(dataset=data)

    input_df = sim.calculate_dataframe(["age", "gender", "region"])

    output_df = model.predict(input_df)

    for column in output_df.columns:
        data.person[column] = output_df[column].values

    dataset.validate()

    return data
