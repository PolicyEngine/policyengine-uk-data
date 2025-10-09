from policyengine_core.data import Dataset
from policyengine_uk_data.storage import STORAGE_FOLDER
import pandas as pd
import numpy as np
from policyengine_uk.data import UKSingleYearDataset


def create_spi(
    spi_data_file_path: str, fiscal_year: int, output_file_path: str
) -> UKSingleYearDataset:
    df = pd.read_csv(spi_data_file_path, delimiter="\t")

    person = pd.DataFrame()
    benunit = pd.DataFrame()
    household = pd.DataFrame()
    person["person_id"] = df.SREF
    person["person_household_id"] = df.SREF
    person["person_benunit_id"] = df.SREF
    benunit["benunit_id"] = df.SREF
    household["household_id"] = df.SREF

    household["household_weight"] = df.FACT
    person["dividend_income"] = df.DIVIDENDS
    person["gift_aid"] = df.GIFTAID
    household["region"] = (
        df.GORCODE.map(
            {
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
        )
        .fillna("SOUTH_EAST")
    )
    household["rent"] = 0
    household["tenure_type"] = "OWNED_OUTRIGHT"
    household["council_tax"] = 0
    person["savings_interest_income"] = df.INCBBS
    person["property_income"] = df.INCPROP
    person["employment_income"] = df.PAY + df.EPB
    person["employment_expenses"] = df.EXPS
    person["private_pension_income"] = df.PENSION
    # The below underestimates those with high amounts of excess pension
    # savings, as it does not include the Annual Allowance
    person["private_pension_contributions"] = df.PSAV_XS
    person["pension_contributions_relief"] = df.PENSRLF
    person["self_employment_income"] = df.PROFITS
    # HMRC seems to assume the trading and property allowance are already deducted
    # (per record inspection of SREF 15494988 in 2020-21)
    person["trading_allowance"] = np.zeros(len(df))
    person["property_allowance"] = np.zeros(len(df))
    person["savings_starter_rate_income"] = np.zeros(len(df))
    person["capital_allowances"] = df.CAPALL
    person["loss_relief"] = df.LOSSBF

    AGE_RANGES = {
        -1: (16, 70),
        1: (16, 25),
        2: (25, 35),
        3: (35, 45),
        4: (45, 55),
        5: (55, 65),
        6: (65, 74),
        7: (74, 90),
    }
    age_range = df.AGERANGE

    # Randomly assign ages in age ranges

    percent_along_age_range = np.random.rand(len(df))
    min_age = np.array([AGE_RANGES[age][0] for age in age_range])
    max_age = np.array([AGE_RANGES[age][1] for age in age_range])
    person["age"] = (
        min_age + (max_age - min_age) * percent_along_age_range
    ).astype(int)

    person["state_pension_reported"] = df.SRP
    person["other_tax_credits"] = df.TAX_CRED
    person["miscellaneous_income"] = (
        df.MOTHINC
        + df.INCPBEN
        + df.OSSBEN
        + df.TAXTERM
        + df.UBISJA
        + df.OTHERINC
    )
    person["gift_aid"] = df.GIFTAID + df.GIFTINV
    person["other_investment_income"] = df.OTHERINV
    person["covenanted_payments"] = df.COVNTS
    person["other_deductions"] = df.MOTHDED + df.DEFICIEN
    person["married_couples_allowance"] = df.MCAS
    person["blind_persons_allowance"] = df.BPADUE
    person["marriage_allowance"] = np.where(df.MAIND == 1, 1_250, 0)

    dataset = UKSingleYearDataset(
        person=person,
        benunit=benunit,
        household=household,
        fiscal_year=fiscal_year,
    )
    return dataset


if __name__ == "__main__":
    spi_data_file_path = STORAGE_FOLDER / "spi_2020_21" / "put2021uk.tab"
    fiscal_year = 2020
    output_file_path = STORAGE_FOLDER / "spi_2020.h5"
    spi = create_spi(spi_data_file_path, fiscal_year)
    spi.save(output_file_path)
