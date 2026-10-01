from policyengine_uk_data.storage import STORAGE_FOLDER
import pandas as pd
import numpy as np
from policyengine_uk.data import UKSingleYearDataset

SPI_RELEASE_NAME = "spi_2022_23"
SPI_TAB_FILENAME = "put2223uk.tab"
SPI_FISCAL_YEAR = 2022
SPI_H5_FILENAME = "spi_2022_23.h5"


# Age-range midpoints for random age imputation.
# Key -1 covers records with no reported AGERANGE — use a broad working-age
# span rather than silently bucketing them into one slot.
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

# SPI GORCODE → policyengine-uk region enum.
# NB the SPI codebook does not include a "region unknown" code; we surface
# unknown codes explicitly rather than silently mapping them to SOUTH_EAST
# (which the previous implementation did, distorting regional income totals).
REGION_MAP = {
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


def _get_allowances(fiscal_year: int) -> tuple[float, float, float]:
    """Return the personal allowance (ITA 2007 s. 35(1)), the income above
    which it tapers, and the Marriage Allowance transferable amount for the
    given UK fiscal year in £.

    The transferable amount is 10% of the personal allowance, rounded up to
    a multiple of £10 (s. 55B(4)-(5)), as in policyengine-uk's
    ``marriage_allowance_transferable_amount``.
    """
    from policyengine_uk.system import system

    instant = f"{fiscal_year}-04-06"
    allowances = system.parameters.gov.hmrc.income_tax.allowances
    pa = allowances.personal_allowance.amount(instant)
    taper_threshold = allowances.personal_allowance.maximum_ANI(instant)
    increment = allowances.marriage_allowance.rounding_increment(instant)
    transferable = pa * allowances.marriage_allowance.max(instant)
    transferable = np.ceil(transferable / increment) * increment
    return float(pa), float(taper_threshold), float(transferable)


def create_spi(
    spi_data_file_path: str,
    fiscal_year: int,
    output_file_path: str | None = None,
    seed: int = 0,
    unknown_region: str = "UNKNOWN",
) -> UKSingleYearDataset:
    """Build a :class:`UKSingleYearDataset` from an SPI microdata `.tab` file.

    Args:
        spi_data_file_path: Path to the SPI `.tab` file (e.g. `put2223uk.tab`).
        fiscal_year: UK fiscal year for the dataset (e.g. 2022 → 2022-23).
        output_file_path: Unused here — callers may save the returned dataset
            themselves with ``dataset.save(path)``. Kept as a kwarg so
            existing call sites don't break.
        seed: Seed for the random age imputation. Fixed by default so builds
            are deterministic.
        unknown_region: Fallback region label for SPI GORCODE values outside
            the documented 1-12 range. Defaults to ``"UNKNOWN"`` so regional
            totals are not silently distorted; pass ``"SOUTH_EAST"`` to
            reproduce legacy behaviour if needed.
    """
    df = pd.read_csv(spi_data_file_path, delimiter="\t")
    rng = np.random.default_rng(seed)

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
    household["region"] = df.GORCODE.map(REGION_MAP).fillna(unknown_region)
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
    # HMRC seems to assume the trading and property allowances are already
    # deducted (per record inspection of SREF 15494988 in 2020-21), so SPI
    # records override the actual deductions rather than the policy parameters.
    person["trading_allowance_deduction"] = np.zeros(len(df))
    person["property_allowance_deduction"] = np.zeros(len(df))
    person["savings_starter_rate_income"] = np.zeros(len(df))
    person["capital_allowances"] = df.CAPALL
    person["loss_relief"] = df.LOSSBF

    age_range = df.AGERANGE

    # Randomly assign ages within each AGERANGE bucket using a seeded local
    # generator so builds are reproducible (previously used the unseeded
    # global np.random.rand).
    percent_along_age_range = rng.random(len(df))
    bounds = np.array([AGE_RANGES.get(int(age), AGE_RANGES[-1]) for age in age_range])
    min_age = bounds[:, 0]
    max_age = bounds[:, 1]
    person["age"] = (min_age + (max_age - min_age) * percent_along_age_range).astype(
        int
    )

    person["state_pension_reported"] = df.SRP
    person["other_tax_credits"] = df.TAX_CRED
    person["miscellaneous_income"] = (
        df.MOTHINC + df.INCPBEN + df.OSSBEN + df.TAXTERM + df.UBISJA + df.OTHERINC
    )
    person["gift_aid"] = df.GIFTAID + df.GIFTINV
    person["other_investment_income"] = df.OTHERINV
    person["covenanted_payments"] = df.COVNTS
    person["other_deductions"] = df.MOTHDED + df.DEFICIEN
    person["married_couples_allowance"] = df.MCAS
    person["blind_persons_allowance"] = df.BPADUE
    # HMRC documents MAIND as "Marriage allowance claimant indicator" (1 =
    # "Claimant") and PAS as "Personal allowance (includes 10% marriage
    # allowance transfer if applicable)". The label does not say which spouse
    # claims, but in the 2022-23 tape every MAIND == 1 record has PAS equal to
    # the personal allowance plus the transferable amount, so it marks the
    # spouse who receives the transfer. policyengine-uk's `marriage_allowance`
    # is that received amount, and gives a tax reduction
    # (`marriage_allowance_tax_reduction`), not extra allowance.
    pa, taper_threshold, transferable = _get_allowances(fiscal_year)
    person["marriage_allowance"] = np.where(df.MAIND == 1, transferable, 0)
    # The tape does not flag the spouse who transfers, but their PAS is the
    # personal allowance less the transferable amount (s. 55B(6)). Below the
    # taper threshold nothing else gives that value. Each record is one
    # person, so policyengine-uk cannot find the electing spouse itself; give
    # the allowance up directly. PAS is not an input, so neither side of a
    # transfer is counted twice.
    transferor = (
        (df.MAIND == 0) & (df.PAS == pa - transferable) & (df.TI < taper_threshold)
    )
    person["marriage_allowance_relinquished"] = np.where(transferor, transferable, 0)

    dataset = UKSingleYearDataset(
        person=person,
        benunit=benunit,
        household=household,
        fiscal_year=fiscal_year,
    )
    return dataset


if __name__ == "__main__":
    spi_data_file_path = STORAGE_FOLDER / SPI_RELEASE_NAME / SPI_TAB_FILENAME
    fiscal_year = SPI_FISCAL_YEAR
    output_file_path = STORAGE_FOLDER / SPI_H5_FILENAME
    spi = create_spi(spi_data_file_path, fiscal_year, output_file_path)
    spi.save(output_file_path)
