from policyengine_core.data import Dataset
from policyengine_uk_data.storage import STORAGE_FOLDER
import pandas as pd
import numpy as np
from policyengine_uk.data import UKSingleYearDataset


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


def _get_marriage_allowance(fiscal_year: int) -> float:
    """Return the maximum Marriage Allowance transfer for the given UK fiscal
    year in £. This equals ``max`` × ``personal_allowance`` at the start of
    the fiscal year (6 April), which is how HMRC publishes it. Falls back to
    the pre-2021-22 hard value of £1,250 if `policyengine_uk` cannot be
    imported (e.g., during unit tests that avoid the heavy import).
    """
    try:
        from policyengine_uk.system import system
    except Exception:
        return 1_250.0

    instant = f"{fiscal_year}-04-06"
    pa = system.parameters.gov.hmrc.income_tax.allowances.personal_allowance.amount(
        instant
    )
    ma_cap_rate = (
        system.parameters.gov.hmrc.income_tax.allowances.marriage_allowance.max(
            instant
        )
    )
    # HMRC rounds to the nearest £10 downward; use the explicit rounding param
    # if it exists, otherwise leave the computed value as-is.
    try:
        rounding_increment = (
            system.parameters.gov.hmrc.income_tax.allowances.marriage_allowance.rounding_increment(
                instant
            )
        )
    except Exception:
        rounding_increment = None

    value = pa * ma_cap_rate
    if rounding_increment:
        # HMRC rounds the cap UP to the nearest rounding increment
        # (Income Tax Act 2007 s. 55B(5)); matches the formula in
        # policyengine_uk.variables.gov.hmrc.income_tax.allowances.marriage_allowance.
        increment = float(rounding_increment)
        value = np.ceil(value / increment) * increment
    return float(value)


def create_spi(
    spi_data_file_path: str,
    fiscal_year: int,
    output_file_path: str | None = None,
    seed: int = 0,
    unknown_region: str = "UNKNOWN",
) -> UKSingleYearDataset:
    """Build a :class:`UKSingleYearDataset` from an SPI microdata `.tab` file.

    Args:
        spi_data_file_path: Path to the SPI `.tab` file (e.g. `put2021uk.tab`).
        fiscal_year: UK fiscal year for the dataset (e.g. 2020 → 2020-21).
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
    # HMRC seems to assume the trading and property allowance are already deducted
    # (per record inspection of SREF 15494988 in 2020-21)
    person["trading_allowance"] = np.zeros(len(df))
    person["property_allowance"] = np.zeros(len(df))
    person["savings_starter_rate_income"] = np.zeros(len(df))
    person["capital_allowances"] = df.CAPALL
    person["loss_relief"] = df.LOSSBF

    age_range = df.AGERANGE

    # Randomly assign ages within each AGERANGE bucket using a seeded local
    # generator so builds are reproducible (previously used the unseeded
    # global np.random.rand).
    percent_along_age_range = rng.random(len(df))
    min_age = np.array([AGE_RANGES[age][0] for age in age_range])
    max_age = np.array([AGE_RANGES[age][1] for age in age_range])
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
    # Pull the Marriage Allowance cap from policyengine-uk parameters keyed
    # on the fiscal year, rather than hardcoding 2020-21's £1,250 figure.
    ma_cap = _get_marriage_allowance(fiscal_year)
    person["marriage_allowance"] = np.where(df.MAIND == 1, ma_cap, 0)

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
    spi = create_spi(spi_data_file_path, fiscal_year, output_file_path)
    spi.save(output_file_path)
