from policyengine_core.data import Dataset
import pandas as pd
from pandas import DataFrame
from policyengine_uk_data.utils.datasets import (
    sum_to_entity,
    categorical,
    sum_from_positive_fields,
    sum_positive_variables,
    fill_with_mean,
    STORAGE_FOLDER,
)
from typing import Dict, List
import numpy as np
from numpy import maximum as max_, where
from typing import Type
import h5py
from policyengine_uk_data.datasets.frs.dwp_frs import *


class FRS(Dataset):
    name = "frs"
    label = "Family Resources Survey"
    data_format = Dataset.TIME_PERIOD_ARRAYS
    dwp_frs: Type[DWP_FRS] = None

    def generate(self):
        dwp_frs_files = self.dwp_frs()
        if not dwp_frs_files.file_path.exists():
            raise FileNotFoundError(
                f"Raw FRS file {dwp_frs_files.file_path} not found."
            )
        else:
            dwp_frs_files = dwp_frs_files.load()
        frs = {}
        TABLES = (
            "adult",
            "child",
            "accounts",
            "benefits",
            "job",
            "oddjob",
            "benunit",
            "househol",
            "chldcare",
            "pension",
            "maint",
            "mortgage",
            "penprov",
        )
        (
            adult,
            child,
            accounts,
            benefits,
            job,
            oddjob,
            benunit,
            household,
            childcare,
            pension,
            maintenance,
            mortgage,
            pen_prov,
        ) = [dwp_frs_files[table] for table in TABLES]
        dwp_frs_files.close()

        person = pd.concat([adult, child]).sort_index().fillna(0)
        add_id_variables(frs, person, household)
        add_personal_variables(frs, person, self.dwp_frs.time_period)
        add_benunit_variables(frs, benunit)
        add_household_variables(frs, household, self.dwp_frs.time_period)
        add_market_income(
            frs, person, pension, job, accounts, household, oddjob
        )
        add_benefit_income(frs, person, benefits, household)
        add_expenses(
            frs,
            person,
            job,
            household,
            maintenance,
            mortgage,
            childcare,
            pen_prov,
        )
        for variable in frs:
            frs[variable] = {self.dwp_frs.time_period: np.array(frs[variable])}

        self.save_dataset(frs)

        impute_brmas(self, frs)

        self.save_dataset(frs)

        self.add_random_variables(frs)

    def add_random_variables(self, frs: dict):
        from policyengine_uk import Microsimulation

        simulation = Microsimulation(dataset=self)
        RANDOM_VARIABLES = [
            "would_evade_tv_licence_fee",
            "would_claim_pc",
            "would_claim_uc",
            "would_claim_child_benefit",
            "main_residential_property_purchased_is_first_home",
            "household_owns_tv",
            "is_higher_earner",
            "attends_private_school",
        ]
        INPUT_PERIODS = list(range(self.time_period, self.time_period + 10))
        for variable in RANDOM_VARIABLES:
            value = simulation.calculate(variable, self.time_period).values
            frs[variable] = {period: value for period in INPUT_PERIODS}

        self.save_dataset(frs)


class FRS_2020_21(FRS):
    dwp_frs = DWP_FRS_2020_21
    name = "frs_2020_21"
    label = "FRS (2020-21)"
    file_path = STORAGE_FOLDER / "frs_2020_21.h5"
    time_period = 2020


class FRS_2021_22(FRS):
    dwp_frs = DWP_FRS_2021_22
    name = "frs_2021_22"
    label = "FRS (2021-22)"
    file_path = STORAGE_FOLDER / "frs_2021_22.h5"
    time_period = 2021


class FRS_2022_23(FRS):
    dwp_frs = DWP_FRS_2022_23
    name = "frs_2022_23"
    label = "FRS (2022-23)"
    file_path = STORAGE_FOLDER / "frs_2022_23.h5"
    time_period = 2022


def add_id_variables(frs: h5py.File, person: DataFrame, household: DataFrame):
    """Adds ID variables and weights.

    Args:
        frs (h5py.File)
        person (DataFrame)
        benunit (DataFrame)
        household (DataFrame)
    """
    # Add primary and foreign keys
    frs["person_id"] = person.index
    frs["person_benunit_id"] = person.benunit_id
    frs["person_household_id"] = person.household_id
    frs["benunit_id"] = person.benunit_id.sort_values().unique()
    frs["household_id"] = person.household_id.sort_values().unique()
    frs["state_id"] = np.array([1])
    frs["person_state_id"] = np.array([1] * len(person))
    frs["state_weight"] = np.array([1])

    # Add grossing weights
    frs["household_weight"] = household.GROSS4


def add_personal_variables(frs: h5py.File, person: DataFrame, year: int):
    """Adds personal variables (age, gender, education).

    Args:
        frs (h5py.File)
        person (DataFrame)
    """
    # Add basic personal variables
    age = person.AGE80 + person.AGE
    frs["age"] = age
    frs["birth_year"] = np.ones_like(person.AGE) * (year - age)
    # Age fields are AGE80 (top-coded) and AGE in the adult and child tables, respectively.
    frs["gender"] = np.where(person.SEX == 1, "MALE", "FEMALE").astype("S")
    frs["hours_worked"] = np.maximum(person.TOTHOURS, 0) * 52
    frs["is_household_head"] = person.HRPID == 1
    frs["is_benunit_head"] = person.UPERSON == 1
    MARITAL = [
        "MARRIED",
        "SINGLE",
        "SINGLE",
        "WIDOWED",
        "SEPARATED",
        "DIVORCED",
    ]
    frs["marital_status"] = categorical(
        person.MARITAL, 2, range(1, 7), MARITAL
    )

    # Add education levels
    if "FTED" in person.columns:
        fted = person.FTED
    else:
        fted = person.EDUCFT  # Renamed in FRS 2022-23
    typeed2 = person.TYPEED2
    frs["current_education"] = np.select(
        [
            fted.isin((2, -1, 0)),  # By default, not in education
            typeed2 == 1,  # In pre-primary
            typeed2.isin((2, 4))  # In primary, or...
            | (
                typeed2.isin((3, 8)) & (age < 11)
            )  # special or private education (and under 11), or...
            | (
                (typeed2 == 0) & (fted == 1) & (age > 5) & (age < 11)
            ),  # not given, full-time and between 5 and 11
            typeed2.isin((5, 6))  # In secondary, or...
            | (
                typeed2.isin((3, 8)) & (age >= 11) & (age <= 16)
            )  # special/private and meets age criteria, or...
            | (
                (typeed2 == 0) & (fted == 1) & (age <= 16)
            ),  # not given, full-time and under 17
            typeed2  # Non-advanced further education, or...
            == 7
            | (
                typeed2.isin((3, 8)) & (age > 16)
            )  # special/private and meets age criteria, or...
            | (
                (typeed2 == 0) & (fted == 1) & (age > 16)
            ),  # not given, full-time and over 16
            typeed2.isin((7, 8)) & (age >= 19),  # In post-secondary
            typeed2
            == 9
            | (
                (typeed2 == 0) & (fted == 1) & (age >= 19)
            ),  # In tertiary, or meets age condition
        ],
        [
            "NOT_IN_EDUCATION",
            "PRE_PRIMARY",
            "PRIMARY",
            "LOWER_SECONDARY",
            "UPPER_SECONDARY",
            "POST_SECONDARY",
            "TERTIARY",
        ],
    ).astype("S")

    # Add employment status
    EMPLOYMENTS = [
        "CHILD",
        "FT_EMPLOYED",
        "PT_EMPLOYED",
        "FT_SELF_EMPLOYED",
        "PT_SELF_EMPLOYED",
        "UNEMPLOYED",
        "RETIRED",
        "STUDENT",
        "CARER",
        "LONG_TERM_DISABLED",
        "SHORT_TERM_DISABLED",
    ]
    frs["employment_status"] = categorical(
        person.EMPSTATI, 1, range(12), EMPLOYMENTS
    )


def add_household_variables(frs: h5py.File, household: DataFrame, year: int):
    """Adds household variables (region, tenure, council tax imputation).

    Args:
        frs (h5py.File)
        household (DataFrame)
    """
    REGIONS = [
        "NORTH_EAST",
        "NORTH_WEST",
        "YORKSHIRE",
        "EAST_MIDLANDS",
        "WEST_MIDLANDS",
        "EAST_OF_ENGLAND",
        "LONDON",
        "SOUTH_EAST",
        "SOUTH_WEST",
        "WALES",
        "SCOTLAND",
        "NORTHERN_IRELAND",
        "UNKNOWN",
    ]
    frs["region"] = categorical(
        household.GVTREGNO, 14, [1, 2] + list(range(4, 15)), REGIONS
    )
    TENURES = [
        "RENT_FROM_COUNCIL",
        "RENT_FROM_HA",
        "RENT_PRIVATELY",
        "RENT_PRIVATELY",
        "OWNED_OUTRIGHT",
        "OWNED_WITH_MORTGAGE",
    ]
    frs["tenure_type"] = categorical(
        household.PTENTYP2, 3, range(1, 7), TENURES
    )
    frs["num_bedrooms"] = household.BEDROOM6
    ACCOMMODATIONS = [
        "HOUSE_DETACHED",
        "HOUSE_SEMI_DETACHED",
        "HOUSE_TERRACED",
        "FLAT",
        "CONVERTED_HOUSE",
        "MOBILE",
        "OTHER",
    ]
    frs["accommodation_type"] = categorical(
        household.TYPEACC, 1, range(1, 8), ACCOMMODATIONS
    )

    # Impute Council Tax

    # Only ~25% of household report Council Tax bills - use
    # these to build a model to impute missing values
    CT_valid = household.CTANNUAL > 0

    # Find the mean reported Council Tax bill for a given
    # (region, CT band, is-single-person-household) triplet
    region = household.GVTREGNO[CT_valid]
    band = household.CTBAND[CT_valid]
    single_person = (household.ADULTH == 1)[CT_valid]
    ctannual = household.CTANNUAL[CT_valid]

    # Build the table
    CT_mean = ctannual.groupby(
        [region, band, single_person], dropna=False
    ).mean()
    CT_mean = CT_mean.replace(-1, CT_mean.mean())

    # For every household consult the table to find the imputed
    # Council Tax bill
    pairs = household.set_index(
        [household.GVTREGNO, household.CTBAND, (household.ADULTH == 1)]
    )
    hh_CT_mean = pd.Series(index=pairs.index)
    has_mean = pairs.index.isin(CT_mean.index)
    hh_CT_mean[has_mean] = CT_mean[pairs.index[has_mean]].values
    hh_CT_mean[~has_mean] = 0
    CT_imputed = hh_CT_mean

    # For households which originally reported Council Tax,
    # use the reported value. Otherwise, use the imputed value
    council_tax = pd.Series(
        np.where(
            # 2018 FRS uses blanks for missing values, 2019 FRS
            # uses -1 for missing values
            (household.CTANNUAL < 0) | household.CTANNUAL.isna(),
            max_(CT_imputed, 0).values,
            household.CTANNUAL,
        )
    )
    frs["council_tax"] = council_tax.fillna(0)
    BANDS = ["A", "B", "C", "D", "E", "F", "G", "H", "I"]
    # Band 1 is the most common
    frs["council_tax_band"] = categorical(
        household.CTBAND, 1, range(1, 10), BANDS
    )
    # Domestic rates variables are all weeklyised, unlike Council Tax variables (despite the variable name suggesting otherwise)
    if year < 2021:
        DOMESTIC_RATES_VARIABLE = "RTANNUAL"
    else:
        DOMESTIC_RATES_VARIABLE = "NIRATLIA"
    frs["domestic_rates"] = (
        np.select(
            [
                household[DOMESTIC_RATES_VARIABLE] >= 0,
                household.RT2REBAM >= 0,
                True,
            ],
            [
                household[DOMESTIC_RATES_VARIABLE],
                household.RT2REBAM,
                0,
            ],
        )
        * 52
    )


def add_market_income(
    frs: h5py.File,
    person: DataFrame,
    pension: DataFrame,
    job: DataFrame,
    account: DataFrame,
    household: DataFrame,
    oddjob: DataFrame,
):
    """Adds income variables (non-benefit).

    Args:
        frs (h5py.File)
        person (DataFrame)
        pension (DataFrame)
        job (DataFrame)
        account (DataFrame)
        household (DataFrame)
        oddjob (DataFrame)
    """
    frs["employment_income"] = person.INEARNS * 52

    pension_payment = sum_to_entity(
        pension.PENPAY * (pension.PENPAY > 0), pension.person_id, person.index
    )
    pension_tax_paid = sum_to_entity(
        (pension.PTAMT * ((pension.PTINC == 2) & (pension.PTAMT > 0))),
        pension.person_id,
        person.index,
    )
    pension_deductions_removed = sum_to_entity(
        pension.POAMT
        * (
            ((pension.POINC == 2) | (pension.PENOTH == 1))
            & (pension.POAMT > 0)
        ),
        pension.person_id,
        person.index,
    )

    frs["private_pension_income"] = (
        pension_payment + pension_tax_paid + pension_deductions_removed
    ) * 52

    frs["self_employment_income"] = person.SEINCAM2 * 52

    INVERTED_BASIC_RATE = 1.25

    frs["tax_free_savings_income"] = (
        sum_to_entity(
            account.ACCINT * (account.ACCOUNT == 21),
            account.person_id,
            person.index,
        )
        * 52
    )
    taxable_savings_interest = (
        sum_to_entity(
            (
                account.ACCINT
                * np.where(account.ACCTAX == 1, INVERTED_BASIC_RATE, 1)
            )
            * (account.ACCOUNT.isin((1, 3, 5, 27, 28))),
            account.person_id,
            person.index,
        )
        * 52
    )
    frs["savings_interest_income"] = (
        taxable_savings_interest + frs["tax_free_savings_income"]
    )
    frs["dividend_income"] = (
        sum_to_entity(
            (
                account.ACCINT
                * np.where(account.INVTAX == 1, INVERTED_BASIC_RATE, 1)
            )
            * (
                ((account.ACCOUNT == 6) & (account.INVTAX == 1))  # GGES
                | account.ACCOUNT.isin((7, 8))  # Stocks/shares/UITs
            ),
            account.person_id,
            person.index,
        )
        * 52
    )
    is_head = person.HRPID == 1
    household_property_income = (
        household.TENTYP2.isin((5, 6)) * household.SUBRENT
    )  # Owned and subletting
    persons_household_property_income = pd.Series(
        household_property_income[person.household_id].values,
        index=person.index,
    ).fillna(0)
    frs["property_income"] = (
        max_(
            0,
            is_head * persons_household_property_income
            + person.CVPAY
            + person.ROYYR1,
        )
        * 52
    )
    maintenance_to_self = max_(
        pd.Series(
            where(person.MNTUS1 == 2, person.MNTUSAM1, person.MNTAMT1)
        ).fillna(0),
        0,
    )
    maintenance_from_DWP = person.MNTAMT2
    frs["maintenance_income"] = (
        sum_positive_variables([maintenance_to_self, maintenance_from_DWP])
        * 52
    )

    odd_job_income = sum_to_entity(
        oddjob.OJAMT * (oddjob.OJNOW == 1), oddjob.person_id, person.index
    )

    MISC_INCOME_FIELDS = [
        "ALLPAY2",
        "ROYYR2",
        "ROYYR3",
        "ROYYR4",
        "CHAMTERN",
        "CHAMTTST",
    ]

    frs["miscellaneous_income"] = (
        odd_job_income + sum_from_positive_fields(person, MISC_INCOME_FIELDS)
    ) * 52

    PRIVATE_TRANSFER_INCOME_FIELDS = [
        "APAMT",
        "APDAMT",
        "PAREAMT",
        "ALLPAY1",
        "ALLPAY3",
        "ALLPAY4",
    ]

    frs["private_transfer_income"] = (
        sum_from_positive_fields(person, PRIVATE_TRANSFER_INCOME_FIELDS) * 52
    )

    frs["lump_sum_income"] = person.REDAMT


def sum_from_positive_fields(
    table: pd.DataFrame, fields: List[str]
) -> np.array:
    """Sum from fields in table, ignoring negative values.

    Args:
        table (DataFrame)
        fields (List[str])

    Returns:
        np.array
    """
    return np.where(
        table[fields].sum(axis=1) > 0, table[fields].sum(axis=1), 0
    )


def sum_positive_variables(variables: List[str]) -> np.array:
    """Sum positive variables.

    Args:
        variables (List[str])

    Returns:
        np.array
    """
    return sum([np.where(variable > 0, variable, 0) for variable in variables])


def fill_with_mean(
    table: pd.DataFrame, code: str, amount: str, multiplier: float = 52
) -> np.array:
    """Fills missing values in a table with the mean of the column.

    Args:
        table (DataFrame): Table to fill.
        code (str): Column signifying existence.
        amount (str): Column with values.
        multiplier (float): Multiplier to apply to amount.

    Returns:
        np.array: Filled values.
    """
    needs_fill = (table[code] == 1) & (table[amount] < 0)
    has_value = (table[code] == 1) & (table[amount] >= 0)
    fill_mean = table[amount][has_value].mean()
    filled_values = np.where(needs_fill, fill_mean, table[amount])
    return np.maximum(filled_values, 0) * multiplier


def add_benefit_income(
    frs: h5py.File,
    person: DataFrame,
    benefits: DataFrame,
    household: DataFrame,
):
    """Adds benefit variables.

    Args:
        frs (h5py.File)
        person (DataFrame)
        benefits (DataFrame)
        household (DataFrame)
    """
    BENEFIT_CODES = dict(
        child_benefit=3,
        income_support=19,
        housing_benefit=94,
        attendance_allowance=12,
        dla_sc=1,
        dla_m=2,
        iidb=15,
        carers_allowance=13,
        sda=10,
        afcs=8,
        ssmg=22,
        pension_credit=4,
        child_tax_credit=91,
        working_tax_credit=90,
        state_pension=5,
        winter_fuel_allowance=62,
        incapacity_benefit=17,
        universal_credit=95,
        pip_m=97,
        pip_dl=96,
    )

    for benefit, code in BENEFIT_CODES.items():
        frs[benefit + "_reported"] = (
            sum_to_entity(
                benefits.BENAMT * (benefits.BENEFIT == code),
                benefits.person_id,
                person.index,
            )
            * 52
        )

    frs["jsa_contrib_reported"] = (
        sum_to_entity(
            benefits.BENAMT
            * (benefits.VAR2.isin((1, 3)))
            * (benefits.BENEFIT == 14),
            benefits.person_id,
            person.index,
        )
        * 52
    )
    frs["jsa_income_reported"] = (
        sum_to_entity(
            benefits.BENAMT
            * (benefits.VAR2.isin((2, 4)))
            * (benefits.BENEFIT == 14),
            benefits.person_id,
            person.index,
        )
        * 52
    )
    frs["esa_contrib_reported"] = (
        sum_to_entity(
            benefits.BENAMT
            * (benefits.VAR2.isin((1, 3)))
            * (benefits.BENEFIT == 16),
            benefits.person_id,
            person.index,
        )
        * 52
    )
    frs["esa_income_reported"] = (
        sum_to_entity(
            benefits.BENAMT
            * (benefits.VAR2.isin((2, 4)))
            * (benefits.BENEFIT == 16),
            benefits.person_id,
            person.index,
        )
        * 52
    )

    frs["bsp_reported"] = (
        sum_to_entity(
            benefits.BENAMT * (benefits.BENEFIT.isin((6, 9))),
            benefits.person_id,
            person.index,
        )
        * 52
    )

    frs["winter_fuel_allowance_reported"] = (
        np.array(frs["winter_fuel_allowance_reported"]) / 52
    )  # This is not weeklyised by default (paid once per year)

    frs["statutory_sick_pay"] = person.SSPADJ * 52
    frs["statutory_maternity_pay"] = person.SMPADJ * 52

    frs["student_loans"] = np.maximum(person.TUBORR, 0)
    if "ADEMA" not in person.columns:
        person["ADEMA"] = person.EDUMA
        person["ADEMAAMT"] = person.EDUMAAMT
    frs["adult_ema"] = fill_with_mean(person, "ADEMA", "ADEMAAMT")
    frs["child_ema"] = fill_with_mean(person, "CHEMA", "CHEMAAMT")

    frs["access_fund"] = np.maximum(person.ACCSSAMT, 0) * 52

    frs["education_grants"] = np.maximum(
        person[["GRTDIR1", "GRTDIR2"]].sum(axis=1), 0
    )

    frs["council_tax_benefit_reported"] = np.maximum(
        (person.HRPID == 1)
        * pd.Series(
            household.CTREBAMT[person.household_id].values, index=person.index
        ).fillna(0)
        * 52,
        0,
    )


def add_expenses(
    frs: h5py.File,
    person: DataFrame,
    job: DataFrame,
    household: DataFrame,
    maintenance: DataFrame,
    mortgage: DataFrame,
    childcare: DataFrame,
    pen_prov: DataFrame,
):
    """Adds expense variables

    Args:
        frs (h5py.File)
        person (DataFrame)
        household (DataFrame)
        maintenance (DataFrame)
        mortgage (DataFrame)
        childcare (DataFrame)
        pen_prov (DataFrame)
    """
    frs["maintenance_expenses"] = (
        pd.Series(
            np.where(
                maintenance.MRUS == 2, maintenance.MRUAMT, maintenance.MRAMT
            )
        )
        .groupby(maintenance.person_id)
        .sum()
        .reindex(person.index)
        .fillna(0)
        * 52
    )

    frs["housing_costs"] = (
        np.where(
            household.GVTREGNO != 13, household.GBHSCOST, household.NIHSCOST
        )
        * 52
    )
    frs["rent"] = household.HHRENT.fillna(0) * 52
    frs["mortgage_interest_repayment"] = household.MORTINT.fillna(0) * 52
    mortgage_capital = np.where(
        mortgage.RMORT == 1, mortgage.RMAMT, mortgage.BORRAMT
    )
    mortgage_capital_repayment = sum_to_entity(
        mortgage_capital / mortgage.MORTEND,
        mortgage.household_id,
        household.index,
    )
    frs["mortgage_capital_repayment"] = mortgage_capital_repayment

    frs["childcare_expenses"] = (
        sum_to_entity(
            childcare.CHAMT
            * (childcare.COST == 1)
            * (childcare.REGISTRD == 1),
            childcare.person_id,
            person.index,
        )
        * 52
    )

    frs["personal_pension_contributions"] = max_(
        0,
        sum_to_entity(
            pen_prov.PENAMT[pen_prov.STEMPPEN.isin((5, 6))],
            pen_prov.person_id,
            person.index,
        ).clip(0, pen_prov.PENAMT.quantile(0.95))
        * 52,
    )
    frs["employee_pension_contributions"] = max_(
        0,
        sum_to_entity(job.DEDUC1.fillna(0), job.person_id, person.index) * 52,
    )
    frs["employer_pension_contributions"] = (
        frs["employee_pension_contributions"] * 3
    )  # Rough estimate based on aggregates.

    frs["housing_service_charges"] = (
        pd.DataFrame(
            [
                household[f"CHRGAMT{i}"] * (household[f"CHRGAMT{i}"] > 0)
                for i in range(1, 10)
            ]
        ).sum()
        * 52
    )
    frs["water_and_sewerage_charges"] = (
        pd.Series(
            np.where(
                household.GVTREGNO == 12,
                household.CSEWAMT + household.CWATAMTD,
                household.WATSEWRT,
            )
        ).fillna(0)
        * 52
    )


def add_benunit_variables(frs: h5py.File, benunit: DataFrame):
    frs["benunit_rent"] = np.maximum(benunit.BURENT.fillna(0) * 52, 0)


def impute_brmas(dataset, frs):
    # Randomly select broad rental market areas from regions.
    from policyengine_uk import Microsimulation

    sim = Microsimulation(dataset=dataset)
    region = (
        sim.populations["benunit"]
        .household("region", dataset.time_period)
        .decode_to_str()
    )
    lha_category = sim.calculate("LHA_category")

    brma = np.empty(len(region), dtype=object)

    # Sample from a random BRMA in the region, weighted by the number of observations in each BRMA
    lha_list_of_rents = pd.read_csv(
        STORAGE_FOLDER / "lha_list_of_rents.csv.gz"
    )
    lha_list_of_rents = lha_list_of_rents.copy()

    for possible_region in lha_list_of_rents.region.unique():
        for possible_lha_category in lha_list_of_rents.lha_category.unique():
            lor_mask = (lha_list_of_rents.region == possible_region) & (
                lha_list_of_rents.lha_category == possible_lha_category
            )
            mask = (region == possible_region) & (
                lha_category == possible_lha_category
            )
            brma[mask] = lha_list_of_rents[lor_mask].brma.sample(
                n=len(region[mask]), replace=True
            )

    # Convert benunit-level BRMAs to household-level BRMAs (pick a random one)

    df = pd.DataFrame(
        {
            "brma": brma,
            "household_id": sim.populations["benunit"].household(
                "household_id", 2023
            ),
        }
    )

    df = df.groupby("household_id").brma.aggregate(
        lambda x: x.sample(n=1).iloc[0]
    )
    brmas = df[sim.calculate("household_id")].values

    frs["brma"] = {dataset.time_period: brmas}


if __name__ == "__main__":
    FRS_2020_21().generate()
    FRS_2022_23().generate()
