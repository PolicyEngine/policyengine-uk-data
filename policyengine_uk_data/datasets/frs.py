"""
Family Resources Survey (FRS) dataset processing for PolicyEngine UK.

This module processes raw FRS survey data into PolicyEngine UK dataset format,
handling household demographics, income, benefits, and other survey variables.
The FRS is the primary source of UK household survey data used for tax-benefit
modelling and policy analysis.
"""

from policyengine_uk.data import UKSingleYearDataset
from pathlib import Path
import pandas as pd
import numpy as np
from policyengine_uk_data.utils.datasets import (
    sum_to_entity,
    categorical,
    sum_from_positive_fields,
    sum_positive_variables,
    fill_with_mean,
    STORAGE_FOLDER,
)


def create_frs(
    raw_frs_folder: str,
    year: int,
) -> UKSingleYearDataset:
    """
    Process raw FRS data into PolicyEngine UK dataset format.

    Transforms the Family Resources Survey microdata from raw tab-delimited
    files into a structured PolicyEngine UK dataset with person, benefit unit,
    and household-level variables mapped to the appropriate tax-benefit system
    variables.

    Args:
        raw_frs_folder: Path to folder containing raw FRS .tab files.
        year: Survey year for the dataset.

    Returns:
        UKSingleYearDataset with processed FRS data ready for policy simulation.
    """
    raw_folder = Path(raw_frs_folder)
    if not raw_folder.exists():
        raise FileNotFoundError(f"Raw folder {raw_folder} does not exist.")

    frs = {}
    # Store SALSAC values before numeric conversion (for salary sacrifice
    # imputation)
    job_salsac_raw = None
    for file in raw_folder.glob("*.tab"):
        table_name = file.stem
        # Read raw data first
        df_raw = pd.read_csv(file, sep="\t")
        df_raw.columns = df_raw.columns.str.lower()

        # Preserve SALSAC column from job table before numeric conversion
        # SALSAC indicates salary sacrifice participation:
        # '1' = Yes, '2' = No, ' ' or blank = skip/not asked
        if table_name == "job" and "salsac" in df_raw.columns:
            job_salsac_raw = df_raw["salsac"].copy()

        # Make numeric where possible
        df = df_raw.apply(pd.to_numeric, errors="coerce")

        # Standardise column names to lower case (already done above)
        # df.columns = df.columns.str.lower()

        # Edit ID variables for simplicity
        if "sernum" in df.columns:
            df.rename(columns={"sernum": "household_id"}, inplace=True)

        if "benunit" in df.columns:
            # In the tables, benunit is the index of the benefit unit *within* the household.
            df.rename(columns={"benunit": "benunit_id"}, inplace=True)
            df["benunit_id"] = (
                df["household_id"] * 1e2 + df["benunit_id"]
            ).astype(int)

        if "person" in df.columns:
            df.rename(columns={"person": "person_id"}, inplace=True)
            df["person_id"] = (
                df["household_id"] * 1e3 + df["person_id"]
            ).astype(int)

        frs[table_name] = df

    # Combine adult and child tables for convenience

    frs["person"] = (
        pd.concat([frs["adult"], frs["child"]]).sort_index().fillna(0)
    )

    person = frs["person"]
    benunit = frs["benunit"]
    household = frs["househol"]
    household = household.set_index("household_id")
    pension = frs["pension"]
    oddjob = frs["oddjob"]
    account = frs["accounts"]
    job = frs["job"]
    # Add raw SALSAC column to job table for salary sacrifice imputation
    # SALSAC values: '1' = Yes (participates), '2' = No, ' '/blank = not asked
    if job_salsac_raw is not None:
        job["salsac_raw"] = job_salsac_raw.values
    benefits = frs["benefits"]
    maintenance = frs["maint"]
    pen_prov = frs["penprov"]
    childcare = frs["chldcare"]
    extchild = frs["extchild"]
    mortgage = frs["mortgage"]

    pe_person = pd.DataFrame()
    pe_benunit = pd.DataFrame()
    pe_household = pd.DataFrame()

    # Add primary and foreign keys
    pe_person["person_id"] = person.person_id
    pe_person["person_benunit_id"] = person.benunit_id
    pe_person["person_household_id"] = person.household_id
    pe_benunit["benunit_id"] = benunit.benunit_id
    pe_household["household_id"] = person.household_id.sort_values().unique()

    # Add grossing weights
    pe_household["household_weight"] = household.gross4.values

    # Add basic personal variables
    age = person.age80 + person.age
    pe_person["age"] = age
    # birth_year should be calculated from age and period in the model,
    # not stored as static data (see PolicyEngine/policyengine-uk#1352)
    # Age fields are AGE80 (top-coded) and AGE in the adult and child tables, respectively.
    pe_person["gender"] = np.where(person.sex == 1, "MALE", "FEMALE")
    pe_person["hours_worked"] = np.maximum(person.tothours, 0) * 52
    pe_person["is_household_head"] = person.hrpid == 1
    pe_person["is_benunit_head"] = person.uperson == 1
    MARITAL = [
        "MARRIED",
        "SINGLE",
        "SINGLE",
        "WIDOWED",
        "SEPARATED",
        "DIVORCED",
    ]
    pe_person["marital_status"] = categorical(
        person.marital, 2, range(1, 7), MARITAL
    ).fillna("SINGLE")

    # Add education levels
    if "fted" in person.columns:
        fted = person.fted
    else:
        fted = person.educft  # Renamed in FRS 2022-23
    typeed2 = person.typeed2

    def determine_education_level(fted_val, typeed2_val, age_val):
        # By default, not in education
        if fted_val in (2, -1, 0):
            return "NOT_IN_EDUCATION"
        # In pre-primary
        elif typeed2_val == 1:
            return "PRE_PRIMARY"
        # In primary education
        elif (
            typeed2_val in (2, 4)
            or (typeed2_val in (3, 8) and age_val < 11)
            or (
                typeed2_val == 0
                and fted_val == 1
                and age_val > 5
                and age_val < 11
            )
        ):
            return "PRIMARY"
        # In lower secondary
        elif (
            typeed2_val in (5, 6)
            or (typeed2_val in (3, 8) and age_val >= 11 and age_val <= 16)
            or (typeed2_val == 0 and fted_val == 1 and age_val <= 16)
        ):
            return "LOWER_SECONDARY"
        # In upper secondary
        elif (
            typeed2_val == 7
            or (typeed2_val in (3, 8) and age_val > 16)
            or (typeed2_val == 0 and fted_val == 1 and age_val > 16)
        ):
            return "UPPER_SECONDARY"
        # In post-secondary
        elif typeed2_val in (7, 8) and age_val >= 19:
            return "POST_SECONDARY"
        # In tertiary
        elif typeed2_val == 9 or (
            typeed2_val == 0 and fted_val == 1 and age_val >= 19
        ):
            return "TERTIARY"
        else:
            return "NOT_IN_EDUCATION"

    # Apply the function to determine education level
    pe_person["current_education"] = pd.Series(
        [
            determine_education_level(f, t, a)
            for f, t, a in zip(fted, typeed2, age)
        ],
        index=pe_person.index,
    )

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
    pe_person["employment_status"] = categorical(
        person.empstati, 1, range(12), EMPLOYMENTS
    ).fillna("LONG_TERM_DISABLED")

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
    pe_household["region"] = categorical(
        household.gvtregno, 14, [1, 2] + list(range(4, 15)), REGIONS
    ).values
    TENURES = [
        "RENT_FROM_COUNCIL",
        "RENT_FROM_HA",
        "RENT_PRIVATELY",
        "RENT_PRIVATELY",
        "OWNED_OUTRIGHT",
        "OWNED_WITH_MORTGAGE",
    ]
    pe_household["tenure_type"] = categorical(
        household.ptentyp2, 3, range(1, 7), TENURES
    ).values
    frs["num_bedrooms"] = household.bedroom6
    ACCOMMODATIONS = [
        "HOUSE_DETACHED",
        "HOUSE_SEMI_DETACHED",
        "HOUSE_TERRACED",
        "FLAT",
        "CONVERTED_HOUSE",
        "MOBILE",
        "OTHER",
    ]
    pe_household["accommodation_type"] = categorical(
        household.typeacc, 1, range(1, 8), ACCOMMODATIONS
    ).values

    # Impute Council Tax

    # Only ~25% of household report Council Tax bills - use
    # these to build a model to impute missing values
    CT_valid = household.ctannual > 0

    # Find the mean reported Council Tax bill for a given
    # (region, CT band, is-single-person-household) triplet
    region = household.gvtregno[CT_valid]
    band = household.ctband[CT_valid]
    single_person = (household.adulth == 1)[CT_valid]
    ctannual = household.ctannual[CT_valid]

    # Build the table
    ct_mean = ctannual.groupby(
        [region, band, single_person], dropna=False
    ).mean()
    ct_mean = ct_mean.replace(-1, ct_mean.mean())

    # For every household consult the table to find the imputed
    # Council Tax bill
    pairs = household.set_index(
        [household.gvtregno, household.ctband, (household.adulth == 1)]
    )
    hh_CT_mean = pd.Series(index=pairs.index)
    has_mean = pairs.index.isin(ct_mean.index)
    hh_CT_mean[has_mean] = ct_mean[pairs.index[has_mean]].values
    hh_CT_mean[~has_mean] = 0
    ct_imputed = hh_CT_mean

    # For households which originally reported Council Tax,
    # use the reported value. Otherwise, use the imputed value
    council_tax = pd.Series(
        np.where(
            # 2018 FRS uses blanks for missing values, 2019 FRS
            # uses -1 for missing values
            (household.ctannual < 0) | household.ctannual.isna(),
            np.maximum(ct_imputed, 0).values,
            household.ctannual,
        )
    )
    pe_household["council_tax"] = council_tax.fillna(0)
    BANDS = ["A", "B", "C", "D", "E", "F", "G", "H", "I"]
    # Band 1 is the most common
    pe_household["council_tax_band"] = (
        categorical(household.ctband, 1, range(1, 10), BANDS)
        .fillna("D")
        .values
    )
    # Domestic rates variables are all weeklyised, unlike Council Tax variables (despite the variable name suggesting otherwise)
    if year < 2021:
        DOMESTIC_RATES_VARIABLE = "rtannual"
    else:
        DOMESTIC_RATES_VARIABLE = "niratlia"
    pe_household["domestic_rates"] = (
        np.select(
            [
                household[DOMESTIC_RATES_VARIABLE] >= 0,
                household.rt2rebam >= 0,
                True,
            ],
            [
                household[DOMESTIC_RATES_VARIABLE],
                household.rt2rebam,
                0,
            ],
        )
        * 52
    ).astype(float)

    WEEKS_IN_YEAR = 365.25 / 7

    pe_person["employment_income"] = person.inearns * WEEKS_IN_YEAR

    pension_payment = sum_to_entity(
        pension.penpay * (pension.penpay > 0),
        pension.person_id,
        person.person_id,
    )
    pension_tax_paid = sum_to_entity(
        (pension.ptamt * ((pension.ptinc == 2) & (pension.ptamt > 0))),
        pension.person_id,
        person.person_id,
    )
    pension_deductions_removed = sum_to_entity(
        pension.poamt
        * (
            ((pension.poinc == 2) | (pension.penoth == 1))
            & (pension.poamt > 0)
        ),
        pension.person_id,
        person.person_id,
    )

    pe_person["private_pension_income"] = (
        pension_payment + pension_tax_paid + pension_deductions_removed
    ) * WEEKS_IN_YEAR

    pe_person["self_employment_income"] = person.seincam2 * WEEKS_IN_YEAR

    INVERTED_BASIC_RATE = 1.25

    pe_person["tax_free_savings_income"] = (
        sum_to_entity(
            account.accint * (account.account == 21),
            account.person_id,
            person.person_id,
        )
        * WEEKS_IN_YEAR
    )
    taxable_savings_interest = (
        sum_to_entity(
            (
                account.accint
                * np.where(account.acctax == 1, INVERTED_BASIC_RATE, 1)
            )
            * (account.account.isin((1, 3, 5, 27, 28))),
            account.person_id,
            person.person_id,
        )
        * WEEKS_IN_YEAR
    )
    pe_person["savings_interest_income"] = (
        taxable_savings_interest + pe_person["tax_free_savings_income"].values
    )
    pe_person["dividend_income"] = (
        sum_to_entity(
            (
                account.accint
                * np.where(account.invtax == 1, INVERTED_BASIC_RATE, 1)
            )
            * (
                ((account.account == 6) & (account.invtax == 1))  # GGES
                | account.account.isin((7, 8))  # Stocks/shares/UITs
            ),
            account.person_id,
            person.index,
        )
        * 52
    )
    is_head = person.hrpid == 1
    household_property_income = (
        household.tentyp2.isin((5, 6)) * household.subrent
    )  # Owned and subletting
    persons_household_property_income = (
        pd.Series(
            household_property_income[person.household_id].values,
            index=person.person_id,
        )
        .fillna(0)
        .values
    )
    pe_person["property_income"] = (
        np.maximum(
            0,
            is_head * persons_household_property_income
            + person.cvpay
            + person.royyr1,
        )
        * WEEKS_IN_YEAR
    )
    maintenance_to_self = np.maximum(
        pd.Series(
            np.where(person.mntus1 == 2, person.mntusam1, person.mntamt1)
        ).fillna(0),
        0,
    )
    maintenance_from_dwp = person.mntamt2
    pe_person["maintenance_income"] = (
        sum_positive_variables([maintenance_to_self, maintenance_from_dwp])
        * WEEKS_IN_YEAR
    )

    odd_job_income = sum_to_entity(
        oddjob.ojamt * (oddjob.ojnow == 1), oddjob.person_id, person.person_id
    )

    MISC_INCOME_FIELDS = [
        "allpay2",
        "royyr2",
        "royyr3",
        "royyr4",
        "chamtern",
        "chamttst",
    ]

    pe_person["miscellaneous_income"] = (
        odd_job_income + sum_from_positive_fields(person, MISC_INCOME_FIELDS)
    ) * WEEKS_IN_YEAR

    PRIVATE_TRANSFER_INCOME_FIELDS = [
        "apamt",
        "apdamt",
        "pareamt",
        "allpay2",
        "allpay3",
        "allpay4",
    ]

    pe_person["private_transfer_income"] = (
        sum_from_positive_fields(person, PRIVATE_TRANSFER_INCOME_FIELDS)
        * WEEKS_IN_YEAR
    )

    pe_person["lump_sum_income"] = person.redamt

    pe_person["student_loan_repayments"] = person.slrepamt * WEEKS_IN_YEAR

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
        pe_person[benefit + "_reported"] = (
            sum_to_entity(
                benefits.benamt * (benefits.benefit == code),
                benefits.person_id.values,
                person.person_id,
            )
            * WEEKS_IN_YEAR
        )

    pe_person["jsa_contrib_reported"] = (
        sum_to_entity(
            benefits.benamt
            * (benefits.var2.isin((1, 3)))
            * (benefits.benefit == 14),
            benefits.person_id,
            person.person_id,
        )
        * WEEKS_IN_YEAR
    )
    pe_person["jsa_income_reported"] = (
        sum_to_entity(
            benefits.benamt
            * (benefits.var2.isin((2, 4)))
            * (benefits.benefit == 14),
            benefits.person_id,
            person.person_id,
        )
        * WEEKS_IN_YEAR
    )
    pe_person["esa_contrib_reported"] = (
        sum_to_entity(
            benefits.benamt
            * (benefits.var2.isin((1, 3)))
            * (benefits.benefit == 16),
            benefits.person_id,
            person.person_id,
        )
        * WEEKS_IN_YEAR
    )
    pe_person["esa_income_reported"] = (
        sum_to_entity(
            benefits.benamt
            * (benefits.var2.isin((2, 4)))
            * (benefits.benefit == 16),
            benefits.person_id,
            person.person_id,
        )
        * WEEKS_IN_YEAR
    )

    pe_person["bsp_reported"] = (
        sum_to_entity(
            benefits.benamt * (benefits.benefit.isin((6, 9))),
            benefits.person_id,
            person.person_id,
        )
        * WEEKS_IN_YEAR
    )

    pe_person["winter_fuel_allowance_reported"] /= WEEKS_IN_YEAR

    pe_person["statutory_sick_pay"] = person.sspadj * WEEKS_IN_YEAR
    pe_person["statutory_maternity_pay"] = person.smpadj * WEEKS_IN_YEAR

    pe_person["student_loans"] = np.maximum(person.tuborr, 0)
    if "adema" not in person.columns:
        person["adema"] = person.eduma
        person["ademaamt"] = person.edumaamt
    pe_person["adult_ema"] = fill_with_mean(person, "adema", "ademaamt")
    pe_person["child_ema"] = fill_with_mean(person, "chema", "chemaamt")

    pe_person["access_fund"] = np.maximum(person.accssamt, 0) * WEEKS_IN_YEAR

    pe_person["education_grants"] = np.maximum(
        person[["grtdir1", "grtdir2"]].sum(axis=1), 0
    )

    pe_person["council_tax_benefit_reported"] = np.maximum(
        (person.hrpid == 1)
        * pd.Series(
            household.ctrebamt[person.household_id.values].values,
            index=person.person_id,
        )
        .fillna(0)
        .values
        * WEEKS_IN_YEAR,
        0,
    )

    pe_person["healthy_start_vouchers"] = person.heartval * WEEKS_IN_YEAR

    pe_person["free_school_breakfasts"] = person.fsbval * WEEKS_IN_YEAR
    pe_person["free_school_fruit_veg"] = person.fsfvval * WEEKS_IN_YEAR
    pe_person["free_school_meals"] = person.fsmval * WEEKS_IN_YEAR

    pe_person["maintenance_expenses"] = (
        pd.Series(
            np.where(
                maintenance.mrus == 2, maintenance.mruamt, maintenance.mramt
            )
        )
        .groupby(maintenance.person_id)
        .sum()
        .reindex(person.person_id)
        .fillna(0)
        .values
        * WEEKS_IN_YEAR
    )
    pe_household["rent"] = household.hhrent.fillna(0).values * WEEKS_IN_YEAR
    pe_household["mortgage_interest_repayment"] = (
        household.mortint.fillna(0).values * WEEKS_IN_YEAR
    )
    mortgage_capital = np.where(
        mortgage.rmort == 1, mortgage.rmamt, mortgage.borramt
    )
    mortgage_capital_repayment = sum_to_entity(
        mortgage_capital / mortgage.mortend,
        mortgage.household_id,
        household.index,
    )
    pe_household["mortgage_capital_repayment"] = mortgage_capital_repayment

    pe_person["childcare_expenses"] = (
        sum_to_entity(
            childcare.chamt
            * (childcare.cost == 1)
            * (childcare.registrd == 1),
            childcare.person_id,
            person.person_id,
        )
        * 52
    )

    pe_person["personal_pension_contributions"] = np.maximum(
        0,
        sum_to_entity(
            pen_prov.penamt[pen_prov.stemppen.isin((5, 6))],
            pen_prov.person_id,
            person.person_id,
        ).clip(0, pen_prov.penamt.quantile(0.95))
        * WEEKS_IN_YEAR,
    )
    pe_person["employee_pension_contributions"] = np.maximum(
        0,
        sum_to_entity(job.deduc1.fillna(0), job.person_id, person.person_id)
        * WEEKS_IN_YEAR,
    )
    pe_person["employer_pension_contributions"] = (
        pe_person["employee_pension_contributions"] * 3
    )  # Rough estimate based on aggregates.
    # Salary sacrifice pension contributions from FRS Job table (SPNAMT field)
    # SPNAMT represents employer pension contributions made via salary sacrifice
    # arrangements where employees forego salary in exchange for increased pension
    # contributions. This is separate from regular employee pension contributions
    # (deduc1) and provides tax advantages for both employer and employee.
    # Uses same pattern as employee_pension_contributions (deduc1) without outlier
    # clipping, as job-level data is generally cleaner than pension provider data.
    # Source: https://datacatalogue.ukdataservice.ac.uk/datasets/dataset/630d4a8d-ba6a-82b3-f33d-c713c66efcb3
    # Note: Values are annualized from weekly amounts reported in the survey.
    pe_person["pension_contributions_via_salary_sacrifice"] = np.maximum(
        0,
        sum_to_entity(job.spnamt.fillna(0), job.person_id, person.person_id)
        * WEEKS_IN_YEAR,
    )

    # Salary sacrifice participation indicator from SALSAC field
    # Used for imputation: 1 = Yes, 0 = No, -1 = not asked (skip)
    # This allows distinguishing between explicit No responses and
    # respondents who were not asked the question (imputation candidates)
    if "salsac_raw" in job.columns:
        salsac_numeric = (
            job["salsac_raw"]
            .map({"1": 1, "2": 0, " ": -1})
            .fillna(-1)
            .astype(int)
        )
        # Aggregate to person level: take max (any job with SS = person has SS)
        pe_person["salary_sacrifice_reported"] = np.clip(
            sum_to_entity(
                (salsac_numeric == 1).astype(int),
                job.person_id,
                person.person_id,
            ),
            0,
            1,
        )
        # Track if person was asked about SS in any job (for imputation)
        pe_person["salary_sacrifice_asked"] = np.clip(
            sum_to_entity(
                (salsac_numeric >= 0).astype(int),
                job.person_id,
                person.person_id,
            ),
            0,
            1,
        )
    else:
        # If SALSAC not available, mark all as not asked
        pe_person["salary_sacrifice_reported"] = 0
        pe_person["salary_sacrifice_asked"] = 0

    pe_household["housing_service_charges"] = (
        pd.DataFrame(
            [
                household[f"chrgamt{i}"] * (household[f"chrgamt{i}"] > 0)
                for i in range(1, 10)
            ]
        )
        .sum()
        .values
        * WEEKS_IN_YEAR
    )
    pe_household["structural_insurance_payments"] = (
        household.struins.values * WEEKS_IN_YEAR
    )
    pe_household["water_and_sewerage_charges"] = (
        pd.Series(
            np.where(
                household.gvtregno == 12,
                household.csewamt + household.cwatamtd,
                household.watsewrt,
            )
        )
        .fillna(0)
        .values
        * WEEKS_IN_YEAR
    )

    pe_household["external_child_payments"] = sum_to_entity(
        extchild.nhhamt * WEEKS_IN_YEAR,
        extchild.household_id,
        household.index,
    )

    dataset = UKSingleYearDataset(
        person=pe_person,
        benunit=pe_benunit,
        household=pe_household,
        fiscal_year=year,
    )

    # Randomly select broad rental market areas from regions.
    from policyengine_uk import Microsimulation

    sim = Microsimulation(dataset=dataset)
    region = sim.populations["benunit"].household(
        "region", dataset.time_period
    )
    lha_category = sim.calculate("LHA_category", year)

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
                "household_id", sim.dataset.time_period
            ),
        }
    )

    df = df.groupby("household_id").brma.aggregate(
        lambda x: x.sample(n=1).iloc[0]
    )
    brmas = df[sim.calculate("household_id")].values

    pe_household["brma"] = brmas

    parameters = sim.tax_benefit_system.parameters
    benefit = parameters(year).gov.dwp

    pe_person["is_disabled_for_benefits"] = (
        pe_person.dla_sc_reported
        + pe_person.dla_m_reported
        + pe_person.pip_m_reported
        + pe_person.pip_dl_reported
    ) > 0

    THRESHOLD_SAFETY_GAP = 1 * WEEKS_IN_YEAR

    pe_person["is_enhanced_disabled_for_benefits"] = (
        pe_person.dla_sc_reported
        > benefit.dla.self_care.higher * WEEKS_IN_YEAR - THRESHOLD_SAFETY_GAP
    )

    # Child Tax Credit Regulations 2002 s. 8
    paragraph_3 = (
        pe_person.dla_sc_reported
        >= benefit.dla.self_care.higher * WEEKS_IN_YEAR - THRESHOLD_SAFETY_GAP
    )
    paragraph_4 = (
        pe_person.pip_dl_reported
        >= benefit.pip.daily_living.enhanced * WEEKS_IN_YEAR
        - THRESHOLD_SAFETY_GAP
    )
    paragraph_5 = pe_person.afcs_reported > 0
    pe_person["is_severely_disabled_for_benefits"] = (
        paragraph_3 | paragraph_4 | paragraph_5
    )

    # Add random variables which are for now in policyengine-uk.

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

    for variable in RANDOM_VARIABLES:
        value = sim.calculate(variable).values
        entity = sim.tax_benefit_system.variables[variable].entity.key
        if entity == "person":
            pe_person[variable] = value
        elif entity == "household":
            pe_household[variable] = value
        elif entity == "benunit":
            pe_benunit[variable] = value

    # Add Tax-Free Childcare assumptions

    count_benunits = len(pe_benunit)

    extended_would_claim = np.random.random(count_benunits) < 0.812
    tfc_would_claim = np.random.random(count_benunits) < 0.586
    universal_would_claim = np.random.random(count_benunits) < 0.563
    targeted_would_claim = np.random.random(count_benunits) < 0.597

    # Generate extended childcare hours usage values with mean 15.019 and sd 4.972
    extended_hours_values = np.random.normal(15.019, 4.972, count_benunits)
    # Clip values to be between 0 and 30 hours
    extended_hours_values = np.clip(extended_hours_values, 0, 30)

    pe_benunit["would_claim_extended_childcare"] = extended_would_claim
    pe_benunit["would_claim_tfc"] = tfc_would_claim
    pe_benunit["would_claim_universal_childcare"] = universal_would_claim
    pe_benunit["would_claim_targeted_childcare"] = targeted_would_claim

    # Add the maximum extended childcare hours usage
    pe_benunit["maximum_extended_childcare_hours_usage"] = (
        extended_hours_values
    )

    # Add marital status at the benefit unit level

    pe_benunit["is_married"] = frs["benunit"].famtypb2.isin([5, 7])

    # Stochastically set property_purchased based on UK housing transaction rate.
    # Previously defaulted to True in policyengine-uk, causing all households
    # to be charged SDLT as if they just bought their property (£370bn total).
    #
    # Sources:
    # - Transactions: HMRC 2024 - 1.1m/year
    #   https://www.gov.uk/government/statistics/monthly-property-transactions-completed-in-the-uk-with-value-40000-or-above
    # - Households: ONS 2024 - 28.6m
    #   https://www.ons.gov.uk/peoplepopulationandcommunity/birthsdeathsandmarriages/families/bulletins/familiesandhouseholds/2024
    # - Rate: 1.1m / 28.6m = 3.85%
    #
    # Verification against official SDLT revenue (2024-25):
    # - Official SDLT: £13.9bn (https://www.gov.uk/government/statistics/uk-stamp-tax-statistics)
    # - With fix (3.85%): £15.7bn (close to official)
    # - Without fix (100%): £370bn (26x too high)
    PROPERTY_PURCHASE_RATE = 0.0385
    pe_household["property_purchased"] = (
        np.random.random(len(pe_household)) < PROPERTY_PURCHASE_RATE
    )

    dataset = UKSingleYearDataset(
        person=pe_person,
        benunit=pe_benunit,
        household=pe_household,
        fiscal_year=year,
    )

    return dataset


if __name__ == "__main__":
    frs = create_frs(
        raw_frs_folder=STORAGE_FOLDER / "frs_2022_23",
        year=2022,
    )
    frs.save(STORAGE_FOLDER / "frs_2022.h5")
