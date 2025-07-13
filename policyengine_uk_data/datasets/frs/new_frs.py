import numpy as np
from policyengine_uk_data.utils.datasets import (
    sum_to_entity,
    categorical,
    sum_from_positive_fields,
    sum_positive_variables,
    fill_with_mean,
    STORAGE_FOLDER,
)

year = 2022

# Combine adult and child tables for convenience

frs["person"] = pd.concat([frs["adult"], frs["child"]]).sort_index().fillna(0)

person = frs["person"]
benunit = frs["benunit"]
household = frs["househol"]
pension = frs["pension"]
oddjob = frs["oddjob"]
account = frs["accounts"]
job = frs["job"]

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
pe_household["household_weight"] = household.gross4

# Add basic personal variables
age = person.age80 + person.age
pe_person["age"] = age
pe_person["birth_year"] = np.ones_like(person.age) * (year - age)
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
if "FTED" in person.columns:
    fted = person.fted
else:
    fted = person.educft  # Renamed in FRS 2022-23
typeed2 = person.typeed2
pe_person["current_education"] = np.select(
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
)
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
)
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
)

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
pe_household["council_tax_band"] = categorical(
    household.ctband, 1, range(1, 10), BANDS
).fillna("D").values
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
    pension.penpay * (pension.penpay > 0), pension.person_id, person.person_id
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
        person.person_id,
    )
    * 52
)
is_head = person.hrpid == 1
household = household.set_index("household_id")
household_property_income = (
    household.tentyp2.isin((5, 6)) * household.subrent
)  # Owned and subletting
persons_household_property_income = pd.Series(
    household_property_income[person.household_id].values,
    index=person.person_id,
).fillna(0).values
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
    sum_from_positive_fields(person, PRIVATE_TRANSFER_INCOME_FIELDS) * WEEKS_IN_YEAR
)

pe_person["lump_sum_income"] = person.redamt

pe_person["student_loan_repayments"] = person.slrepamt * WEEKS_IN_YEAR

