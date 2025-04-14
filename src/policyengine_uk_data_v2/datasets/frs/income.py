import numpy as np
from policyengine_uk_data_v2.utils import *

def add_incomes(person, benunit, household, state, frs, _frs_person):
    person["employment_income"] = _frs_person.INEARNS * 52

    pension = frs.pension

    pension_payment = sum_to_entity(
        pension.PENPAY * (pension.PENPAY > 0),
        pension.person_id,
        person.person_id,
    )
    pension_tax_paid = sum_to_entity(
        (pension.PTAMT * ((pension.PTINC == 2) & (pension.PTAMT > 0))),
        pension.person_id,
        person.person_id,
    )
    pension_deductions_removed = sum_to_entity(
        pension.POAMT
        * (((pension.POINC == 2) | (pension.PENOTH == 1)) & (pension.POAMT > 0)),
        pension.person_id,
        person.person_id,
    )

    person["private_pension_income"] = (
        pension_payment + pension_tax_paid + pension_deductions_removed
    ) * 52

    person["self_employment_income"] =  _frs_person.SEINCAM2

    INVERTED_BASIC_RATE = 1.25
    account = frs.accounts
    person["tax_free_savings_income"] = (
        sum_to_entity(
            account.ACCINT * (account.ACCOUNT == 21),
            account.person_id,
            person.person_id,
        )
        * 52
    )
    taxable_savings_interest = (
        sum_to_entity(
            (account.ACCINT * np.where(account.ACCTAX == 1, INVERTED_BASIC_RATE, 1))
            * (account.ACCOUNT.isin((1, 3, 5, 27, 28))),
            account.person_id,
            person.person_id,
        )
        * 52
    )
    person["savings_interest_income"] = (
        taxable_savings_interest + person["tax_free_savings_income"]
    )
    person["dividend_income"] = (
        sum_to_entity(
            (account.ACCINT * np.where(account.INVTAX == 1, INVERTED_BASIC_RATE, 1))
            * (
                ((account.ACCOUNT == 6) & (account.INVTAX == 1))  # GGES
                | account.ACCOUNT.isin((7, 8))  # Stocks/shares/UITs
            ),
            account.person_id,
            person.person_id,
        )
        * 52
    )
    is_head = _frs_person.HRPID == 1
    household_property_income = (
        frs.househol.TENTYP2.isin((5, 6)) * frs.househol.SUBRENT
    )  # Owned and subletting
    persons_household_property_income = (
        pd.Series(
            household_property_income, index=frs.househol.household_id.values
        )
        .loc[person.person_household_id]
        .values
    )
    person["property_income"] = (
        max_(
            0,
            is_head * persons_household_property_income
            + concat(frs.adult.CVPAY, np.zeros_like(frs.child.CVPAY))
            + concat(frs.adult.ROYYR1, np.zeros_like(frs.child.ROYYR1)),
        )
        * 52
    )

    
    maintenance_to_self = max_(
        pd.Series(
            np.where(
                _frs_person.MNTUS1 == 2, _frs_person.MNTUSAM1, _frs_person.MNTAMT1
            )
        ).fillna(0),
        0,
    )
    maintenance_from_dwp = _frs_person.MNTAMT2.values
    person["maintenance_income"] = (
        sum_positive_variables([maintenance_to_self, maintenance_from_dwp]) * 52
    )

    odd_job_income = sum_to_entity(
        frs.oddjob.OJAMT * (frs.oddjob.OJNOW == 1),
        frs.oddjob.person_id,
        person.person_id,
    ).values

    MISC_INCOME_FIELDS = [
        "ALLPAY2",
        "ROYYR2",
        "ROYYR3",
        "ROYYR4",
        "CHAMTERN",
        "CHAMTTST",
    ]

    person["miscellaneous_income"] = (
        odd_job_income
        + sum_from_positive_fields(_frs_person, MISC_INCOME_FIELDS).values
    ) * 52

    PRIVATE_TRANSFER_INCOME_FIELDS = [
        "APAMT",
        "APDAMT",
        "PAREAMT",
        "ALLPAY1",
        "ALLPAY3",
        "ALLPAY4",
    ]

    person["private_transfer_income"] = (
        sum_from_positive_fields(_frs_person, PRIVATE_TRANSFER_INCOME_FIELDS).values
        * 52
    )

    person["lump_sum_income"] = _frs_person.REDAMT.values
    
    return person, benunit, household, state