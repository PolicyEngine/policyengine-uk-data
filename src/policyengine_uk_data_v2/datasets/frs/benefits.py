import numpy as np
import pandas as pd

from policyengine_uk_data_v2.utils import sum_to_entity

from .ukda import FRS


def add_benefits(person, benunit, household, _frs_person, frs: FRS, policy_parameters):
    sp_age = np.zeros_like(person.birth_year)
    for i in range(len(person)):
        sp_age[i] = calculate_state_pension_age(
            person["birth_year"].loc[i],
            person["gender"].loc[i],
        )
    person["state_pension_age"] = sp_age.astype(int)
    is_sp_age = person["age"] >= person["state_pension_age"]
    year_reaching_sp_age = person["birth_year"] + person["state_pension_age"]
    new_sp_active_year = 2016
    reached_sp_age_after_new_sp = year_reaching_sp_age > new_sp_active_year
    person["state_pension_type"] = np.select(
        [
            is_sp_age & ~reached_sp_age_after_new_sp,
            is_sp_age & reached_sp_age_after_new_sp,
            ~is_sp_age,
        ],
        [
            "BASIC",
            "NEW",
            "NONE",
        ],
    )

    benunit["income_support"] = (
        sum_person_to_benunit(
            get_benefit_with_code(19, person, frs.benefits), person, benunit
        )
        * 52
    )
    person["ssmg"] = get_benefit_with_code(22, person, frs.benefits) * 52
    person["bsp"] = (
        get_benefit_with_code(6, person, frs.benefits)
        + get_benefit_with_code(9, person, frs.benefits)
    ) * 52
    person["statutory_maternity_pay"] = _frs_person.SMPADJ * 52

    person["care_hours"] = np.where(
        get_benefit_with_code(13, person, frs.benefits) > 0, 35 + 5, 0
    )  # If receiving carer's allowance assume 40 hours of care (35 is minimum)
    person["esa_contrib"] = (
        sum_to_entity(
            frs.benefits.BENAMT
            * (frs.benefits.VAR2.isin((1, 3)))
            * (frs.benefits.BENEFIT == 16),
            frs.benefits.person_id,
            person.person_id,
        ).values
        * 52
    )
    benunit["would_claim_pc"] = np.random.random(len(benunit)) < 0.6
    benunit["esa_income"] = (
        sum_person_to_benunit(
            sum_to_entity(
                frs.benefits.BENAMT
                * (frs.benefits.VAR2.isin((2, 4)))
                * (frs.benefits.BENEFIT == 16),
                frs.benefits.person_id,
                person.person_id,
            ),
            person,
            benunit,
        )
        * 52
    )

    dla_sc = get_benefit_with_code(1, person, frs.benefits)
    dla_m = get_benefit_with_code(2, person, frs.benefits)
    dla = policy_parameters.gov.dwp.dla

    person["dla_sc_category"] = np.select(
        [
            dla_sc >= dla.self_care.higher,
            dla_sc >= dla.self_care.middle,
            dla_sc >= dla.self_care.lower,
            True,
        ],
        [
            "HIGHER",
            "MIDDLE",
            "LOWER",
            "NONE",
        ],
    )

    person["dla_m_category"] = np.select(
        [
            dla_m >= dla.mobility.higher,
            dla_m >= dla.mobility.lower,
            True,
        ],
        [
            "HIGHER",
            "LOWER",
            "NONE",
        ],
    )

    pip_dl = get_benefit_with_code(96, person, frs.benefits)
    pip_m = get_benefit_with_code(97, person, frs.benefits)
    pip = policy_parameters.gov.dwp.pip

    person["pip_dl_category"] = np.select(
        [
            pip_dl >= pip.daily_living.enhanced,
            pip_dl >= pip.daily_living.standard,
            True,
        ],
        [
            "HIGHER",
            "LOWER",
            "NONE",
        ],
    )

    person["pip_m_category"] = np.select(
        [
            pip_m >= pip.mobility.enhanced,
            pip_m >= pip.mobility.standard,
            True,
        ],
        [
            "HIGHER",
            "LOWER",
            "NONE",
        ],
    )

    aa_reported = get_benefit_with_code(12, person, frs.benefits)
    aa = policy_parameters.gov.dwp.attendance_allowance

    person["aa_category"] = np.select(
        [
            aa_reported >= aa.higher,
            aa_reported >= aa.lower,
            True,
        ],
        [
            "HIGHER",
            "LOWER",
            "NONE",
        ],
    )

    benunit["would_claim_housing_benefit"] = np.random.random(len(benunit)) < 0.7
    benunit["would_claim_child_benefit"] = np.random.random(len(benunit)) < 0.7
    benunit["would_claim_universal_credit"] = np.random.random(len(benunit)) < 0.7

    household["would_evade_tv_licence_fee"] = np.random.random(len(household)) < 0.07

    return person, benunit, household


def sum_person_to_benunit(values, person, benunit):
    return (
        values.groupby(person.person_id.values)
        .sum()
        .reindex(benunit.benunit_id)
        .fillna(0)
        .values
    )


def get_benefit_with_code(code: int, person: pd.DataFrame, benefits: pd.DataFrame):
    return pd.Series(
        sum_to_entity(
            benefits.BENAMT * (code == benefits.BENEFIT),
            benefits.person_id,
            person.person_id,
        ).values
    )


def calculate_state_pension_age(birth_year, gender):
    """
    Calculate the state pension age based on year of birth and gender.

    Args:
        birth_year (int): Year of birth
        gender (str): 'male' or 'female'

    Returns:
        int: State pension age in years
    """
    # Default pension ages before any changes
    if gender.lower() == "MALE" and birth_year < 1954:
        return 65

    if gender.lower() == "FEMALE":
        # Pre-1950 women
        if birth_year < 1950:
            return 60
        # Gradual increase for women born 1950-1953
        elif birth_year == 1950:
            return 61
        elif birth_year == 1951:
            return 62
        elif birth_year == 1952:
            return 63
        elif birth_year == 1953:
            return 64

    # People born 1954-1960 - pension age 66
    # (Pensions Act 2011)
    if 1954 <= birth_year <= 1960:
        return 66

    # People born 1961-1976 - pension age 67
    # (Pensions Act 2014)
    if 1961 <= birth_year <= 1976:
        return 67

    # People born 1977 and later - pension age 68
    # (Pensions Act 2007)
    return 68
