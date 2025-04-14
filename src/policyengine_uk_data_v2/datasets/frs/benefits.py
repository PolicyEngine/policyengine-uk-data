import numpy as np
import pandas as pd
from policyengine_uk_data_v2.utils import *
from .ukda import FRS

def add_benefits(person, frs: FRS, policy_parameters):
    sp_age = np.zeros_like(person.birth_year)
    for i in range(len(person)):
        sp_age[i] = calculate_state_pension_age(
            person["birth_year"].loc[i],
            person["gender"].loc[i],
        )
    person["state_pension_age"] = sp_age.astype(int)
    is_sp_age = person["age"] >= person["state_pension_age"]
    year_reaching_sp_age = person["birth_year"] + person["state_pension_age"]
    sp_params = policy_parameters.gov.dwp.state_pension.new_state_pension
    new_sp_active_year = 2016
    reached_sp_age_after_new_sp = (
        year_reaching_sp_age > new_sp_active_year
    )
    person["state_pension_type"] = np.select([
        is_sp_age & ~reached_sp_age_after_new_sp,
        is_sp_age & reached_sp_age_after_new_sp,
        ~is_sp_age,
    ], [
        "BASIC",
        "NEW",
        "NONE",
    ])

    # Income Support

    person["would_claim_IS"] = get_benefit_with_code(19, person, frs.benefits) > 0

    return person

def get_benefit_with_code(code: int, person: pd.DataFrame, benefits: pd.DataFrame):
    return sum_to_entity(
        benefits.BENAMT * (benefits.BENEFIT == code),
        benefits.person_id,
        person.person_id,
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