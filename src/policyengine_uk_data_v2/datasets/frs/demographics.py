import numpy as np
import pandas as pd
from typing import Tuple

from policyengine_uk_data_v2.utils import concat
from .ukda import FRS


def add_demographics(
    person: pd.DataFrame,
    household: pd.DataFrame,
    frs: FRS,
    _frs_person: pd.DataFrame,
    year: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Add demographic information to person and household dataframes.
    
    Args:
        person (pd.DataFrame): Person-level dataframe.
        household (pd.DataFrame): Household-level dataframe.
        frs (FRS): The FRS data object.
        _frs_person (pd.DataFrame): Combined adult and child dataframe.
        year (int): The year of the data.
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Updated person and household dataframes.
    """
    # Add grossing weights
    household["household_weight"] = frs.househol.GROSS4

    # Add basic personal variables
    person["age"] = _frs_person.AGE80 + _frs_person.AGE
    person["birth_year"] = np.ones_like(person.age) * (year - person.age)
    # Age fields are AGE80 (top-coded) and AGE in the adult and
    # child tables, respectively.
    person["gender"] = np.where(_frs_person.SEX == 1, "MALE", "FEMALE")
    person["hours_worked"] = _frs_person.TOTHOURS.fillna(0).clip(lower=0) * 52
    person["is_household_head"] = concat(
        frs.adult.HRPID == 1, np.zeros_like(frs.child.index, dtype=bool)
    )
    person["is_benunit_head"] = concat(
        frs.adult.UPERSON == 1,
        np.zeros_like(frs.child.index, dtype=bool),
    )
    person["marital_status"] = (
        _frs_person.MARITAL.fillna(2)
        .map(
            {
                1: "MARRIED",
                2: "SINGLE",
                3: "SINGLE",
                4: "WIDOWED",
                5: "SEPARATED",
                6: "DIVORCED",
            },
        )
        .fillna("SINGLE")
    )

    fted = _frs_person.FTED if "FTED" in _frs_person.columns else _frs_person.EDUCFT
    typeed2 = _frs_person.TYPEED2
    age = person.age
    person["current_education"] = np.select(
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
            (typeed2 == 9)
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
        default="NOT_IN_EDUCATION",
    )

    # Add employment status
    person["employment_status"] = _frs_person.EMPSTATI.map(
        {
            0: "CHILD",
            1: "FT_EMPLOYED",
            2: "PT_EMPLOYED",
            3: "FT_SELF_EMPLOYED",
            4: "PT_SELF_EMPLOYED",
            5: "UNEMPLOYED",
            6: "RETIRED",
            7: "STUDENT",
            8: "CARER",
            9: "LONG_TERM_DISABLED",
            10: "SHORT_TERM_DISABLED",
        }
    ).fillna("CHILD")

    return person, household
