import numpy as np
import pandas as pd
from typing import Tuple

from .ukda import FRS


def add_ids(
    person: pd.DataFrame,
    benunit: pd.DataFrame,
    household: pd.DataFrame,
    state: pd.DataFrame,
    frs: FRS,
    _frs_person: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Add ID columns to all dataframes and establish entity relationships.
    
    Args:
        person (pd.DataFrame): Person-level dataframe.
        benunit (pd.DataFrame): Benefit unit-level dataframe.
        household (pd.DataFrame): Household-level dataframe.
        state (pd.DataFrame): State-level dataframe.
        frs (FRS): The FRS data object.
        _frs_person (pd.DataFrame): Combined adult and child dataframe.
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]: Updated dataframes.
    """
    person["person_id"] = _frs_person.person_id
    person["person_benunit_id"] = _frs_person.benunit_id
    person["person_household_id"] = _frs_person.household_id
    person["person_state_id"] = np.ones(len(_frs_person), dtype=int)
    benunit["benunit_id"] = frs.benunit.benunit_id
    household["household_id"] = frs.househol.household_id
    state["state_id"] = np.array([1])
    return person, benunit, household, state
