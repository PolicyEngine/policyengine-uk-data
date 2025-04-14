import numpy as np


def add_ids(person, benunit, household, state, frs, _frs_person):
    person["person_id"] = _frs_person.person_id
    person["person_benunit_id"] = _frs_person.benunit_id
    person["person_household_id"] = _frs_person.household_id
    person["person_state_id"] = np.ones(len(_frs_person), dtype=int)
    benunit["benunit_id"] = frs.benunit.benunit_id
    household["household_id"] = frs.househol.household_id
    state["state_id"] = np.array([1])
    return person, benunit, household, state
