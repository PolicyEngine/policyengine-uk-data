import numpy as np
import pandas as pd
import pytest

from policyengine_uk_data.datasets.frs import frs_liable_for_share_of_household_rent

CONVENTIONAL, SHARED = 1, 2


def frames(units, adults, hhstat):
    benunit = pd.DataFrame(units, columns=["benunit_id", "household_id", "hbothamt"])
    person = pd.DataFrame(adults, columns=["person_id", "benunit_id", "srentamt"])
    household = pd.DataFrame(
        {"hhstat": list(hhstat.values())}, index=list(hhstat.keys())
    )
    return benunit, person, household


def test_later_units_of_a_shared_household_that_pay_rent_are_liable():
    benunit, person, household = frames(
        [(1_01, 1, 0), (1_02, 1, 0), (1_03, 1, 0)],
        [(1_001, 1_01, 0), (1_002, 1_02, 120), (1_003, 1_03, 110)],
        {1: SHARED},
    )
    liable = frs_liable_for_share_of_household_rent(benunit, person, household)
    # Benefit unit 1 is the household head's, liable for the rent anyway.
    assert liable.tolist() == [False, True, True]


def test_housing_benefit_alone_marks_a_unit_liable():
    benunit, person, household = frames(
        [(1_01, 1, 0), (1_02, 1, 85)],
        [(1_001, 1_01, 0), (1_002, 1_02, 0)],
        {1: SHARED},
    )
    liable = frs_liable_for_share_of_household_rent(benunit, person, household)
    assert liable.tolist() == [False, True]


@pytest.mark.parametrize("srentamt", [0, -1, np.nan])
def test_a_later_unit_paying_nothing_is_not_liable(srentamt):
    benunit, person, household = frames(
        [(1_01, 1, 0), (1_02, 1, np.nan)],
        [(1_001, 1_01, 0), (1_002, 1_02, srentamt)],
        {1: SHARED},
    )
    liable = frs_liable_for_share_of_household_rent(benunit, person, household)
    assert liable.tolist() == [False, False]


def test_units_of_conventional_households_are_never_sharers():
    benunit, person, household = frames(
        [(1_01, 1, 0), (1_02, 1, 50), (2_01, 2, 0), (2_02, 2, 0)],
        [(1_001, 1_01, 0), (1_002, 1_02, 0), (2_001, 2_01, 0), (2_002, 2_02, 90)],
        {1: CONVENTIONAL, 2: CONVENTIONAL},
    )
    liable = frs_liable_for_share_of_household_rent(benunit, person, household)
    assert not liable.any()


def test_a_couple_counts_once_and_households_do_not_leak():
    # A couple in unit 2 of a shared household reports SRENTAMT on one
    # partner's record; the next household is conventional.
    benunit, person, household = frames(
        [(1_01, 1, 0), (1_02, 1, 0), (2_01, 2, 0), (2_02, 2, 0)],
        [
            (1_001, 1_01, 0),
            (1_002, 1_02, 200),
            (1_003, 1_02, 0),
            (2_001, 2_01, 0),
            (2_002, 2_02, 0),
        ],
        {1: SHARED, 2: CONVENTIONAL},
    )
    liable = frs_liable_for_share_of_household_rent(benunit, person, household)
    assert liable.tolist() == [False, True, False, False]
