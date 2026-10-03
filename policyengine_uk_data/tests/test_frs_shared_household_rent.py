import numpy as np
import pandas as pd
import pytest

from policyengine_uk_data.datasets.frs import frs_liable_for_share_of_household_rent

CONVENTIONAL, SHARED = 1, 2
UNIVERSAL_CREDIT, HOUSING_BENEFIT = 95, 94


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


def uc_records(rows):
    """FRS benefits rows: (benefit unit, benefit code, weekly housing element)."""
    return pd.DataFrame(rows, columns=["benunit_id", "benefit", "uchousel"])


def test_the_result_is_one_boolean_per_benefit_unit():
    benunit, person, household = frames(
        [(1_01, 1, 0), (1_02, 1, 0), (1_03, 1, 40)],
        [(1_001, 1_01, 0), (1_002, 1_02, 120), (1_003, 1_03, 0)],
        {1: SHARED},
    )
    liable = frs_liable_for_share_of_household_rent(benunit, person, household)
    assert isinstance(liable, np.ndarray)
    assert liable.dtype == bool
    assert liable.shape == (len(benunit),)


def test_the_first_unit_is_never_marked_whatever_it_reports():
    # Benefit unit 1 is liable as the household head's family; the flag is
    # only for the later units.
    benunit, person, household = frames(
        [(1_01, 1, 60), (1_02, 1, 0)],
        [(1_001, 1_01, 150), (1_002, 1_02, 0)],
        {1: SHARED},
    )
    benefits = uc_records([(1_01, UNIVERSAL_CREDIT, 100)])
    liable = frs_liable_for_share_of_household_rent(
        benunit, person, household, benefits
    )
    assert liable.tolist() == [False, False]


def test_a_paying_unit_does_not_mark_a_non_paying_unit_of_its_household():
    benunit, person, household = frames(
        [(1_01, 1, 0), (1_02, 1, 0), (1_03, 1, 0)],
        [(1_001, 1_01, 0), (1_002, 1_02, 130), (1_003, 1_03, 0)],
        {1: SHARED},
    )
    liable = frs_liable_for_share_of_household_rent(benunit, person, household)
    assert liable.tolist() == [False, True, False]


def test_a_paying_unit_does_not_mark_the_same_numbered_unit_elsewhere():
    # Two shared households: unit 2 pays in the first, not in the second.
    benunit, person, household = frames(
        [(1_01, 1, 0), (1_02, 1, 0), (2_01, 2, 0), (2_02, 2, 0)],
        [(1_001, 1_01, 0), (1_002, 1_02, 130), (2_001, 2_01, 0), (2_002, 2_02, 0)],
        {1: SHARED, 2: SHARED},
    )
    liable = frs_liable_for_share_of_household_rent(benunit, person, household)
    assert liable.tolist() == [False, True, False, False]


def test_rent_on_the_second_adult_of_a_couple_counts():
    benunit, person, household = frames(
        [(1_01, 1, 0), (1_02, 1, 0)],
        [(1_001, 1_01, 0), (1_002, 1_02, 0), (1_003, 1_02, 140)],
        {1: SHARED},
    )
    liable = frs_liable_for_share_of_household_rent(benunit, person, household)
    assert liable.tolist() == [False, True]


def test_a_missing_code_on_one_partner_does_not_cancel_the_other_partners_rent():
    benunit, person, household = frames(
        [(1_01, 1, 0), (1_02, 1, 0)],
        [(1_001, 1_01, 0), (1_002, 1_02, -140), (1_003, 1_02, 140)],
        {1: SHARED},
    )
    liable = frs_liable_for_share_of_household_rent(benunit, person, household)
    assert liable.tolist() == [False, True]


@pytest.mark.parametrize("hbothamt", [-1, -9, 0, np.nan])
def test_a_missing_or_zero_housing_benefit_amount_does_not_mark_a_unit(hbothamt):
    benunit, person, household = frames(
        [(1_01, 1, 0), (1_02, 1, hbothamt)],
        [(1_001, 1_01, 0), (1_002, 1_02, 0)],
        {1: SHARED},
    )
    liable = frs_liable_for_share_of_household_rent(benunit, person, household)
    assert liable.tolist() == [False, False]


def test_the_order_of_the_tables_does_not_matter():
    units = [(1_01, 1, 0), (1_02, 1, 0), (2_01, 2, 0), (2_02, 2, 70), (2_03, 2, 0)]
    adults = [
        (1_001, 1_01, 0),
        (1_002, 1_02, 120),
        (2_001, 2_01, 0),
        (2_002, 2_02, 0),
        (2_003, 2_03, 0),
    ]
    benunit, person, household = frames(units, adults, {1: SHARED, 2: SHARED})
    expected = dict(zip(benunit.benunit_id, [False, True, False, True, False]))
    shuffled_units = benunit.iloc[[3, 0, 4, 2, 1]].reset_index(drop=True)
    shuffled_people = person.iloc[[4, 2, 0, 3, 1]].reset_index(drop=True)
    shuffled_households = household.iloc[[1, 0]]
    liable = frs_liable_for_share_of_household_rent(
        shuffled_units, shuffled_people, shuffled_households
    )
    assert liable.tolist() == [expected[i] for i in shuffled_units.benunit_id]


def test_a_universal_credit_housing_element_marks_a_unit_that_pays_nothing_itself():
    # The questionnaire takes SRENTAMT after state help with the rent, so a
    # joint tenant whose share is wholly met by Universal Credit reports no
    # rent and no housing benefit.
    benunit, person, household = frames(
        [(1_01, 1, 0), (1_02, 1, 0), (1_03, 1, 0)],
        [(1_001, 1_01, 0), (1_002, 1_02, 0), (1_003, 1_03, 0)],
        {1: SHARED},
    )
    benefits = uc_records([(1_02, UNIVERSAL_CREDIT, 100)])
    liable = frs_liable_for_share_of_household_rent(
        benunit, person, household, benefits
    )
    assert liable.tolist() == [False, True, False]


@pytest.mark.parametrize(
    "record",
    [
        (1_02, UNIVERSAL_CREDIT, 0),
        (1_02, UNIVERSAL_CREDIT, -1),
        (1_02, UNIVERSAL_CREDIT, np.nan),
        # A housing element amount on a record of another benefit.
        (1_02, HOUSING_BENEFIT, 100),
        # Another unit's Universal Credit.
        (2_02, UNIVERSAL_CREDIT, 100),
    ],
)
def test_other_benefit_records_do_not_mark_a_unit(record):
    benunit, person, household = frames(
        [(1_01, 1, 0), (1_02, 1, 0), (2_01, 2, 0), (2_02, 2, 0)],
        [(1_001, 1_01, 0), (1_002, 1_02, 0), (2_001, 2_01, 0), (2_002, 2_02, 0)],
        {1: SHARED, 2: CONVENTIONAL},
    )
    liable = frs_liable_for_share_of_household_rent(
        benunit, person, household, uc_records([record])
    )
    assert not liable.any()


def test_a_benefits_table_without_the_housing_element_uses_the_other_two_signals():
    benunit, person, household = frames(
        [(1_01, 1, 0), (1_02, 1, 0), (1_03, 1, 55)],
        [(1_001, 1_01, 0), (1_002, 1_02, 0), (1_003, 1_03, 0)],
        {1: SHARED},
    )
    benefits = uc_records([(1_02, UNIVERSAL_CREDIT, 100)]).drop(columns="uchousel")
    liable = frs_liable_for_share_of_household_rent(
        benunit, person, household, benefits
    )
    assert liable.tolist() == [False, False, True]
