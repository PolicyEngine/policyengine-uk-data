"""FRS EMPSTATI codes to ``employment_status``, and what the ESA proxies read."""

import numpy as np
import pytest
from hypothesis import given, strategies as st
from policyengine_uk.variables.household.income.employment_status import (
    EmploymentStatus,
)

from policyengine_uk_data.datasets.frs import (
    ESA_HEALTH_EMPLOYMENT_STATUSES,
    ESA_MIN_AGE,
    FRS_EMPSTATI_EMPLOYMENT_STATUS,
    derive_employment_status_from_frs,
    derive_esa_health_condition_proxy,
    derive_esa_support_group_proxy,
)

# EMPSTATI ("Adult - Employment Status - ILO definition") value labels in the
# UKDS FRS 2024-25 data dictionary (SN 9563, adult table), with the status each
# label means. Every adult in the 2020-21, 2022-23 and 2023-24 releases also
# has a code from 1 to 11.
DATA_DICTIONARY = {
    1: ("Full-time Employee", "FT_EMPLOYED"),
    2: ("Part-time Employee", "PT_EMPLOYED"),
    3: ("Full-time Self-Employed", "FT_SELF_EMPLOYED"),
    4: ("Part-time Self-Employed", "PT_SELF_EMPLOYED"),
    5: ("Unemployed", "UNEMPLOYED"),
    6: ("Retired", "RETIRED"),
    7: ("Student", "STUDENT"),
    8: ("Looking after family/home", "CARER"),
    9: ("Permanently sick/disabled", "LONG_TERM_DISABLED"),
    10: ("Temporarily sick/injured", "SHORT_TERM_DISABLED"),
    11: ("Other Inactive", "OTHER_INACTIVE"),
}
ADULT_CODES = sorted(DATA_DICTIONARY)
HEALTH_CODES = (9, 10)

adult_rows = st.tuples(st.just(True), st.sampled_from(ADULT_CODES))
# Child-table rows have no EMPSTATI (0 once the person table fills blanks); the
# code must not matter for them.
child_rows = st.tuples(
    st.just(False),
    st.one_of(st.just(0), st.just(np.nan), st.integers(-9, 99)),
)
people = st.lists(st.one_of(adult_rows, child_rows), max_size=60)
unknown_adult_codes = st.one_of(
    st.just(np.nan),
    st.integers(-99, 0),
    st.integers(12, 999),
    st.floats(0.5, 11.5).filter(lambda x: not float(x).is_integer()),
)


def derive(rows):
    is_adult = [adult for adult, _ in rows]
    codes = [code for _, code in rows]
    return derive_employment_status_from_frs(codes, is_adult)


def expected(adult, code):
    return DATA_DICTIONARY[code][1] if adult else "CHILD"


@pytest.mark.parametrize("code", range(12))
def test_every_code_from_0_to_11(code):
    assert derive_employment_status_from_frs([code], [False]).tolist() == ["CHILD"]
    if code == 0:
        with pytest.raises(ValueError, match="EMPSTATI"):
            derive_employment_status_from_frs([code], [True])
    else:
        label, status = DATA_DICTIONARY[code]
        assert derive_employment_status_from_frs([code], [True]).tolist() == [status], (
            label
        )


def test_other_inactive_is_not_long_term_disabled():
    result = derive_employment_status_from_frs([9, 11], [True, True])
    assert result.tolist() == ["LONG_TERM_DISABLED", "OTHER_INACTIVE"]


def test_code_table_is_the_data_dictionary():
    assert FRS_EMPSTATI_EMPLOYMENT_STATUS == {
        code: status for code, (_, status) in DATA_DICTIONARY.items()
    }


def test_adult_codes_and_child_rows_cover_each_status_once():
    statuses = [*FRS_EMPSTATI_EMPLOYMENT_STATUS.values(), "CHILD"]
    assert len(set(statuses)) == len(statuses)
    assert set(statuses) == set(EmploymentStatus.__members__)


@pytest.mark.parametrize("code", [0, -1, 12, 11.5, np.nan])
def test_unknown_adult_code_fails_the_build(code):
    with pytest.raises(ValueError, match="EMPSTATI"):
        derive_employment_status_from_frs([1, code], [True, True])


def listed_codes(message):
    return message.split("FRS_EMPSTATI_EMPLOYMENT_STATUS: ")[1].split(". Map")[0]


def test_failure_message_lists_codes_in_numeric_order_with_blank_last():
    # Formatted from the float codes, not Series.astype(str), whose NaN
    # handling differs between pandas 2 and 3. A string sort would put 100
    # before 11.5.
    codes = [100, 12, np.nan, -1, 12, 11.5, 1]
    with pytest.raises(ValueError) as error:
        derive_employment_status_from_frs(codes, [True] * len(codes))
    assert listed_codes(str(error.value)) == "-1, 11.5, 12, 100, blank"


@given(st.lists(unknown_adult_codes, min_size=1, max_size=20))
def test_failure_message_lists_each_unknown_code_once_in_order(bad_codes):
    with pytest.raises(ValueError) as error:
        derive_employment_status_from_frs(bad_codes, [True] * len(bad_codes))
    listed = listed_codes(str(error.value)).split(", ")
    blank = any(np.isnan(code) for code in bad_codes)
    assert (listed[-1] == "blank") == blank
    numbers = [float(code) for code in listed[: len(listed) - blank]]
    assert numbers == sorted(numbers)
    assert set(listed) - {"blank"} == {
        f"{code:g}" for code in bad_codes if not np.isnan(code)
    }


@given(st.integers(1, 30), st.integers(1, 30))
def test_failure_message_discloses_no_count(n_twelve, n_thirteen):
    # Adults are not survey households, so no count of them is safe to print
    # in a public build log: the message depends only on which codes occur.
    codes = [12] * n_twelve + [13] * n_thirteen
    with pytest.raises(ValueError) as error:
        derive_employment_status_from_frs(codes, [True] * len(codes))
    message = str(error.value)
    assert listed_codes(message) == "12, 13"
    assert not any(character.isdigit() for character in message.replace("12, 13", ""))


@given(people)
def test_each_row_maps_on_its_own(rows):
    result = derive(rows)
    assert result.tolist() == [expected(adult, code) for adult, code in rows]


@given(people, st.randoms(use_true_random=False))
def test_mapping_commutes_with_row_order(rows, rng):
    order = list(range(len(rows)))
    rng.shuffle(order)
    statuses = derive(rows)
    assert derive([rows[i] for i in order]).tolist() == [statuses[i] for i in order]


@given(people, unknown_adult_codes, st.integers(0, 60))
def test_any_unknown_adult_code_fails_the_build(rows, bad_code, position):
    rows = list(rows)
    rows.insert(min(position, len(rows)), (True, bad_code))
    with pytest.raises(ValueError, match="EMPSTATI"):
        derive(rows)


@given(
    st.lists(
        st.tuples(
            st.integers(0, 100),  # age
            st.sampled_from(ADULT_CODES),
            st.integers(60, 68),  # State Pension age
            st.integers(0, 3_000),  # annual hours worked
            st.booleans(),  # EMPSTATI reported
        ),
        min_size=1,
        max_size=60,
    )
)
def test_esa_proxies_read_only_the_sick_or_disabled_codes(rows):
    age, codes, spa, hours, reported = map(np.array, zip(*rows))
    status = derive_employment_status_from_frs(codes, np.ones(len(rows), bool))
    health = derive_esa_health_condition_proxy(
        age=age,
        employment_status=status,
        employment_status_reported=reported,
        state_pension_age=spa,
    )
    support = derive_esa_support_group_proxy(
        age=age,
        employment_status=status,
        hours_worked=hours,
        esa_health_condition_proxy=health,
        employment_status_reported=reported,
        state_pension_age=spa,
    )
    working_age = (age >= ESA_MIN_AGE) & (age < spa)
    assert (
        health.tolist()
        == (reported & working_age & np.isin(codes, HEALTH_CODES)).tolist()
    )
    assert not (support & ~health).any()
    assert not support[codes != 9].any()
    assert not health[codes == 11].any()


@pytest.mark.parametrize("dataset_name", ["frs", "enhanced_frs"])
def test_built_dataset_statuses(dataset_name, request):
    person = request.getfixturevalue(dataset_name).person
    status = person["employment_status"].astype(str)
    assert set(status) <= set(EmploymentStatus.__members__)
    assert (status == "OTHER_INACTIVE").any()
    not_sick = ~status.isin(ESA_HEALTH_EMPLOYMENT_STATUSES)
    assert not person.loc[not_sick, "esa_health_condition_proxy"].any()
    assert not person.loc[not_sick, "esa_support_group_proxy"].any()
