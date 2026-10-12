"""FRS dividends land on the person who holds the account (policyengine-uk#1948)."""

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings, strategies as st

from policyengine_uk_data.datasets.frs import WEEKS_IN_YEAR, frs_dividend_income


def test_dividends_are_keyed_on_person_id_not_row_position():
    # Person ids are household_id * 1000 + person number, so they never
    # coincide with the positional index the rows happen to sit at.
    person_ids = pd.Series([1_001, 1_002, 2_001, 2_002])
    account = pd.DataFrame(
        {
            "person_id": [1_002, 2_001, 2_001, 2_002, 1_001, 2_002],
            # 8: stocks and shares; 7: unit/investment trusts; 6: gilts; 1: bank account
            "account": [8, 7, 6, 6, 1, 8],
            "accint": [10.0, 4.0, 2.0, 3.0, 50.0, 0.0],
            "invtax": [2, 1, 1, 2, 2, 2],
        }
    )
    dividends = frs_dividend_income(account, person_ids)
    weekly = np.array(
        [
            0.0,  # 1,001 has only a bank account
            10.0,  # 1,002: shares, not taxed at source
            4.0 * 1.25 + 2.0 * 1.25,  # 2,001: trusts and gilts taxed at source
            0.0,  # 2,002: gilts not taxed at source are excluded; shares pay 0
        ]
    )
    assert dividends == pytest.approx(weekly * WEEKS_IN_YEAR)


def test_dividends_are_never_negative():
    person_ids = pd.Series([1_001])
    account = pd.DataFrame(
        {"person_id": [1_001], "account": [8], "accint": [-5.0], "invtax": [2]}
    )
    assert frs_dividend_income(account, person_ids).tolist() == [0.0]


@st.composite
def dividend_account_cases(draw):
    household_ids = draw(
        st.lists(st.integers(1, 50), min_size=1, max_size=8, unique=True)
    )
    person_ids = [household_id * 1_000 + 1 for household_id in household_ids]
    rows = draw(
        st.lists(
            st.tuples(
                st.sampled_from(person_ids),
                st.sampled_from([1, 6, 7, 8, 9]),
                st.integers(0, 1_000_000),
                st.sampled_from([1, 2]),
            ),
            max_size=40,
        )
    )
    account = pd.DataFrame(rows, columns=["person_id", "account", "pence", "invtax"])
    account["accint"] = account.pop("pence") / 100
    # Neither frame's index can be mistaken for an FRS person identifier.
    account.index = np.arange(len(account)) * 7 + 100
    person_ids = pd.Series(person_ids[::-1], index=np.arange(len(person_ids)) * 3 + 10)
    return account, person_ids


@given(dividend_account_cases())
@settings(max_examples=100, deadline=None)
def test_eligible_gross_dividends_are_conserved_when_joining_to_people(case):
    account, person_ids = case
    expected_by_person = {person_id: 0.0 for person_id in person_ids}
    for row in account.itertuples():
        if row.account in (7, 8) or (row.account == 6 and row.invtax == 1):
            gross_weekly_amount = row.accint / (0.8 if row.invtax == 1 else 1)
            expected_by_person[row.person_id] += gross_weekly_amount * WEEKS_IN_YEAR

    dividends = frs_dividend_income(account, person_ids)

    assert dividends.sum() == pytest.approx(sum(expected_by_person.values()))
    assert dict(zip(person_ids, dividends)) == pytest.approx(expected_by_person)
