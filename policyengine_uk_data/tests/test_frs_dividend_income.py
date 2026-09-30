"""FRS dividends land on the person who holds the account (policyengine-uk#1948)."""

import numpy as np
import pandas as pd
import pytest

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
