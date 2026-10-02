"""FRS self-employment profit is split into a profit and a trading loss.

SEINCAM2 records a loss as a negative profit. The build puts the profit in
``self_employment_income`` and the loss, as a positive amount, in
``trading_loss``, so policyengine-uk can apply each programme's own loss rule.

Invariants, for any weekly profit (including missing values):

1. Both outputs are non-negative and at most one is positive.
2. Conservation: profit less loss is the annualised reported profit.
3. Profit never falls and loss never rises as the reported profit rises.

And for the built datasets:

4. No FRS person has both a profit and a loss, and no loss is negative.
5. SPI-donor rows of the enhanced FRS carry no trading loss; their
   self-employment profits come from the SPI, not their FRS donor.
"""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from policyengine_uk_data.datasets.frs import (
    WEEKS_IN_YEAR,
    split_self_employment_profit,
)

weekly_profits = arrays(
    np.float64,
    st.integers(1, 50),
    elements=st.one_of(
        st.floats(-50_000, 50_000, allow_nan=False, allow_infinity=False),
        st.just(0.0),
        st.just(np.nan),
    ),
)


@settings(max_examples=200, deadline=None, derandomize=True)
@given(weekly_profits)
def test_split_is_non_negative_exclusive_and_conserves_profit(weekly):
    income, loss = split_self_employment_profit(weekly)
    assert np.all(income >= 0) and np.all(loss >= 0)
    assert np.all((income == 0) | (loss == 0))
    reported = np.nan_to_num(weekly, nan=0.0) * WEEKS_IN_YEAR
    np.testing.assert_allclose(income - loss, reported, rtol=0, atol=1e-6)


@settings(max_examples=200, deadline=None, derandomize=True)
@given(weekly_profits, st.floats(0, 10_000, allow_nan=False))
def test_split_is_monotone_in_reported_profit(weekly, rise):
    income, loss = split_self_employment_profit(weekly)
    higher_income, higher_loss = split_self_employment_profit(
        np.nan_to_num(weekly, nan=0.0) + rise
    )
    assert np.all(higher_income >= income)
    assert np.all(higher_loss <= loss)


def test_frs_profit_and_loss_never_both_positive(frs):
    if "trading_loss" not in frs.person.columns:
        pytest.skip("Dataset built before trading_loss was added")
    income = frs.person["self_employment_income"].to_numpy()
    loss = frs.person["trading_loss"].to_numpy()
    assert loss.min() >= 0
    assert not np.any((income > 0) & (loss > 0))


def test_spi_donor_rows_carry_no_trading_loss(enhanced_frs):
    person = enhanced_frs.person
    if "trading_loss" not in person.columns:
        pytest.skip("Dataset built before trading_loss was added")
    household = enhanced_frs.household
    synthetic_households = household.household_id[
        household.household_is_spi_synthetic.astype(bool)
    ]
    synthetic = person.person_household_id.isin(synthetic_households).to_numpy()
    loss = person["trading_loss"].to_numpy()
    assert synthetic.any(), "expected SPI-donor rows in the enhanced FRS"
    assert np.all(loss[synthetic] == 0)
    assert loss.min() >= 0
