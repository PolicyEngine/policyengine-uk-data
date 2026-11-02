"""The State Pension annualises at the 52 weeks policyengine-uk reads it back at.

policyengine-uk divides state_pension_reported by policyengine-core's 52 weeks
a year, splits the weekly amount at the full rate of the person's State
Pension type, and pays the excess as additional State Pension (basic) or a
protected payment (new). At 365.25 / 7 every full-rate pensioner read 0.34%
above the full rate and carried an add-on they do not get. These tests hold
the build to the reading: the column round-trips to the weekly amounts, a
full-rate pensioner has no add-on, and an add-on is exactly the weekly excess.

policyengine-core holds amounts as float32, so £11,502.40 read back at 52
weeks is a ten-thousandth of a penny above £221.20 a week; the comparisons
allow a penny a year.
"""

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from policyengine_uk import Microsimulation
from policyengine_uk.data import UKSingleYearDataset

from policyengine_uk_data.datasets.frs import (
    BENEFIT_CODES,
    STATE_PENSION_WEEKS_IN_YEAR,
    WEEKS_IN_YEAR,
    reported_benefit_amounts,
)

DATA_YEAR = 2024
# Full weekly rates in 2024-25 (Social Security Benefits Up-rating Order 2024,
# SI 2024/242): new State Pension £221.20, basic State Pension £169.50.
FULL_RATE = {"NEW": 221.20, "BASIC": 169.50}
# Ages held through the projection: reached State Pension age in 2020 (new
# State Pension) and in 2005 (basic State Pension).
AGE = {"NEW": 70, "BASIC": 85}
PENNY_A_YEAR = 0.01


def benefits_table(rows) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["person_id", "benefit", "benamt"])


def state_pension_dataset(weekly, pension_types) -> UKSingleYearDataset:
    """One pensioner per household, each with a single FRS State Pension row
    of `weekly`, annualised as create_frs annualises it."""
    ids = np.arange(len(weekly))
    benefits = benefits_table(
        [(i, BENEFIT_CODES["state_pension"], amount) for i, amount in zip(ids, weekly)]
    )
    reported = reported_benefit_amounts(benefits, pd.Series(ids))
    person = pd.DataFrame(
        {
            "person_id": ids,
            "person_benunit_id": ids,
            "person_household_id": ids,
            "age": [AGE[t] for t in pension_types],
            "gender": "FEMALE",
            "state_pension_reported": reported["state_pension_reported"],
        }
    )
    household = pd.DataFrame(
        {
            "household_id": ids,
            "household_weight": np.ones(len(ids)),
            "region": "LONDON",
            "tenure_type": "OWNED_OUTRIGHT",
            "rent": 0.0,
            "council_tax": 0.0,
        }
    )
    return UKSingleYearDataset(
        person=person,
        benunit=pd.DataFrame({"benunit_id": ids}),
        household=household,
        fiscal_year=DATA_YEAR,
    )


def components(sim, year) -> dict:
    values = {
        v: sim.calculate(v, year).values.astype(float)
        for v in (
            "basic_state_pension",
            "new_state_pension",
            "additional_state_pension",
            "state_pension",
        )
    }
    values["type"] = sim.calculate("state_pension_type", year).values.astype(str)
    return values


def test_the_state_pension_alone_annualises_at_52_weeks():
    assert STATE_PENSION_WEEKS_IN_YEAR == 52
    benefits = benefits_table(
        [
            (0, 5, 221.20),  # the full new State Pension
            (1, 5, 169.50),  # the full basic State Pension...
            (1, 5, 10.00),  # ...and a second State Pension row
            (1, 13, 81.90),  # Carer's Allowance
            (2, 13, 81.90),
        ]
    )
    reported = reported_benefit_amounts(benefits, pd.Series([0, 1, 2]))

    state_pension = reported["state_pension_reported"]
    np.testing.assert_allclose(state_pension, [11_502.40, 9_334.00, 0.0])
    # Read back at policyengine-uk's 52 weeks, the full rate is the full rate.
    np.testing.assert_allclose(state_pension / 52, [221.20, 179.50, 0.0])
    np.testing.assert_allclose(
        reported["carers_allowance_reported"],
        [0.0, 81.90 * WEEKS_IN_YEAR, 81.90 * WEEKS_IN_YEAR],
    )


def test_every_other_benefit_keeps_365_25_over_7():
    # One row of £100 a week for each code, each for its own person.
    codes = list(BENEFIT_CODES.items())
    benefits = benefits_table([(i, code, 100.0) for i, (_, code) in enumerate(codes)])
    reported = reported_benefit_amounts(benefits, pd.Series(range(len(codes))))

    assert set(reported) == {f"{benefit}_reported" for benefit, _ in codes}
    for i, (benefit, _) in enumerate(codes):
        weeks = 52 if benefit == "state_pension" else 365.25 / 7
        expected = np.zeros(len(codes))
        expected[i] = 100.0 * weeks
        np.testing.assert_allclose(reported[f"{benefit}_reported"], expected)


pence = st.integers(min_value=1, max_value=100_000).map(lambda p: p / 100)
rows = st.lists(
    st.tuples(
        st.integers(min_value=0, max_value=4),
        st.sampled_from(sorted(set(BENEFIT_CODES.values()))),
        pence,
    ),
    max_size=30,
)


@given(rows=rows)
def test_each_column_reads_back_to_its_weekly_amounts(rows):
    """Divided by the weeks it was annualised at, each person's column is the
    sum of their weekly BENAMT rows with that code."""
    benefits = benefits_table(rows)
    reported = reported_benefit_amounts(benefits, pd.Series(range(5)))
    for benefit, code in BENEFIT_CODES.items():
        weeks = STATE_PENSION_WEEKS_IN_YEAR if code == 5 else WEEKS_IN_YEAR
        weekly = np.zeros(5)
        for person, row_code, amount in rows:
            if row_code == code:
                weekly[person] += amount
        np.testing.assert_allclose(
            reported[f"{benefit}_reported"] / weeks, weekly, rtol=1e-12, atol=1e-9
        )


@pytest.mark.parametrize("year", range(DATA_YEAR, 2030))
def test_full_rate_pensioners_have_no_additional_state_pension(year):
    types = ["NEW", "BASIC"]
    sim = Microsimulation(
        dataset=state_pension_dataset([FULL_RATE[t] for t in types], types)
    )
    c = components(sim, year)

    assert c["type"].tolist() == types
    assert np.all(np.abs(c["additional_state_pension"]) < PENNY_A_YEAR)
    full = sim.tax_benefit_system.parameters.gov.dwp.state_pension
    np.testing.assert_allclose(
        c["new_state_pension"][0], 52 * full.new_state_pension.amount(year), atol=0.01
    )
    np.testing.assert_allclose(
        c["basic_state_pension"][1],
        52 * full.basic_state_pension.amount(year),
        atol=0.01,
    )


def assert_split_at_the_full_rate(weekly, types):
    sim = Microsimulation(dataset=state_pension_dataset(weekly, types))
    c = components(sim, DATA_YEAR)
    full = np.array([FULL_RATE[t] for t in types])
    flat = c["new_state_pension"] + c["basic_state_pension"]

    assert c["type"].tolist() == list(types)
    np.testing.assert_allclose(
        c["additional_state_pension"],
        52 * np.maximum(weekly - full, 0),
        atol=PENNY_A_YEAR,
    )
    np.testing.assert_allclose(flat, 52 * np.minimum(weekly, full), atol=0.01)
    np.testing.assert_allclose(c["state_pension"], 52 * weekly, atol=0.01)
    at_or_below = weekly <= full
    assert np.all(c["additional_state_pension"][at_or_below] < PENNY_A_YEAR)


def test_every_weekly_amount_within_3_pounds_of_the_full_rates_splits_there():
    """Every whole-penny weekly amount from £3 below to £3 above each full
    rate, the resolution the FRS records BENAMT at."""
    offsets = np.arange(-300, 301) / 100
    types = ["NEW"] * len(offsets) + ["BASIC"] * len(offsets)
    weekly = np.round(
        np.r_[FULL_RATE["NEW"] + offsets, FULL_RATE["BASIC"] + offsets], 2
    )
    assert_split_at_the_full_rate(weekly, types)


near_full_rate = st.tuples(
    st.sampled_from(["NEW", "BASIC"]),
    st.floats(min_value=-20, max_value=20, allow_nan=False),
)


@settings(
    max_examples=15,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(cases=st.lists(near_full_rate, min_size=1, max_size=40))
def test_any_weekly_amount_near_the_full_rates_splits_there(cases):
    types = [t for t, _ in cases]
    weekly = np.array([FULL_RATE[t] + offset for t, offset in cases])
    assert_split_at_the_full_rate(weekly, types)
