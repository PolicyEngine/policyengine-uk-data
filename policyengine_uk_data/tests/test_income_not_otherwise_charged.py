"""FRS odd-job income is the income not otherwise charged.

policyengine-uk's trading allowance relieves income chargeable under ITTOIA
2005 Part 5 Chapter 8 (s. 783AB), which it reads from
``income_not_otherwise_charged``, the part of ``miscellaneous_income`` that
Chapter 8 charges. The build sets it to the FRS odd-job income in
``miscellaneous_income``.

Invariants, for any person table and ODDJOB table (amounts including FRS
missing-value codes and blanks):

1. Both outputs are non-negative, and the income not otherwise charged never
   exceeds miscellaneous income.
2. Conservation: miscellaneous income less the income not otherwise charged
   is the annualised floored sum of the six other fields, and the income not
   otherwise charged is the annualised odd-job income from jobs in the last
   7 days.
3. Raising an odd-job amount never lowers either output; the six other fields
   never change the income not otherwise charged.
4. Differential: miscellaneous income is what the build computed before this
   change whenever no odd-job amount is a missing-value code.

And for the built datasets:

5. The FRS and the enhanced FRS keep invariant 1 on every row.
6. SPI-donor rows of the enhanced FRS hold none (their trading profits are
   SPI PROFITS, which may already have used the allowance).
7. The uprating table moves it exactly as miscellaneous income, so the bound
   holds in every year, once policyengine-uk reads it.

The data checks assert on counts only, so a failure never prints a record's
amounts (the FRS is licensed microdata).
"""

import numpy as np
import pandas as pd
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from policyengine_uk.data import UKSingleYearDataset

from policyengine_uk_data.datasets.frs import (
    MISC_INCOME_FIELDS,
    WEEKS_IN_YEAR,
    frs_miscellaneous_income,
)
from policyengine_uk_data.datasets.imputations.income import (
    clear_income_not_otherwise_charged,
)
from policyengine_uk_data.storage import STORAGE_FOLDER
from policyengine_uk_data.utils.datasets import (
    sum_from_positive_fields,
    sum_to_entity,
)

VARIABLE = "income_not_otherwise_charged"
MISSING_CODES = st.sampled_from([-9.0, -8.0, -7.0, -1.0])
weekly_amounts = st.one_of(
    st.floats(0, 5_000, allow_nan=False, allow_infinity=False),
    st.just(0.0),
    MISSING_CODES,
)


@st.composite
def frs_tables(draw, missing_odd_job_amounts=True):
    """A person table with the six fields and an ODDJOB table for it."""
    n_people = draw(st.integers(1, 8))
    person_ids = np.arange(n_people) * 7 + 1001
    person = pd.DataFrame({"person_id": person_ids})
    for field in MISC_INCOME_FIELDS:
        person[field] = draw(
            st.lists(weekly_amounts, min_size=n_people, max_size=n_people)
        )
    n_jobs = draw(st.integers(0, 12))
    amounts = st.one_of(
        st.floats(0, 5_000, allow_nan=False, allow_infinity=False),
        st.just(np.nan),
        *([MISSING_CODES] if missing_odd_job_amounts else []),
    )
    oddjob = pd.DataFrame(
        {
            "person_id": draw(
                st.lists(
                    st.sampled_from(list(person_ids)),
                    min_size=n_jobs,
                    max_size=n_jobs,
                )
            ),
            "ojamt": draw(st.lists(amounts, min_size=n_jobs, max_size=n_jobs)),
            "ojnow": draw(
                st.lists(
                    st.sampled_from([1.0, 2.0, -1.0, np.nan]),
                    min_size=n_jobs,
                    max_size=n_jobs,
                )
            ),
        },
        dtype=float,
    )
    return person, oddjob


@settings(max_examples=300, deadline=None, derandomize=True)
@given(frs_tables())
def test_non_negative_and_within_miscellaneous_income(tables):
    person, oddjob = tables
    miscellaneous, chapter_8 = frs_miscellaneous_income(person, oddjob)
    assert np.all(chapter_8 >= 0)
    assert np.all(miscellaneous >= 0)
    assert np.all(chapter_8 <= miscellaneous)


@settings(max_examples=300, deadline=None, derandomize=True)
@given(frs_tables())
def test_conserves_odd_job_and_other_income(tables):
    person, oddjob = tables
    miscellaneous, chapter_8 = frs_miscellaneous_income(person, oddjob)
    rest = sum_from_positive_fields(person, MISC_INCOME_FIELDS) * WEEKS_IN_YEAR
    np.testing.assert_allclose(miscellaneous - chapter_8, rest, rtol=1e-12, atol=1e-6)
    done_last_week = oddjob.ojnow == 1
    expected = [
        oddjob.ojamt[done_last_week & (oddjob.person_id == pid)].clip(lower=0).sum()
        * WEEKS_IN_YEAR
        for pid in person.person_id
    ]
    np.testing.assert_allclose(chapter_8, expected, rtol=1e-12, atol=1e-6)


@settings(max_examples=300, deadline=None, derandomize=True)
@given(
    frs_tables(),
    st.integers(0, 11),
    st.floats(0, 5_000, allow_nan=False),
    st.floats(-5_000, 5_000, allow_nan=False),
)
def test_monotone_in_odd_jobs_and_blind_to_other_fields(
    tables, job, rise, other_change
):
    person, oddjob = tables
    miscellaneous, chapter_8 = frs_miscellaneous_income(person, oddjob)

    if job < len(oddjob):
        raised = oddjob.copy()
        amount = raised.loc[job, "ojamt"]
        raised.loc[job, "ojamt"] = (0 if np.isnan(amount) else max(amount, 0)) + rise
        higher_miscellaneous, higher_chapter_8 = frs_miscellaneous_income(
            person, raised
        )
        assert np.all(higher_chapter_8 >= chapter_8)
        assert np.all(higher_miscellaneous >= miscellaneous - 1e-6)

    changed = person.copy()
    changed[MISC_INCOME_FIELDS[0]] += other_change
    _, unchanged_chapter_8 = frs_miscellaneous_income(changed, oddjob)
    np.testing.assert_array_equal(unchanged_chapter_8, chapter_8)


def _miscellaneous_income_before_this_change(person, oddjob):
    """The build's miscellaneous income at policyengine-uk-data 1.59.1."""
    odd_job_income = sum_to_entity(
        oddjob.ojamt * (oddjob.ojnow == 1), oddjob.person_id, person.person_id
    )
    return (
        odd_job_income + sum_from_positive_fields(person, MISC_INCOME_FIELDS)
    ) * WEEKS_IN_YEAR


@settings(max_examples=300, deadline=None, derandomize=True)
@given(frs_tables(missing_odd_job_amounts=False))
def test_miscellaneous_income_is_unchanged(tables):
    person, oddjob = tables
    assume((oddjob.ojamt.fillna(0) >= 0).all())
    miscellaneous, _ = frs_miscellaneous_income(person, oddjob)
    np.testing.assert_array_equal(
        miscellaneous, _miscellaneous_income_before_this_change(person, oddjob)
    )


def _unit_dataset(chapter_8=None) -> UKSingleYearDataset:
    person = pd.DataFrame(
        {
            "person_id": [1, 2],
            "person_benunit_id": [1, 1],
            "person_household_id": [1, 1],
            "miscellaneous_income": [900.0, 50.0],
        }
    )
    if chapter_8 is not None:
        person[VARIABLE] = chapter_8
    return UKSingleYearDataset(
        person=person,
        benunit=pd.DataFrame({"benunit_id": [1]}),
        household=pd.DataFrame({"household_id": [1]}),
        fiscal_year=2024,
    )


def test_clear_zeroes_only_the_income_not_otherwise_charged():
    cleared = clear_income_not_otherwise_charged(_unit_dataset([800.0, 50.0]))
    assert cleared.person[VARIABLE].tolist() == [0.0, 0.0]
    assert cleared.person.miscellaneous_income.tolist() == [900.0, 50.0]


def test_clear_adds_no_column_to_a_dataset_without_one():
    cleared = clear_income_not_otherwise_charged(_unit_dataset())
    assert VARIABLE not in cleared.person.columns


def _bound_violations(person, mask=None) -> dict:
    chapter_8 = person[VARIABLE].to_numpy()
    miscellaneous = person["miscellaneous_income"].to_numpy()
    keep = np.ones(len(person), dtype=bool) if mask is None else mask
    return {
        "negative": int((chapter_8[keep] < 0).sum()),
        "above_miscellaneous_income": int((chapter_8 > miscellaneous)[keep].sum()),
        "not_finite": int((~np.isfinite(chapter_8[keep])).sum()),
    }


NO_VIOLATIONS = {"negative": 0, "above_miscellaneous_income": 0, "not_finite": 0}


def test_frs_within_miscellaneous_income(frs):
    if VARIABLE not in frs.person.columns:
        pytest.skip(f"Dataset built before {VARIABLE} was added")
    assert _bound_violations(frs.person) == NO_VIOLATIONS
    assert int((frs.person[VARIABLE] > 0).sum()) > 0, "expected FRS odd-job income"


def test_enhanced_frs_within_bound_and_none_on_spi_donor_rows(enhanced_frs):
    person = enhanced_frs.person
    if VARIABLE not in person.columns:
        pytest.skip(f"Dataset built before {VARIABLE} was added")
    household = enhanced_frs.household
    synthetic_households = household.household_id[
        household.household_is_spi_synthetic.astype(bool)
    ]
    synthetic = person.person_household_id.isin(synthetic_households).to_numpy()
    assert int(synthetic.sum()) > 0, "expected SPI-donor rows in the enhanced FRS"
    assert _bound_violations(person) == NO_VIOLATIONS
    assert int((person[VARIABLE].to_numpy()[synthetic] != 0).sum()) == 0
    assert int((person[VARIABLE].to_numpy()[~synthetic] > 0).sum()) > 0


def test_uprated_exactly_as_miscellaneous_income():
    from policyengine_uk.system import system

    if VARIABLE not in system.variables:
        pytest.skip(
            "The locked policyengine-uk does not read "
            f"{VARIABLE} (PolicyEngine/policyengine-uk#2726)"
        )
    table = pd.read_csv(STORAGE_FOLDER / "uprating_factors.csv", index_col="Variable")
    assert VARIABLE in table.index, "regenerate uprating_factors.csv"
    pd.testing.assert_series_equal(
        table.loc[VARIABLE], table.loc["miscellaneous_income"], check_names=False
    )
