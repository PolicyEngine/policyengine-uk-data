import numpy as np
import pandas as pd
import pytest

from policyengine_uk_data.datasets.frs import (
    FRS_RENTPROF_LOSS,
    WEEKS_IN_YEAR,
    frs_property_income,
)
from policyengine_uk_data.datasets.frs_release import CURRENT_FRS_RELEASE
from policyengine_uk_data.storage import STORAGE_FOLDER

HRP, NOT_HRP = 1, 2
NOT_ASKED, PROFIT, LOSS = 0, 1, FRS_RENTPROF_LOSS
# FRS missing-value codes run from -1 to -9.
MISSING, REFUSED = -1, -9
OWNED_WITH_MORTGAGE, OWNED_OUTRIGHT = 5, 6
COUNCIL_RENTED, PRIVATE_RENTED_FURNISHED = 1, 4
TENURES = range(1, 9)
BEFORE_EXPENSES, AFTER_EXPENSES = 1, 2


def make_tables(people, households):
    person = pd.DataFrame(
        people,
        columns=["household_id", "person_id", "hrpid", "royyr1", "rentprof", "cvpay"],
    )
    household = pd.DataFrame(
        households, columns=["household_id", "tentyp2", "subrent", "suballow"]
    ).set_index("household_id")
    return person, household


def test_rent_paid_by_a_lodger_is_not_their_property_income():
    # Owner-occupier household with a lodger who pays £100 a week (CVPAY).
    person, household = make_tables(
        [(1, 1_001, HRP, 0, NOT_ASKED, 0), (1, 1_002, NOT_HRP, 0, NOT_ASKED, 100)],
        [(1, OWNED_OUTRIGHT, 0, 0)],
    )
    assert frs_property_income(person, household).tolist() == [0, 0]


def test_rent_from_other_property_counts_for_any_adult():
    person, household = make_tables(
        [(1, 1_001, HRP, 0, NOT_ASKED, 0), (1, 1_002, NOT_HRP, 50, PROFIT, 0)],
        [(1, COUNCIL_RENTED, 0, 0)],
    )
    np.testing.assert_allclose(
        frs_property_income(person, household), [0, 50 * WEEKS_IN_YEAR]
    )


def test_a_loss_on_other_property_is_not_income():
    # ROYYR1 holds the size of the loss as a positive amount; RENTPROF flags it.
    person, household = make_tables(
        [(1, 1_001, HRP, 90, LOSS, 0), (1, 1_002, NOT_HRP, 50, PROFIT, 0)],
        [(1, OWNED_WITH_MORTGAGE, 0, 0)],
    )
    np.testing.assert_allclose(
        frs_property_income(person, household), [0, 50 * WEEKS_IN_YEAR]
    )


def test_a_loss_on_other_property_does_not_reduce_subletting_rent():
    person, household = make_tables(
        [(1, 1_001, HRP, 30, LOSS, 0)],
        [(1, OWNED_OUTRIGHT, 80, AFTER_EXPENSES)],
    )
    np.testing.assert_allclose(
        frs_property_income(person, household), [80 * WEEKS_IN_YEAR]
    )


def test_rent_with_no_profit_or_loss_answer_still_counts():
    # Only an explicit loss removes the amount.
    person, household = make_tables(
        [(1, 1_001, HRP, 50, NOT_ASKED, 0)], [(1, OWNED_OUTRIGHT, 0, 0)]
    )
    np.testing.assert_allclose(
        frs_property_income(person, household), [50 * WEEKS_IN_YEAR]
    )


@pytest.mark.parametrize("tenure", TENURES)
def test_subletting_rent_goes_to_the_household_reference_person(tenure):
    # Every tenure is asked SubLet, so renting households count too.
    person, household = make_tables(
        [(1, 1_001, NOT_HRP, 0, NOT_ASKED, 0), (1, 1_002, HRP, 0, NOT_ASKED, 0)],
        [(1, tenure, 80, BEFORE_EXPENSES)],
    )
    np.testing.assert_allclose(
        frs_property_income(person, household), [0, 80 * WEEKS_IN_YEAR]
    )


@pytest.mark.parametrize("suballow", [0, BEFORE_EXPENSES, AFTER_EXPENSES])
def test_subletting_rent_is_used_as_reported_whatever_its_expenses_basis(suballow):
    person, household = make_tables(
        [(1, 1_001, HRP, 0, NOT_ASKED, 0)],
        [(1, PRIVATE_RENTED_FURNISHED, 80, suballow)],
    )
    np.testing.assert_allclose(
        frs_property_income(person, household), [80 * WEEKS_IN_YEAR]
    )


def test_missing_value_codes_count_as_zero():
    # A missing code in one amount neither becomes income nor cancels the other.
    person, household = make_tables(
        [
            (1, 1_001, HRP, MISSING, MISSING, 0),
            (1, 1_002, NOT_HRP, REFUSED, MISSING, 0),
            (2, 2_001, HRP, 50, PROFIT, 0),
        ],
        [(1, COUNCIL_RENTED, 10, AFTER_EXPENSES), (2, OWNED_OUTRIGHT, MISSING, 0)],
    )
    np.testing.assert_allclose(
        frs_property_income(person, household),
        [10 * WEEKS_IN_YEAR, 0, 50 * WEEKS_IN_YEAR],
    )


def test_adult_and_child_rows_sharing_index_labels():
    # create_frs stacks the adult and child tables, so index labels repeat.
    adults, household = make_tables(
        [(1, 1_001, HRP, 20, PROFIT, 0), (2, 2_001, HRP, 40, LOSS, 60)],
        [
            (1, OWNED_OUTRIGHT, 80, BEFORE_EXPENSES),
            (2, PRIVATE_RENTED_FURNISHED, 30, AFTER_EXPENSES),
        ],
    )
    children, _ = make_tables([(1, 1_002, 0, 0, NOT_ASKED, 0)], [])
    person = pd.concat([adults, children]).sort_index(kind="stable")
    assert person.index.tolist() == [0, 0, 1]
    np.testing.assert_allclose(
        frs_property_income(person, household),
        [100 * WEEKS_IN_YEAR, 0, 30 * WEEKS_IN_YEAR],
    )


def random_tables(seed: int):
    """Random households of one to four adults; the first is the HRP."""
    rng = np.random.default_rng(seed)
    people, households = [], []
    for household_id in range(1, rng.integers(1, 30) + 1):
        tenure = int(rng.integers(1, 9))
        subrent = float(rng.choice([0, MISSING, rng.uniform(0, 500)]))
        suballow = int(rng.integers(1, 3)) if subrent > 0 else 0
        households.append((household_id, tenure, subrent, suballow))
        for person in range(1, rng.integers(1, 5) + 1):
            royyr1 = float(rng.choice([0, MISSING, rng.uniform(0, 2_000)]))
            if royyr1 > 0:
                rentprof = int(rng.choice([PROFIT, PROFIT, LOSS]))
            else:
                rentprof = MISSING if royyr1 < 0 else NOT_ASKED
            people.append(
                (
                    household_id,
                    household_id * 1_000 + person,
                    HRP if person == 1 else NOT_HRP,
                    royyr1,
                    rentprof,
                    float(rng.choice([0, rng.uniform(0, 400)])),
                )
            )
    return make_tables(people, households)


def property_income_one_person_at_a_time(person, household):
    """The same rules written as a loop, to check the vectorised helper."""
    result = []
    for row in person.itertuples():
        weekly = 0.0
        if row.rentprof != LOSS:
            weekly += max(0.0, row.royyr1)
        if row.hrpid == HRP and row.household_id in household.index:
            weekly += max(0.0, household.subrent[row.household_id])
        result.append(weekly * WEEKS_IN_YEAR)
    return np.array(result)


SEEDS = range(200)


@pytest.mark.parametrize("seed", SEEDS)
def test_property_income_matches_the_loop_version(seed):
    person, household = random_tables(seed)
    np.testing.assert_allclose(
        frs_property_income(person, household),
        property_income_one_person_at_a_time(person, household),
    )


@pytest.mark.parametrize("seed", SEEDS)
def test_property_income_does_not_depend_on_cvpay_tenure_or_expenses_basis(seed):
    person, household = random_tables(seed)
    rng = np.random.default_rng(seed)
    other_person = person.assign(cvpay=rng.uniform(0, 400, len(person)))
    other_household = household.assign(
        tentyp2=rng.integers(1, 9, len(household)),
        suballow=rng.integers(1, 3, len(household)),
    )
    np.testing.assert_array_equal(
        frs_property_income(person, household),
        frs_property_income(other_person, other_household),
    )


@pytest.mark.parametrize("seed", SEEDS)
def test_property_income_conserves_reported_rent(seed):
    # Each household's SUBRENT is counted once (on its HRP) and every ROYYR1
    # that is not a loss is counted on its own record, so the totals match.
    person, household = random_tables(seed)
    result = frs_property_income(person, household)
    assert (result >= 0).all()
    profits = person.royyr1.clip(lower=0)[person.rentprof != LOSS]
    subrent = household.subrent.clip(lower=0)
    expected = (subrent.sum() + profits.sum()) * WEEKS_IN_YEAR
    assert result.sum() == pytest.approx(expected)


@pytest.mark.parametrize("seed", SEEDS)
def test_marking_rent_as_a_loss_removes_it_from_that_person_only(seed):
    person, household = random_tables(seed)
    before = frs_property_income(person, household)
    row = seed % len(person)
    was_counted = person.loc[row, "rentprof"] != LOSS
    person.loc[row, "rentprof"] = LOSS
    after = frs_property_income(person, household)
    change = np.zeros(len(person))
    change[row] = -max(0, person.loc[row, "royyr1"]) * WEEKS_IN_YEAR * was_counted
    np.testing.assert_allclose(after - before, change, atol=1e-6)


@pytest.mark.parametrize("seed", SEEDS)
def test_extra_rent_from_other_property_moves_only_that_person(seed):
    person, household = random_tables(seed)
    before = frs_property_income(person, household)
    row = seed % len(person)
    person.loc[row, "royyr1"] = max(0, person.loc[row, "royyr1"]) + 10
    after = frs_property_income(person, household)
    change = np.zeros(len(person))
    change[row] = 10 * WEEKS_IN_YEAR * (person.loc[row, "rentprof"] != LOSS)
    np.testing.assert_allclose(after - before, change, atol=1e-6)


@pytest.mark.parametrize("seed", SEEDS)
def test_extra_subletting_rent_moves_only_that_households_reference_person(seed):
    person, household = random_tables(seed)
    before = frs_property_income(person, household)
    household_id = household.index[seed % len(household)]
    household.loc[household_id, "subrent"] = (
        max(0, household.loc[household_id, "subrent"]) + 10
    )
    after = frs_property_income(person, household)
    is_its_hrp = (person.household_id == household_id) & (person.hrpid == HRP)
    np.testing.assert_allclose(
        after - before, is_its_hrp * 10 * WEEKS_IN_YEAR, atol=1e-6
    )


def test_built_frs_matches_the_raw_tables(frs):
    """The built base FRS against a merge-based reading of the raw tables.

    Covers the call site in ``create_frs`` and the raw column names and codes.
    The assertion carries no values, because the dataset is licensed microdata.
    """
    raw_folder = STORAGE_FOLDER / CURRENT_FRS_RELEASE.name
    if not (raw_folder / "adult.tab").exists():
        pytest.skip("Raw FRS tables not available")
    adult = pd.read_csv(
        raw_folder / "adult.tab",
        sep="\t",
        usecols=lambda c: (
            c.upper() in ("SERNUM", "PERSON", "HRPID", "ROYYR1", "RENTPROF")
        ),
    )
    househol = pd.read_csv(
        raw_folder / "househol.tab",
        sep="\t",
        usecols=lambda c: c.upper() in ("SERNUM", "SUBRENT"),
    )
    adult.columns = adult.columns.str.upper()
    househol.columns = househol.columns.str.upper()
    raw = adult.merge(househol, on="SERNUM", how="left").apply(
        pd.to_numeric, errors="coerce"
    )
    profit = raw.ROYYR1.clip(lower=0).where(raw.RENTPROF != 2, 0).fillna(0)
    subrent = raw.SUBRENT.clip(lower=0).where(raw.HRPID == 1, 0).fillna(0)
    expected = pd.Series(
        ((profit + subrent) * WEEKS_IN_YEAR).values,
        index=(raw.SERNUM * 1_000 + raw.PERSON).astype(int),
    )
    built = pd.Series(
        frs.person.property_income.values, index=frs.person.person_id.values
    )
    adults_match = np.allclose(built.reindex(expected.index), expected, atol=0.01)
    children_have_none = (built.drop(expected.index) == 0).all()
    assert adults_match and children_have_none, (
        "property_income in the built FRS differs from the raw FRS tables"
    )
