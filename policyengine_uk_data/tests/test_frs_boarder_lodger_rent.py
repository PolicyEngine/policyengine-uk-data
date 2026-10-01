import numpy as np
import pandas as pd
import pytest

from policyengine_uk_data.datasets.frs import (
    WEEKS_IN_YEAR,
    frs_boarder_and_lodger_rent,
    frs_property_income,
)

HRP, NOT_HRP = 1, 2
BOARDER, LODGER, NEITHER, NOT_ASKED = 1, 2, 3, 0
OWNED_OUTRIGHT = 6


def make_person(people):
    return pd.DataFrame(people, columns=["person_id", "convbl", "cvpay"])


def test_a_boarder_pays_rent_as_a_boarder():
    boarder, lodger = frs_boarder_and_lodger_rent(
        make_person([(1_001, NOT_ASKED, 0), (1_002, BOARDER, 120)])
    )
    np.testing.assert_allclose(boarder, [0, 120 * WEEKS_IN_YEAR])
    assert lodger.tolist() == [0, 0]


def test_a_lodger_pays_rent_as_a_lodger():
    boarder, lodger = frs_boarder_and_lodger_rent(
        make_person([(1_001, NOT_ASKED, 0), (1_002, LODGER, 100)])
    )
    assert boarder.tolist() == [0, 0]
    np.testing.assert_allclose(lodger, [0, 100 * WEEKS_IN_YEAR])


@pytest.mark.parametrize("convbl", [NEITHER, NOT_ASKED, -1, np.nan])
def test_rent_without_a_boarder_code_is_a_lodgers(convbl):
    boarder, lodger = frs_boarder_and_lodger_rent(make_person([(1_001, convbl, 90)]))
    assert boarder.tolist() == [0]
    np.testing.assert_allclose(lodger, [90 * WEEKS_IN_YEAR])


@pytest.mark.parametrize("cvpay", [0, -1, -30, np.nan])
@pytest.mark.parametrize("convbl", [BOARDER, LODGER, NEITHER])
def test_no_positive_amount_means_no_rent(convbl, cvpay):
    boarder, lodger = frs_boarder_and_lodger_rent(make_person([(1_001, convbl, cvpay)]))
    assert boarder.tolist() == [0]
    assert lodger.tolist() == [0]


def test_adult_and_child_rows_sharing_index_labels():
    # create_frs stacks the adult and child tables and fills the gaps with
    # zero, so index labels repeat and child rows carry CVPAY 0 and CONVBL 0.
    adults = make_person([(1_001, NOT_ASKED, 0), (1_002, BOARDER, 70)])
    children = pd.DataFrame({"person_id": [1_003]})
    person = pd.concat([adults, children]).sort_index(kind="stable").fillna(0)
    assert person.index.tolist() == [0, 0, 1]
    boarder, lodger = frs_boarder_and_lodger_rent(person)
    np.testing.assert_allclose(boarder, [0, 0, 70 * WEEKS_IN_YEAR])
    assert lodger.tolist() == [0, 0, 0]


def random_tables(seed: int):
    """Random households of one to four adults; the first is the HRP."""
    rng = np.random.default_rng(seed)
    people, households = [], []
    for household_id in range(1, rng.integers(1, 30) + 1):
        households.append((household_id, int(rng.integers(1, 9)), 0.0))
        for person in range(1, rng.integers(1, 5) + 1):
            people.append(
                (
                    household_id,
                    household_id * 1_000 + person,
                    HRP if person == 1 else NOT_HRP,
                    float(rng.choice([0, rng.uniform(0, 2_000)])),
                    rng.choice([BOARDER, LODGER, NEITHER, NOT_ASKED, -1, np.nan]),
                    rng.choice([0, rng.uniform(0, 400), -1, np.nan]),
                )
            )
    person = pd.DataFrame(
        people,
        columns=["household_id", "person_id", "hrpid", "royyr1", "convbl", "cvpay"],
    )
    household = pd.DataFrame(
        households, columns=["household_id", "tentyp2", "subrent"]
    ).set_index("household_id")
    return person, household


SEEDS = range(200)


@pytest.mark.parametrize("seed", SEEDS)
def test_boarder_and_lodger_rent_conserves_reported_rent(seed):
    # Every positive CVPAY is counted once, in exactly one of the two classes.
    person, _ = random_tables(seed)
    boarder, lodger = frs_boarder_and_lodger_rent(person)
    assert (boarder >= 0).all() and (lodger >= 0).all()
    assert not ((boarder > 0) & (lodger > 0)).any()
    paid = person.cvpay.where(person.cvpay > 0, 0)
    np.testing.assert_allclose(boarder + lodger, paid * WEEKS_IN_YEAR)
    np.testing.assert_allclose(
        boarder.sum(), paid[person.convbl == BOARDER].sum() * WEEKS_IN_YEAR
    )


@pytest.mark.parametrize("seed", SEEDS)
def test_extra_rent_moves_only_the_person_who_pays_it(seed):
    person, _ = random_tables(seed)
    person["cvpay"] = person.cvpay.where(person.cvpay > 0, 0)
    before = sum(frs_boarder_and_lodger_rent(person))
    row = seed % len(person)
    person.loc[row, "cvpay"] += 10
    after = sum(frs_boarder_and_lodger_rent(person))
    change = np.zeros(len(person))
    change[row] = 10 * WEEKS_IN_YEAR
    np.testing.assert_allclose(after - before, change)


@pytest.mark.parametrize("seed", SEEDS)
def test_rent_paid_and_property_income_do_not_affect_each_other(seed):
    person, household = random_tables(seed)
    property_income = frs_property_income(person.fillna(0), household)
    rent_paid = frs_boarder_and_lodger_rent(person)
    np.testing.assert_array_equal(
        frs_property_income(
            person.fillna(0).assign(cvpay=0.0, convbl=NOT_ASKED), household
        ),
        property_income,
    )
    other_property = person.assign(royyr1=person.royyr1 + 50)
    for changed, original in zip(
        frs_boarder_and_lodger_rent(other_property), rent_paid
    ):
        np.testing.assert_array_equal(changed, original)
