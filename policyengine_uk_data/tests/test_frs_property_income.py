import numpy as np
import pandas as pd
import pytest

from policyengine_uk_data.datasets.frs import WEEKS_IN_YEAR, frs_property_income

HRP, NOT_HRP = 1, 2
OWNED_WITH_MORTGAGE, OWNED_OUTRIGHT = 5, 6
COUNCIL_RENTED, PRIVATE_RENTED_FURNISHED = 1, 4


def make_tables(people, households):
    person = pd.DataFrame(
        people, columns=["household_id", "person_id", "hrpid", "royyr1", "cvpay"]
    )
    household = pd.DataFrame(
        households, columns=["household_id", "tentyp2", "subrent"]
    ).set_index("household_id")
    return person, household


def test_rent_paid_by_a_lodger_is_not_their_property_income():
    # Owner-occupier household with a lodger who pays £100 a week (CVPAY).
    person, household = make_tables(
        [(1, 1_001, HRP, 0, 0), (1, 1_002, NOT_HRP, 0, 100)],
        [(1, OWNED_OUTRIGHT, 0)],
    )
    assert frs_property_income(person, household).tolist() == [0, 0]


def test_rent_from_other_property_counts_for_any_adult():
    person, household = make_tables(
        [(1, 1_001, HRP, 0, 0), (1, 1_002, NOT_HRP, 50, 0)],
        [(1, COUNCIL_RENTED, 0)],
    )
    np.testing.assert_allclose(
        frs_property_income(person, household), [0, 50 * WEEKS_IN_YEAR]
    )


def test_subletting_rent_goes_to_the_owner_household_reference_person():
    person, household = make_tables(
        [(1, 1_001, NOT_HRP, 0, 0), (1, 1_002, HRP, 0, 0)],
        [(1, OWNED_WITH_MORTGAGE, 80)],
    )
    np.testing.assert_allclose(
        frs_property_income(person, household), [0, 80 * WEEKS_IN_YEAR]
    )


def random_tables(seed: int):
    """Random households of one to four adults; the first is the HRP."""
    rng = np.random.default_rng(seed)
    people, households = [], []
    for household_id in range(1, rng.integers(1, 30) + 1):
        tenure = int(rng.integers(1, 9))
        subrent = float(rng.choice([0, rng.uniform(0, 500)]))
        households.append((household_id, tenure, subrent))
        for person in range(1, rng.integers(1, 5) + 1):
            people.append(
                (
                    household_id,
                    household_id * 1_000 + person,
                    HRP if person == 1 else NOT_HRP,
                    float(rng.choice([0, rng.uniform(0, 2_000)])),
                    float(rng.choice([0, rng.uniform(0, 400)])),
                )
            )
    return make_tables(people, households)


SEEDS = range(200)


@pytest.mark.parametrize("seed", SEEDS)
def test_property_income_does_not_depend_on_cvpay(seed):
    person, household = random_tables(seed)
    without_cvpay = person.assign(cvpay=0.0)
    np.testing.assert_array_equal(
        frs_property_income(person, household),
        frs_property_income(without_cvpay, household),
    )


@pytest.mark.parametrize("seed", SEEDS)
def test_property_income_conserves_reported_rent(seed):
    # Each owner household's SUBRENT is counted once (on its HRP) and every
    # ROYYR1 is counted on its own record, so the totals match.
    person, household = random_tables(seed)
    result = frs_property_income(person, household)
    assert (result >= 0).all()
    owner = household.tentyp2.isin((OWNED_WITH_MORTGAGE, OWNED_OUTRIGHT))
    expected = (household.subrent[owner].sum() + person.royyr1.sum()) * WEEKS_IN_YEAR
    assert result.sum() == pytest.approx(expected)


@pytest.mark.parametrize("seed", SEEDS)
def test_extra_rent_from_other_property_moves_only_that_person(seed):
    person, household = random_tables(seed)
    before = frs_property_income(person, household)
    row = seed % len(person)
    person.loc[row, "royyr1"] += 10
    after = frs_property_income(person, household)
    change = np.zeros(len(person))
    change[row] = 10 * WEEKS_IN_YEAR
    np.testing.assert_allclose(after - before, change)
