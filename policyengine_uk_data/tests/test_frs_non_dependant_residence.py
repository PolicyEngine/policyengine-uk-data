import numpy as np
import pandas as pd
from hypothesis import given, settings, strategies as st

from policyengine_uk_data.datasets.frs import frs_non_dependant_normally_resides_with

EVERY, HEAD, OTHERS = (
    "EVERY_JOINT_OCCUPIER",
    "HOUSEHOLD_HEAD_FAMILY",
    "OTHER_JOINT_OCCUPIERS",
)
# FRS 2024-25 adult and child data dictionaries, R01-R14.
SISTER, SON, PARENT, GRANDCHILD, NON_RELATIVE = 11, 3, 7, 15, 18
RELATED = {*range(1, 18), 20}
GRID = [f"r{k:02d}" for k in range(1, 15)]


def frames(households):
    """Build FRS-style benunit and person frames.

    ``households`` maps a household id to a list of benefit units, each a
    list of people as dicts with optional ``hrp`` and ``codes`` (person
    number in the household -> relationship code). People are numbered 1, 2,
    ... across the household in the order given.
    """
    benunits, people = [], []
    for household_id, units in households.items():
        number = 0
        for unit, members in enumerate(units, start=1):
            benunit_id = household_id * 100 + unit
            benunits.append((benunit_id, household_id))
            for member in members:
                number += 1
                row = {
                    "person_id": household_id * 1000 + number,
                    "benunit_id": benunit_id,
                    "household_id": household_id,
                    "hrpid": int(member.get("hrp", False)),
                }
                row.update({column: 0 for column in GRID})
                for other, code in member.get("codes", {}).items():
                    if other <= 14:
                        row[f"r{other:02d}"] = code
                people.append(row)
    benunit = pd.DataFrame(benunits, columns=["benunit_id", "household_id"])
    return benunit, pd.DataFrame(people)


def resides(households, liable):
    benunit, person = frames(households)
    return frs_non_dependant_normally_resides_with(
        benunit, person, np.array(liable)
    ).tolist()


def test_a_joint_tenants_sister_resides_with_that_tenant_only():
    # LHA Guidance Manual 2.093, example 2: Sarah (the reference person) and
    # Rachel are joint tenants; Susan, Sarah's sister, is Sarah's non-dependant.
    sarah = {"hrp": True, "codes": {2: NON_RELATIVE, 3: SISTER}}
    rachel = {"codes": {1: NON_RELATIVE, 3: NON_RELATIVE}}
    susan = {"codes": {1: SISTER, 2: NON_RELATIVE}}
    assert resides({1: [[sarah], [rachel], [susan]]}, [False, True, False]) == [
        EVERY,
        EVERY,
        HEAD,
    ]


def test_a_sharers_sister_resides_with_the_sharer_only():
    head = {"hrp": True, "codes": {2: NON_RELATIVE, 3: NON_RELATIVE}}
    sharer = {"codes": {1: NON_RELATIVE, 3: SISTER}}
    sister = {"codes": {1: NON_RELATIVE, 2: SISTER}}
    assert resides({1: [[head], [sharer], [sister]]}, [False, True, False]) == [
        EVERY,
        EVERY,
        OTHERS,
    ]


def test_a_friend_of_the_joint_tenants_is_shared():
    # LHA Guidance Manual 2.110: Peter, a friend of joint tenants John and
    # Alex, is counted in each one's size criteria.
    john = {"hrp": True, "codes": {2: NON_RELATIVE, 3: NON_RELATIVE}}
    alex = {"codes": {1: NON_RELATIVE, 3: NON_RELATIVE}}
    peter = {"codes": {1: NON_RELATIVE, 2: NON_RELATIVE}}
    assert resides({1: [[john], [alex], [peter]]}, [False, True, False]) == [EVERY] * 3


def test_a_relative_of_one_sharer_among_several_keeps_the_default():
    # OTHER_JOINT_OCCUPIERS would mean both sharers, so it cannot be used.
    head = {"hrp": True}
    sharer = {"codes": {4: SISTER}}
    other_sharer = {}
    sister = {"codes": {2: SISTER}}
    households = {1: [[head], [sharer], [other_sharer], [sister]]}
    assert resides(households, [False, True, True, False]) == [EVERY] * 4


def test_a_relative_of_two_joint_occupiers_is_shared():
    # The reference person and the sharer are siblings; the non-dependant is
    # one's son and so the other's nephew.
    head = {"hrp": True, "codes": {2: SISTER, 3: PARENT}}
    sharer = {"codes": {1: SISTER, 3: 17}}
    son = {"codes": {1: SON, 2: 17}}
    assert resides({1: [[head], [sharer], [son]]}, [False, True, False]) == [EVERY] * 3


def test_households_without_a_sharer_keep_the_default():
    head = {"hrp": True, "codes": {2: PARENT}}
    adult_son = {"codes": {1: SON}}
    assert resides({1: [[head], [adult_son]]}, [False, False]) == [EVERY] * 2


def test_a_tie_on_either_record_counts():
    # Only the joint tenant's record says they are related.
    sarah = {"hrp": True, "codes": {3: SISTER}}
    rachel = {}
    susan = {"codes": {1: NON_RELATIVE}}
    assert resides({1: [[sarah], [rachel], [susan]]}, [False, True, False]) == [
        EVERY,
        EVERY,
        HEAD,
    ]


def test_any_member_of_the_family_can_carry_the_tie():
    # A lodger-like friend whose child is the reference person's grandchild.
    head = {"hrp": True}
    sharer = {}
    friend = {"codes": {1: NON_RELATIVE, 2: NON_RELATIVE}}
    grandchild = {"codes": {1: GRANDCHILD, 2: NON_RELATIVE, 3: SON}}
    households = {1: [[head], [sharer], [friend, grandchild]]}
    assert resides(households, [False, True, False]) == [EVERY, EVERY, HEAD]


def test_codes_for_people_outside_the_household_are_ignored():
    head = {"hrp": True}
    sharer = {}
    visitor_code = {"codes": {9: SISTER}}
    households = {1: [[head], [sharer], [visitor_code]]}
    assert resides(households, [False, True, False]) == [EVERY] * 3


def test_households_do_not_leak():
    sarah = {"hrp": True, "codes": {3: SISTER}}
    susan = {"codes": {1: SISTER}}
    households = {
        1: [[sarah], [{}], [susan]],
        2: [[{"hrp": True}], [{}], [{}]],
    }
    assert resides(households, [False, True, False, False, True, False]) == [
        EVERY,
        EVERY,
        HEAD,
        EVERY,
        EVERY,
        EVERY,
    ]


# Properties over generated households.

CODES = st.sampled_from([0, -1, np.nan, *sorted(RELATED), NON_RELATIVE])


@st.composite
def survey(draw):
    """Households of up to five benefit units and 20 people, one reference
    person each, random liability flags and random (possibly one-sided)
    relationship codes."""
    households, liable = {}, []
    for household_id in range(1, draw(st.integers(1, 4)) + 1):
        sizes = draw(st.lists(st.integers(1, 4), min_size=1, max_size=5))
        size = sum(sizes)
        hrp = draw(st.integers(1, size))
        units, number = [], 0
        for unit_size in sizes:
            members = []
            for _ in range(unit_size):
                number += 1
                # Codes may point at absent people and beyond the grid's 14.
                others = [k for k in range(1, 17) if k != number]
                codes = draw(st.dictionaries(st.sampled_from(others), CODES))
                members.append({"hrp": number == hrp, "codes": codes})
            units.append(members)
        households[household_id] = units
        liable += draw(
            st.lists(st.booleans(), min_size=len(units), max_size=len(units))
        )
    return households, liable


def reference(households, liable):
    """The rule, one family at a time."""
    out, position = [], 0
    for units in households.values():
        numbered, number = [], 0
        for members in units:
            numbered.append([(number + i + 1, m) for i, m in enumerate(members)])
            number += len(members)
        head = [any(m.get("hrp") for _, m in u) for u in numbered]
        flags = liable[position : position + len(units)]
        position += len(units)
        sharer = [f and not h for f, h in zip(flags, head)]
        joint = [h or s for h, s in zip(head, sharer)]

        def related(a, b):
            for k, m in numbered[a]:
                for j, o in numbered[b]:
                    if m.get("codes", {}).get(j) in RELATED and j <= 14:
                        return True
                    if o.get("codes", {}).get(k) in RELATED and k <= 14:
                        return True
            return False

        for u in range(len(units)):
            value = EVERY
            if not joint[u] and any(sharer):
                tied = [v for v in range(len(units)) if joint[v] and related(u, v)]
                if len(tied) == 1 and head[tied[0]]:
                    value = HEAD
                elif len(tied) == 1 and sum(sharer) == 1:
                    value = OTHERS
            out.append(value)
    return out


@settings(max_examples=300, deadline=None)
@given(survey())
def test_matches_the_rule_applied_family_by_family(case):
    households, liable = case
    assert resides(households, liable) == reference(households, liable)


@settings(max_examples=200, deadline=None)
@given(survey())
def test_only_non_dependants_of_households_with_a_sharer_move(case):
    households, liable = case
    benunit, person = frames(households)
    result = frs_non_dependant_normally_resides_with(benunit, person, np.array(liable))
    head = benunit.benunit_id.isin(person[person.hrpid == 1].benunit_id).values
    sharer = np.array(liable) & ~head
    has_sharer = pd.Series(sharer).groupby(benunit.household_id.values).transform("any")
    moved = result != EVERY
    assert not moved[head | sharer | ~has_sharer.values].any()
    assert not (
        (result == OTHERS)
        & (pd.Series(sharer).groupby(benunit.household_id.values).transform("sum") != 1)
    ).any()


@settings(max_examples=200, deadline=None)
@given(survey(), st.randoms(use_true_random=False))
def test_row_order_does_not_matter(case, random):
    households, liable = case
    benunit, person = frames(households)
    expected = frs_non_dependant_normally_resides_with(
        benunit, person, np.array(liable)
    )
    order = list(range(len(benunit)))
    random.shuffle(order)
    people = list(range(len(person)))
    random.shuffle(people)
    shuffled = frs_non_dependant_normally_resides_with(
        benunit.iloc[order].reset_index(drop=True),
        person.iloc[people].reset_index(drop=True),
        np.array(liable)[order],
    )
    assert shuffled.tolist() == expected[order].tolist()


@settings(max_examples=200, deadline=None)
@given(survey(), st.sampled_from([0, -1, np.nan]))
def test_non_relatives_and_missing_codes_are_interchangeable(case, blank):
    households, liable = case
    benunit, person = frames(households)
    expected = frs_non_dependant_normally_resides_with(
        benunit, person, np.array(liable)
    )
    grid = person[GRID]
    person[GRID] = grid.where(grid.isin(RELATED), blank)
    assert (
        frs_non_dependant_normally_resides_with(benunit, person, np.array(liable))
        == expected
    ).all()
