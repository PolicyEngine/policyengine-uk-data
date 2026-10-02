"""The claimant-or-partner role: FRS derivation, stacking, and built datasets."""

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from policyengine_uk.data import UKSingleYearDataset

from policyengine_uk_data.datasets.frs import (
    derive_is_claimant_or_partner_from_frs_microdata,
    derive_is_parent_from_frs_microdata,
)
from policyengine_uk_data.utils.stack import stack_datasets


def test_grown_up_child_heads_their_own_benefit_unit():
    # Lone parent and dependent child in one benefit unit; grown-up son in his own.
    result = derive_is_claimant_or_partner_from_frs_microdata(
        person_ids=np.array([1_001, 1_002, 1_003]),
        person_benunit_ids=np.array([101, 102, 101]),
        adult_person_ids=np.array([1_001, 1_002]),
    )
    assert result.tolist() == [True, True, False]


def test_couple_with_children():
    result = derive_is_claimant_or_partner_from_frs_microdata(
        person_ids=np.array([2_003, 2_001, 2_002]),
        person_benunit_ids=np.array([201, 201, 201]),
        adult_person_ids=np.array([2_002, 2_001]),
    )
    assert result.tolist() == [False, True, True]


@pytest.mark.parametrize("adults", [[], [3_001, 3_002, 3_003]])
def test_benefit_unit_without_one_or_two_adults_is_rejected(adults):
    with pytest.raises(ValueError, match="one or two adult-table records"):
        derive_is_claimant_or_partner_from_frs_microdata(
            person_ids=np.array([3_001, 3_002, 3_003]),
            person_benunit_ids=np.array([301, 301, 301]),
            adult_person_ids=np.array(adults, dtype=int),
        )


@st.composite
def frs_households(draw):
    """FRS-shaped records: benefit units of one or two adults plus children."""
    rows = []
    for household in range(1, draw(st.integers(1, 6)) + 1):
        person = 0
        for benunit in range(1, draw(st.integers(1, 3)) + 1):
            n_adults = draw(st.integers(1, 2))
            n_children = draw(st.integers(0, 4))
            for index in range(n_adults + n_children):
                person += 1
                rows.append(
                    (
                        household * 1_000 + person,
                        household * 100 + benunit,
                        index < n_adults,
                    )
                )
    order = draw(st.permutations(range(len(rows))))
    frame = pd.DataFrame(rows, columns=["person_id", "benunit_id", "adult"])
    return frame.iloc[list(order)].reset_index(drop=True)


def _derive(frame, adult_mask=None):
    adult = frame.adult if adult_mask is None else adult_mask
    return derive_is_claimant_or_partner_from_frs_microdata(
        person_ids=frame.person_id,
        person_benunit_ids=frame.benunit_id,
        adult_person_ids=frame.person_id[adult],
    )


@settings(max_examples=300, deadline=None)
@given(frs_households())
def test_every_benefit_unit_has_a_claimant_and_at_most_one_partner(frame):
    result = _derive(frame)
    assert (result == frame.adult.to_numpy()).all()
    counts = pd.Series(result).groupby(frame.benunit_id.to_numpy()).sum()
    assert counts.between(1, 2).all()


@settings(max_examples=200, deadline=None)
@given(frs_households(), st.randoms(use_true_random=False))
def test_record_order_does_not_change_anyones_role(frame, rng):
    order = list(range(len(frame)))
    rng.shuffle(order)
    shuffled = frame.iloc[order].reset_index(drop=True)
    by_person = dict(zip(frame.person_id, _derive(frame)))
    assert [by_person[p] for p in shuffled.person_id] == _derive(shuffled).tolist()


@settings(max_examples=200, deadline=None)
@given(frs_households(), st.data())
def test_parents_are_always_claimant_or_partner(frame, data):
    benunits = np.sort(frame.benunit_id.unique())
    children = data.draw(
        st.lists(st.integers(0, 4), min_size=len(benunits), max_size=len(benunits))
    )
    is_parent = derive_is_parent_from_frs_microdata(
        person_ids=frame.person_id,
        person_benunit_ids=frame.benunit_id,
        adult_person_ids=frame.person_id[frame.adult],
        benunit_ids=benunits,
        dependent_children=np.array(children),
    )
    assert not (is_parent & ~_derive(frame)).any()


@settings(max_examples=200, deadline=None)
@given(frs_households(), st.data())
def test_a_benefit_unit_with_no_adult_or_a_third_adult_is_rejected(frame, data):
    benunit = data.draw(st.sampled_from(sorted(frame.benunit_id.unique())))
    members = frame.benunit_id == benunit
    if data.draw(st.booleans()):
        adult = frame.adult & ~members
    else:
        n_extra = 3 - int(frame.adult[members].sum())
        start = frame.person_id.max() + 1
        extra = pd.DataFrame(
            {
                "person_id": range(start, start + n_extra),
                "benunit_id": benunit,
                "adult": True,
            }
        )
        frame = pd.concat([frame, extra], ignore_index=True)
        adult = frame.adult
    with pytest.raises(ValueError, match="one or two adult-table records"):
        _derive(frame, adult)


def _tiny_dataset(roles=True):
    person = pd.DataFrame(
        {
            "person_id": [1, 2, 3],
            "person_benunit_id": [1, 1, 1],
            "person_household_id": [1, 1, 1],
            "age": [40, 38, 10],
        }
    )
    if roles:
        person["is_claimant_or_partner"] = [True, True, False]
    return UKSingleYearDataset(
        person=person,
        benunit=pd.DataFrame({"benunit_id": [1]}),
        household=pd.DataFrame({"household_id": [1], "household_weight": [1.0]}),
        fiscal_year=2024,
    )


def test_stacking_keeps_roles():
    stacked = stack_datasets(_tiny_dataset(), _tiny_dataset())
    stacked.validate()
    assert stacked.person.is_claimant_or_partner.dtype == bool
    assert stacked.person.is_claimant_or_partner.tolist() == [True, True, False] * 2


@pytest.mark.parametrize("roles", [(True, False), (False, True)])
def test_stacking_refuses_a_table_without_roles(roles):
    with pytest.raises(ValueError, match="is_claimant_or_partner"):
        stack_datasets(_tiny_dataset(roles[0]), _tiny_dataset(roles[1]))


@pytest.mark.parametrize("fixture", ["frs", "enhanced_frs"])
def test_built_dataset_roles(fixture, request):
    person = request.getfixturevalue(fixture).person
    role = person.is_claimant_or_partner
    assert role.dtype == bool
    counts = role.groupby(person.person_benunit_id).sum()
    assert counts.between(1, 2).all()
    assert not (person.is_benunit_head & ~role).any()
    assert not (person.is_parent & ~role).any()
    assert (person.age[role] >= 16).all()
