"""Wholly pension-age benefit units never get would_claim_uc."""

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from policyengine_uk_data.datasets.frs import (
    derive_all_claimants_over_state_pension_age,
)
from policyengine_uk_data.utils.benefit_units import claimant_or_partner_variable


def test_examples():
    # Units: pensioner couple, mixed-age couple, working-age single, a
    # pensioner and a younger person flagged as claimant or partner, a unit
    # with no claimant, and a unit with no members at all.
    result = derive_all_claimants_over_state_pension_age(
        person_benunit_ids=[10, 10, 20, 20, 30, 40, 40, 50],
        is_claimant_or_partner=[1, 1, 1, 1, 1, 1, 1, 0],
        is_over_state_pension_age=[1, 1, 1, 0, 0, 1, 0, 0],
        benunit_ids=[10, 20, 30, 40, 50, 60],
    )
    np.testing.assert_array_equal(result, [True, False, False, False, False, False])


@settings(max_examples=300, deadline=None)
@given(
    st.lists(
        st.tuples(st.integers(0, 6), st.booleans(), st.booleans()),
        max_size=30,
    ),
    st.permutations(list(range(8))),
)
def test_matches_a_direct_definition(people, benunit_order):
    """Differential check against a per-unit loop, for any unit order."""
    benunit_ids = np.array(benunit_order)
    result = derive_all_claimants_over_state_pension_age(
        person_benunit_ids=[p[0] for p in people],
        is_claimant_or_partner=[p[1] for p in people],
        is_over_state_pension_age=[p[2] for p in people],
        benunit_ids=benunit_ids,
    )
    for i, unit in enumerate(benunit_ids):
        adults = [over for b, adult, over in people if b == unit and adult]
        assert result[i] == (len(adults) > 0 and all(adults))


def test_claimant_or_partner_variable_prefers_the_legal_flag():
    assert claimant_or_partner_variable({"is_adult": 0}) == "is_adult"
    assert (
        claimant_or_partner_variable({"is_adult": 0, "is_claimant_or_partner": 0})
        == "is_claimant_or_partner"
    )


def test_built_dataset_has_no_pension_age_uc_claimants(baseline):
    year = 2025
    claimant = claimant_or_partner_variable(baseline.tax_benefit_system.variables)
    adult = baseline.calculate(claimant, year).values.astype(bool)
    over = adult & baseline.calculate("is_SP_age", year).values.astype(bool)
    adults = baseline.map_result(adult.astype(float), "person", "benunit")
    adults_over = baseline.map_result(over.astype(float), "person", "benunit")
    wholly_pension_age = (adults > 0) & (adults_over == adults)
    would_claim_uc = baseline.calculate("would_claim_uc", year).values.astype(bool)
    assert wholly_pension_age.any()
    assert not (would_claim_uc & wholly_pension_age).any()
