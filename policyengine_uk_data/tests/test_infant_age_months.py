"""Children recorded as aged 0 get an age in months (frs.impute_infant_age_in_months).

Data-independent: checks the imputation on synthetic ages.
"""

import inspect

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from policyengine_uk_data.datasets import frs
from policyengine_uk_data.datasets.frs import impute_infant_age_in_months

MONTH_MIDPOINTS = (np.arange(12) + 0.5) / 12


@settings(max_examples=200, deadline=None)
@given(
    st.lists(st.integers(0, 80), min_size=1, max_size=200),
    st.integers(0, 2**32 - 1),
)
def test_only_infants_change_and_every_whole_year_threshold_holds(ages, seed):
    ages = np.array(ages, dtype=float)
    out = impute_infant_age_in_months(ages, np.random.default_rng(seed))
    # Nobody's whole year moves, so every whole-year threshold is unchanged.
    assert (np.floor(out) == ages).all()
    # Only children recorded as 0 change, and each gets a month midpoint.
    assert (out[ages > 0] == ages[ages > 0]).all()
    infants = out[ages == 0]
    assert np.isin(np.round(infants * 24), np.round(MONTH_MIDPOINTS * 24)).all()
    # The input is not modified in place.
    assert (ages == np.floor(out)).all()


def test_a_quarter_of_infants_are_nine_to_eleven_months():
    out = impute_infant_age_in_months(np.zeros(120_000), np.random.default_rng(0))
    assert abs((out >= 0.75).mean() - 0.25) < 0.01


def test_the_imputation_is_reproducible():
    ages = np.array([0, 3, 0, 40, 0, 0], dtype=float)
    a = impute_infant_age_in_months(
        ages, np.random.default_rng(frs.INFANT_AGE_MONTHS_SEED)
    )
    b = impute_infant_age_in_months(
        ages, np.random.default_rng(frs.INFANT_AGE_MONTHS_SEED)
    )
    assert (a == b).all()


def test_the_build_uses_its_own_generator():
    """A dedicated generator leaves every other random draw in the build as it
    was; drawing from the shared one would shift them all."""
    source = inspect.getsource(frs.create_frs)
    assert (
        "impute_infant_age_in_months(\n"
        "        age.values, np.random.default_rng(INFANT_AGE_MONTHS_SEED)\n"
        "    )" in source
    )
