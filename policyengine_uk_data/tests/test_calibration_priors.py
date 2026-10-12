"""Tests for the calibration's starting weights.

``prior_equivalent_weights`` expresses the prior of every household, survey
row or zero-weight row, as a weight, and ``rake_priors_to_targets`` scales
the priors so that chosen national targets are met before the first epoch.
"""

import numpy as np
import pandas as pd
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from policyengine_uk_data.utils.calibrate import (
    DEFAULT_ZERO_WEIGHT_PRIOR_TOTAL_SHARE,
    initialize_weight_priors,
    prior_equivalent_weights,
    rake_priors_to_targets,
)

weights = st.lists(
    st.one_of(st.just(0.0), st.floats(0.01, 1e5)), min_size=1, max_size=50
).map(np.array)


@given(weights)
def test_equivalent_weights_earn_each_row_its_prior(original):
    assume((original > 0).any())
    equivalent = prior_equivalent_weights(original)
    priors = initialize_weight_priors(original)
    positive = original > 0

    # A survey row is worth its own weight.
    np.testing.assert_allclose(equivalent[positive], original[positive])
    if positive.all():
        return
    # A zero-weight row is worth the weight that earns a survey row the
    # zero-weight row's prior.
    np.testing.assert_allclose(
        equivalent[~positive] * (1 - DEFAULT_ZERO_WEIGHT_PRIOR_TOTAL_SHARE),
        priors[~positive],
    )


def test_equivalent_weights_without_survey_rows_are_the_priors():
    np.testing.assert_array_equal(prior_equivalent_weights(np.zeros(3)), np.ones(3))


@st.composite
def raking_problems(draw):
    n = draw(st.integers(1, 40))
    priors = np.array(draw(st.lists(st.floats(0.01, 1e4), min_size=n, max_size=n)))
    n_targets = draw(st.integers(1, 5))
    # Each household counts towards at most one target (-1: none).
    member = np.array(
        draw(st.lists(st.integers(-1, n_targets - 1), min_size=n, max_size=n))
    )
    counts = np.array(draw(st.lists(st.integers(1, 3), min_size=n, max_size=n)))
    matrix = pd.DataFrame(
        {
            f"band_{t}": np.where(member == t, counts, 0).astype(float)
            for t in range(n_targets)
        }
    )
    matrix["other"] = 1.0
    targets = pd.Series(
        draw(
            st.lists(
                st.floats(0.01, 1e6), min_size=n_targets + 1, max_size=n_targets + 1
            )
        ),
        index=matrix.columns,
    )
    return priors, matrix, targets, [f"band_{t}" for t in range(n_targets)]


@given(raking_problems())
@settings(max_examples=200)
def test_rake_meets_every_named_target_and_touches_nothing_else(problem):
    priors, matrix, targets, names = problem
    raked = rake_priors_to_targets(priors, matrix, targets, names)

    populated = [name for name in names if (matrix[name] > 0).any()]
    np.testing.assert_allclose(raked @ matrix[populated], targets[populated])
    outside = (matrix[names] == 0).all(axis=1).to_numpy()
    np.testing.assert_array_equal(raked[outside], priors[outside])
    assert (raked > 0).all()
    # Within a target, households keep their relative priors.
    for name in populated:
        members = (matrix[name] > 0).to_numpy()
        ratio = raked[members] / priors[members]
        np.testing.assert_allclose(ratio, ratio[0])
    # Raking again changes nothing, and the input is not modified.
    np.testing.assert_allclose(
        rake_priors_to_targets(raked, matrix, targets, names), raked
    )
    assert raked is not priors


def test_rake_skips_names_the_matrix_lacks_and_targets_with_no_households():
    priors = np.array([1.0, 2.0, 3.0])
    matrix = pd.DataFrame({"a": [1.0, 0.0, 0.0], "empty": [0.0, 0.0, 0.0]})
    targets = pd.Series({"a": 5.0, "empty": 7.0})
    raked = rake_priors_to_targets(priors, matrix, targets, ["a", "empty", "absent"])
    np.testing.assert_array_equal(raked, [5.0, 2.0, 3.0])


def test_rake_rejects_overlapping_targets():
    matrix = pd.DataFrame({"a": [1.0, 1.0], "b": [0.0, 1.0]})
    with pytest.raises(ValueError, match="disjoint"):
        rake_priors_to_targets(
            np.ones(2), matrix, pd.Series({"a": 1.0, "b": 1.0}), ["a", "b"]
        )


def test_rake_rejects_a_non_positive_target():
    matrix = pd.DataFrame({"a": [1.0, 0.0]})
    with pytest.raises(ValueError, match="non-positive"):
        rake_priors_to_targets(np.ones(2), matrix, pd.Series({"a": 0.0}), ["a"])
