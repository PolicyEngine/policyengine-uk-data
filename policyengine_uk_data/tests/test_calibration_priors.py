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
    calibrate_local_areas,
    initialize_weight_priors,
    prior_equivalent_weights,
    rake_priors_to_targets,
)

weights = st.lists(
    st.one_of(st.just(0.0), st.floats(0.01, 1e5)), min_size=1, max_size=50
).map(np.array)


@given(weights)
@settings(deadline=None)
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
@settings(max_examples=200, deadline=None)
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


@pytest.mark.parametrize(
    ("carried_priors", "rake_targets", "expected_household_priors"),
    [
        pytest.param(
            [10.0, 20.0, 40.0, 30.0],
            ("band_a", "band_b"),
            [100.0, 100.0, 400.0, 150.0],
            id="carried-priors-raked",
        ),
        pytest.param(
            [10.0, 20.0, 40.0, 30.0],
            (),
            [10.0, 20.0, 40.0, 30.0],
            id="carried-priors-without-rake",
        ),
    ],
)
def test_calibrate_rakes_national_priors_before_splitting_between_areas(
    tmp_path, monkeypatch, carried_priors, rake_targets, expected_household_priors
):
    import h5py

    from policyengine_uk_data.tests.test_calibrate_save import _StubDataset
    from policyengine_uk_data.utils import calibrate as calibrate_module

    national_matrix = pd.DataFrame(
        {"band_a": [1.0, 0.0, 1.0, 0.0], "band_b": [0.0, 1.0, 0.0, 1.0]}
    )
    national_targets = pd.Series({"band_a": 500.0, "band_b": 250.0})
    # Local columns deliberately count different households, and their targets
    # differ from the national targets. The rake must use the national inputs.
    local_matrix = national_matrix[["band_b", "band_a"]].copy()
    local_matrix.columns = national_matrix.columns
    local_targets = pd.DataFrame({"band_a": [1.0, 2.0, 3.0], "band_b": [4.0, 5.0, 6.0]})
    # The first country has two areas; the second has one. Each country's
    # split must sum back to its households' full national starting priors.
    area_mask = np.array(
        [[1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]]
    )
    initial_log_weights = []
    adam = calibrate_module.torch.optim.Adam

    def freeze_optimizer(parameters, **kwargs):
        initial_log_weights.append(parameters[0].detach().numpy().copy())
        optimizer = adam(parameters, **kwargs)
        # Exercise the actual initialization and save paths without allowing
        # an optimizer update to obscure the prior or area-split assertions.
        monkeypatch.setattr(optimizer, "step", lambda: None)
        return optimizer

    monkeypatch.setattr(calibrate_module.torch.optim, "Adam", freeze_optimizer)
    monkeypatch.setattr(calibrate_module, "STORAGE_FOLDER", tmp_path)
    household_columns = {"household_calibration_prior": carried_priors}
    original_weights = np.array([40.0, 0.0, 60.0, 0.0])
    result = calibrate_local_areas(
        dataset=_StubDataset(original_weights, **household_columns),
        matrix_fn=lambda _d: (local_matrix, local_targets, area_mask),
        national_matrix_fn=lambda _d: (national_matrix, national_targets),
        area_count=3,
        weight_file="raked_priors.h5",
        dataset_key="2024",
        epochs=1,
        prior_rake_targets=rake_targets,
    )

    initial_area_weights = np.exp(initial_log_weights[0]) * area_mask
    expected_household_priors = np.array(expected_household_priors)
    expected_national_estimates = (
        national_targets
        if rake_targets
        else expected_household_priors @ national_matrix
    )
    np.testing.assert_allclose(
        initial_area_weights.sum(axis=0), expected_household_priors, rtol=1e-6
    )
    np.testing.assert_allclose(
        initial_area_weights.sum(axis=0) @ national_matrix,
        expected_national_estimates,
        rtol=1e-6,
    )
    expected_area_weights = (
        expected_household_priors / area_mask.sum(axis=0) * area_mask
    )
    np.testing.assert_allclose(initial_area_weights, expected_area_weights, rtol=1e-6)
    with h5py.File(tmp_path / "raked_priors.h5") as file:
        saved_area_weights = file["2024"][:]
    np.testing.assert_allclose(saved_area_weights, initial_area_weights, rtol=1e-6)
    np.testing.assert_allclose(
        saved_area_weights.sum(axis=0) @ national_matrix,
        expected_national_estimates,
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        result.household.household_weight,
        expected_household_priors,
        rtol=1e-6,
    )
    if not rake_targets:
        assert result.household.household_weight.sum() == pytest.approx(
            original_weights.sum()
        )
