"""Tests for post-calibration population rescaling."""

import numpy as np
import pandas as pd
import pytest

from policyengine_uk_data.utils.calibrate import rescale_weights_to_population


def _make_national_matrix(pop_metric, other_metric=None):
    """Build a small national matrix DataFrame with an ons/uk_population column."""
    data = {"ons/uk_population": pop_metric}
    if other_metric is not None:
        data["obr/income_tax"] = other_metric
    return pd.DataFrame(data)


def _make_targets(values, index):
    return pd.Series(values, index=index)


class TestRescaleWeightsToPopulation:
    def test_scales_weights_down_when_population_too_high(self):
        """Weights should shrink when actual population exceeds target."""
        n_areas, n_hh = 3, 4
        weights = np.ones((n_areas, n_hh)) * 10.0  # each HH weight=10 per area
        # pop_metric=1 per household → actual_pop = sum(axis=0)*1 = 30 per HH, total=120
        matrix = _make_national_matrix(np.ones(n_hh))
        target_pop = 60.0  # half of actual
        targets = _make_targets([target_pop], index=["ons/uk_population"])

        rescaled, scale = rescale_weights_to_population(weights, matrix, targets)

        assert scale == pytest.approx(0.5, rel=1e-6)
        assert rescaled.sum() == pytest.approx(weights.sum() * 0.5, rel=1e-6)

    def test_scales_weights_up_when_population_too_low(self):
        """Weights should grow when actual population is below target."""
        n_areas, n_hh = 2, 5
        weights = np.ones((n_areas, n_hh)) * 5.0
        matrix = _make_national_matrix(np.ones(n_hh))
        # national_weights = sum(axis=0) = [10,10,10,10,10], actual_pop = 50
        target_pop = 100.0
        targets = _make_targets([target_pop], index=["ons/uk_population"])

        rescaled, scale = rescale_weights_to_population(weights, matrix, targets)

        assert scale == pytest.approx(2.0, rel=1e-6)
        np.testing.assert_allclose(rescaled, weights * 2.0)

    def test_no_change_when_population_matches(self):
        """Scale should be 1.0 when actual population already matches target."""
        weights = np.array([[3.0, 7.0]])
        matrix = _make_national_matrix(np.array([1.0, 1.0]))
        # national_weights = [3, 7], actual_pop = 10
        targets = _make_targets([10.0], index=["ons/uk_population"])

        rescaled, scale = rescale_weights_to_population(weights, matrix, targets)

        assert scale == pytest.approx(1.0, rel=1e-6)
        np.testing.assert_array_equal(rescaled, weights)

    def test_no_rescale_when_population_column_missing(self):
        """Should return scale=1.0 when ons/uk_population is not in the matrix."""
        weights = np.ones((2, 3)) * 10.0
        matrix = pd.DataFrame({"obr/income_tax": np.ones(3)})
        targets = _make_targets([500.0], index=["obr/income_tax"])

        rescaled, scale = rescale_weights_to_population(weights, matrix, targets)

        assert scale == 1.0
        np.testing.assert_array_equal(rescaled, weights)

    def test_no_rescale_when_actual_population_zero(self):
        """Should return scale=1.0 when actual weighted population is zero."""
        weights = np.zeros((2, 3))
        matrix = _make_national_matrix(np.ones(3))
        targets = _make_targets([69_000_000.0], index=["ons/uk_population"])

        rescaled, scale = rescale_weights_to_population(weights, matrix, targets)

        assert scale == 1.0
        np.testing.assert_array_equal(rescaled, weights)

    def test_works_with_multiple_columns(self):
        """Population column is found even when other columns are present."""
        weights = np.array([[2.0, 8.0]])
        matrix = _make_national_matrix(
            np.array([1.0, 1.0]),
            other_metric=np.array([100.0, 200.0]),
        )
        # actual_pop = 2+8 = 10, target = 20
        targets = _make_targets(
            [20.0, 999.0], index=["ons/uk_population", "obr/income_tax"]
        )

        rescaled, scale = rescale_weights_to_population(weights, matrix, targets)

        assert scale == pytest.approx(2.0, rel=1e-6)

    def test_works_with_numpy_arrays(self):
        """Should handle raw numpy arrays (no DataFrame/Series)."""
        weights = np.ones((2, 4)) * 5.0
        # Without columns attr, pop_idx stays None → no rescaling
        matrix = np.ones((4, 2))
        targets = np.array([40.0, 100.0])

        rescaled, scale = rescale_weights_to_population(weights, matrix, targets)

        assert scale == 1.0  # no columns attr → no rescaling

    def test_1d_weights(self):
        """Should work with 1D weight vectors (single area or flat weights).

        For 1D weights, sum(axis=0) returns the scalar total, so
        actual_pop = total_weight * pop_metric.sum().
        """
        weights = np.array([5.0, 10.0, 15.0])
        matrix = _make_national_matrix(np.array([1.0, 1.0, 1.0]))
        # national_weights = sum(axis=0) = scalar 30
        # actual_pop = (30 * [1,1,1]).sum() = 90
        target_pop = 45.0
        targets = _make_targets([target_pop], index=["ons/uk_population"])

        rescaled, scale = rescale_weights_to_population(weights, matrix, targets)

        assert scale == pytest.approx(0.5, rel=1e-6)
        np.testing.assert_allclose(rescaled, weights * 0.5)

    def test_does_not_mutate_input(self):
        """Input weights array should not be modified in place."""
        weights = np.ones((2, 3)) * 10.0
        original = weights.copy()
        matrix = _make_national_matrix(np.ones(3))
        targets = _make_targets([15.0], index=["ons/uk_population"])

        rescale_weights_to_population(weights, matrix, targets)

        np.testing.assert_array_equal(weights, original)

    def test_realistic_6pct_overshoot(self):
        """Simulate the real-world ~6% overshoot scenario from issue #217."""
        target_pop = 69_000_000.0
        actual_pop = 74_000_000.0  # 7.2% overshoot
        n_hh = 100
        weights = np.ones((1, n_hh)) * (actual_pop / n_hh)
        matrix = _make_national_matrix(np.ones(n_hh))
        targets = _make_targets([target_pop], index=["ons/uk_population"])

        rescaled, scale = rescale_weights_to_population(weights, matrix, targets)

        weighted_pop = rescaled.sum(axis=0).sum()
        assert weighted_pop == pytest.approx(target_pop, rel=1e-6)
        assert scale == pytest.approx(target_pop / actual_pop, rel=1e-6)
