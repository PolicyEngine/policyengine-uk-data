"""Tests for the fitted Pareto upper tail of the capital gains imputation.

The Advani & Summers percentile table stops at p95. Before this tail was
added, sampling uniform quantiles through the linear spline extrapolated off
the last segment, which removed the heavy tail of capital gains entirely.
These tests pin the three properties that matter:

* the body below p95 is still exactly the Advani & Summers spline;
* the tail above p95 reproduces the concentration implied by the published
  band means (and, in aggregate, HMRC's size-of-gain distribution);
* sampling stays deterministic under the build's seed.
"""

import importlib

import numpy as np
import pytest

# The imputations package re-exports a DataFrame named ``capital_gains``, which
# shadows the submodule of the same name, so import it explicitly.
cg_module = importlib.import_module(
    "policyengine_uk_data.datasets.imputations.capital_gains"
)

TAIL_START_QUANTILE = cg_module.TAIL_START_QUANTILE
capital_gains = cg_module.capital_gains
sample_band_gains = cg_module.sample_band_gains


@pytest.fixture(scope="module")
def bands():
    return [capital_gains.iloc[i] for i in range(len(capital_gains))]


def test_body_below_p95_is_unchanged_spline(bands):
    """Below p95 the sampler must be bit-for-bit the original spline."""
    quantiles = np.linspace(0, TAIL_START_QUANTILE, 500, endpoint=False)
    for row in bands:
        expected = cg_module._band_spline(row)(quantiles)
        actual = sample_band_gains(row, quantiles)
        np.testing.assert_allclose(actual, expected, rtol=0, atol=0)


def test_body_tracks_published_percentiles(bands):
    """The body still follows the published percentiles.

    Note this is a *smoothing* spline (scipy's default ``s``), not an
    interpolating one, so it does not pass exactly through the published
    points. That is pre-existing behaviour and is deliberately preserved here;
    the tolerance below just pins that the body has not drifted.
    """
    published = np.array([0.05, 0.1, 0.25, 0.5, 0.75, 0.90])
    for row in bands:
        expected = np.array([row.p05, row.p10, row.p25, row.p50, row.p75, row.p90])
        actual = sample_band_gains(row, published)
        scale = max(abs(row.p90), 1.0)
        assert np.max(np.abs(actual - expected)) < 0.5 * scale


def test_tail_is_anchored_at_p95(bands):
    """The tail starts at p95 and is monotonically increasing in the quantile."""
    for row in bands:
        if row.p95 <= 0:
            continue
        q = np.linspace(TAIL_START_QUANTILE, 1 - 1e-9, 2000)
        tail = sample_band_gains(row, q)
        assert tail[0] == pytest.approx(row.p95, rel=1e-6)
        assert np.all(np.diff(tail) >= 0)
        assert tail.max() <= cg_module.MAX_SINGLE_GAIN


def test_tail_reproduces_published_band_means(bands):
    """The fitted tail is what makes each band hit its published mean.

    This is the calibration condition the shape parameter is solved from, so
    it is the property that would break first if the fit regressed.
    """
    rng = np.random.default_rng(0)
    floored = 0
    for row in bands:
        alpha = cg_module._fit_tail_alpha(row)
        sampled = sample_band_gains(row, rng.random(200_000))
        if alpha <= cg_module.MIN_TAIL_ALPHA:
            # A small number of bands have an anomalously low p95 relative to
            # their published mean; the alpha floor deliberately stops us
            # chasing those with an unstably heavy tail, at the cost of
            # undershooting the band mean.
            floored += 1
            continue
        assert sampled.mean() == pytest.approx(row.mean_gains_given_gains, rel=0.15)
    # If many bands start hitting the floor, the fit has regressed.
    assert floored <= 3


def test_spline_alone_undershoots_published_means(bands):
    """Guards the diagnosis: the body alone cannot reach the published mean.

    If this ever stops holding, the tail correction is no longer justified on
    these grounds and the fit should be revisited.
    """
    rng = np.random.default_rng(0)
    shortfalls = []
    for row in bands:
        body_only = cg_module._band_spline(row)(rng.random(100_000))
        shortfalls.append(body_only.mean() / row.mean_gains_given_gains)
    # Every band undershoots, typically by 35-50%.
    assert max(shortfalls) < 0.95


def test_fitted_alphas_are_in_a_plausible_heavy_tail_range(bands):
    alphas = np.array([cg_module._fit_tail_alpha(row) for row in bands])
    assert np.all(alphas >= cg_module.MIN_TAIL_ALPHA)
    # Capital gains tails are heavy; a median far outside this range would
    # indicate the moment condition is being solved against bad inputs.
    assert 1.2 < np.median(alphas) < 2.0


def test_tail_delivers_concentration(bands):
    """Sampled gains must concentrate, which the truncated spline never did."""
    rng = np.random.default_rng(0)
    sampled = np.concatenate(
        [sample_band_gains(row, rng.random(50_000)) for row in bands]
    )
    positive = sampled[sampled > 0]
    share_1m = positive[positive >= 1e6].sum() / positive.sum()
    # HMRC put ~61% of gains in disposals of £1m+. The imputation is fitted to
    # Advani & Summers band means rather than to HMRC directly, so it lands
    # below that, but it must be far above the ~2% the old spline produced.
    assert share_1m > 0.25

    # The old spline could not produce a gain above ~£2m at all.
    assert positive.max() > 5e6


def test_sampling_is_deterministic_under_the_seed(bands):
    """The build seeds its generator because unseeded draws moved CGT revenue."""
    row = bands[-1]

    def draw():
        rng = np.random.default_rng(0)
        return sample_band_gains(row, rng.random(10_000))

    np.testing.assert_array_equal(draw(), draw())


def test_hmrc_reference_table_loads_and_is_concentrated():
    """The committed HMRC reference table is the external validation anchor."""
    import pandas as pd

    from policyengine_uk_data.storage import STORAGE_FOLDER

    table = pd.read_csv(
        STORAGE_FOLDER / "capital_gains_size_distribution_hmrc.csv",
        comment="#",
    )
    assert {
        "lower_limit",
        "taxpayers_thousands",
        "gains_gbp_millions",
    } <= set(table.columns)

    total = table.gains_gbp_millions.sum()
    above_1m = table.loc[table.lower_limit >= 1_000_000, "gains_gbp_millions"].sum()
    # HMRC 2023-24: ~61% of gains accrue to disposals of £1m or more.
    assert 0.55 < above_1m / total < 0.70
