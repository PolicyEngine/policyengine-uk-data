"""Splitting private renters across BRMAs, and the calibration that ties them.

Invariants, for any distributions, keys and generator state:

1. Support: a quantile never selects a BRMA with zero probability, and
   always a column of the household's region.
2. Exact marginals: each record's BRMA has exactly its household's
   distribution (``household_brma_probabilities``), whatever the walk order.
3. Spacing: a household's ``k`` quantiles fall one in each interval of width
   ``1/k``, so walking BRMAs in increasing order of an outcome, the records'
   mean of that outcome is within ``range / k`` of its expectation.
4. Conservation: a split household's records carry exactly its weight, people
   and benefit units; other households are untouched; every id stays unique;
   ``benunit_id // 100 == household_id`` holds wherever it held.
5. Determinism: the same seed gives the same records.
6. Tying: a group's calibrated weights are equal, its target rows identical,
   and its zero-weight rows share one household's prior.
"""

import importlib.util

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from policyengine_uk_data.datasets.brma import (
    BRMA_SPLIT_GROUP_COLUMN,
    brmas_at_quantiles,
    household_brma_probabilities,
    load_brma_weights,
    spaced_quantiles,
)
from policyengine_uk_data.utils.calibrate import (
    initialize_weight_priors,
    tie_rows,
    tie_weights,
)


@st.composite
def distributions(draw, max_rows=6, max_width=7):
    """Rows of probabilities with some zero columns, and keys to order them."""
    rows = draw(st.integers(1, max_rows))
    width = draw(st.integers(1, max_width))
    raw = draw(
        hnp.arrays(
            float,
            (rows, width),
            elements=st.sampled_from([0.0, 0.0, 1e-9, 0.1, 0.5, 1.0, 3.0, 1e6]),
        )
    )
    raw[raw.sum(axis=1) == 0, 0] = 1.0  # every row needs some probability
    keys = draw(
        hnp.arrays(float, (rows, width), elements=st.floats(-5, 5, allow_nan=False))
    )
    return raw / raw.sum(axis=1, keepdims=True), keys


def exact_frequencies(probabilities, keys, grid=20_000):
    """Share of a fine quantile grid landing on each column, per row."""
    tau = (np.arange(grid) + 0.5) / grid
    out = np.zeros_like(probabilities)
    for row in range(len(probabilities)):
        columns = brmas_at_quantiles(
            np.repeat(probabilities[row : row + 1], grid, axis=0),
            None if keys is None else np.repeat(keys[row : row + 1], grid, axis=0),
            tau,
        )
        out[row] = np.bincount(columns, minlength=probabilities.shape[1]) / grid
    return out


@settings(max_examples=60, deadline=None, derandomize=True)
@given(distributions(), st.booleans())
def test_quantiles_have_exact_marginals_and_support(case, ordered):
    probabilities, keys = case
    keys = keys if ordered else None
    frequencies = exact_frequencies(probabilities, keys)
    # A grid of n points matches each column's probability to within 1/n per
    # boundary; zero-probability columns are never chosen.
    assert np.abs(frequencies - probabilities).max() <= 2 / 20_000 + 1e-12
    assert (frequencies[probabilities == 0] == 0).all()


@settings(max_examples=200, deadline=None, derandomize=True)
@given(distributions(), st.floats(0, 1, exclude_max=True))
def test_quantiles_never_leave_the_support(case, tau):
    probabilities, keys = case
    for t in (tau, np.nextafter(1.0, 0.0), 0.0):
        columns = brmas_at_quantiles(probabilities, keys, np.full(len(keys), t))
        assert (probabilities[np.arange(len(columns)), columns] > 0).all()


@settings(max_examples=100, deadline=None, derandomize=True)
@given(distributions())
def test_walk_follows_the_keys(case):
    probabilities, keys = case
    tau = np.linspace(0, 1, 50, endpoint=False)
    for row in range(len(probabilities)):
        columns = brmas_at_quantiles(
            np.repeat(probabilities[row : row + 1], len(tau), axis=0),
            np.repeat(keys[row : row + 1], len(tau), axis=0),
            tau,
        )
        assert (np.diff(keys[row][columns]) >= 0).all()


@settings(max_examples=100, deadline=None, derandomize=True)
@given(distributions(), st.integers(1, 9), st.integers(0, 2**32 - 1))
def test_spaced_records_track_the_expectation(case, k, seed):
    probabilities, keys = case
    quantiles = spaced_quantiles(len(probabilities), k, np.random.default_rng(seed))
    assert ((quantiles >= 0) & (quantiles < 1)).all()
    # One quantile in each interval [i/k, (i+1)/k).
    assert (np.sort(np.floor(quantiles * k), axis=1) == np.arange(k)).all()
    columns = brmas_at_quantiles(
        np.repeat(probabilities, k, axis=0), np.repeat(keys, k, axis=0), quantiles.ravel()
    ).reshape(-1, k)
    # Outcome = the key itself (any outcome monotone in the walk order).
    outcome = np.where(probabilities > 0, keys, 0)
    mean = np.take_along_axis(outcome, columns, axis=1).mean(axis=1)
    expected = (probabilities * outcome).sum(axis=1)
    support = np.where(probabilities > 0, keys, np.nan)
    spread = np.nanmax(support, axis=1) - np.nanmin(support, axis=1)
    assert (np.abs(mean - expected) <= spread / k + 1e-9).all()


def test_spaced_quantiles_reject_k_below_one():
    with pytest.raises(ValueError):
        spaced_quantiles(3, 0, np.random.default_rng(0))


@pytest.fixture(scope="module")
def weights():
    return load_brma_weights()


def test_household_distribution_is_the_mean_of_its_benefit_units(weights):
    region = np.array(["SCOTLAND", "SCOTLAND", "NORTHERN_IRELAND"])
    benunit_household = np.array([0, 1, 1, 2])
    category = np.array(["C", "A", "E", "B"])
    brmas, p = household_brma_probabilities(region, benunit_household, category, weights)
    assert np.allclose(p.sum(axis=1), 1)

    def cell(r, c):
        g = weights[(weights.region == r) & (weights.lha_category == c)]
        v = g.set_index("brma").weight.reindex(brmas[r]).fillna(0).to_numpy()
        return np.pad(v / v.sum(), (0, p.shape[1] - len(v)))

    assert np.allclose(p[0], cell("SCOTLAND", "C"))
    assert np.allclose(p[1], (cell("SCOTLAND", "A") + cell("SCOTLAND", "E")) / 2)
    assert np.allclose(p[2], cell("NORTHERN_IRELAND", "B"))
    assert brmas["SCOTLAND"] == sorted(brmas["SCOTLAND"])
    assert (p[:, len(brmas["NORTHERN_IRELAND"]) :][2] == 0).all()


def test_household_distribution_fails_closed(weights):
    table = weights[~((weights.region == "WALES") & (weights.lha_category == "D"))]
    with pytest.raises(ValueError, match="No BRMA weights"):
        household_brma_probabilities(
            np.array(["WALES"]), np.array([0]), np.array(["D"]), table
        )
    with pytest.raises(ValueError, match="benefit unit"):
        household_brma_probabilities(
            np.array(["WALES", "WALES"]), np.array([0]), np.array(["C"]), weights
        )


# --- splitting a dataset (runs PolicyEngine UK on a few synthetic households) ---

needs_policyengine = pytest.mark.skipif(
    importlib.util.find_spec("policyengine_uk") is None,
    reason="policyengine_uk not available",
)


def toy_dataset():
    from policyengine_uk.data import UKSingleYearDataset

    regions = ["LONDON", "LONDON", "SCOTLAND", "WALES", "NORTH_WEST", "LONDON"] * 2
    tenure = ["RENT_PRIVATELY", "OWNED_OUTRIGHT", "RENT_PRIVATELY"] * 4
    n = len(regions)
    household = pd.DataFrame(
        {
            "household_id": np.arange(1, n + 1),
            "household_weight": np.r_[np.linspace(100, 2_000, n - 2), 0.0, 0.0],
            "region": regions,
            "tenure_type": tenure,
            "rent": np.linspace(4_000, 30_000, n),
            "brma": ["INNER_NORTH_LONDON"] * n,
            "council_tax": np.full(n, 1_500.0),
        }
    )
    people, units = [], []
    for h in household.household_id:
        # Households 3 and 6 hold two benefit units (a lodger family).
        for b in range(1, 3 if h % 3 == 0 else 2):
            units.append({"benunit_id": h * 100 + b})
            for p in range(1, 3):
                people.append(
                    {
                        "person_id": h * 1000 + b * 10 + p,
                        "person_household_id": h,
                        "person_benunit_id": h * 100 + b,
                        "age": 25 + 10 * b + p,
                    }
                )
    return UKSingleYearDataset(
        person=pd.DataFrame(people),
        benunit=pd.DataFrame(units),
        household=household,
        fiscal_year=2024,
    )


@needs_policyengine
@pytest.mark.parametrize("k", [1, 3])
def test_split_conserves_households_and_keeps_ids_unique(k, weights):
    from policyengine_uk_data.datasets.brma import split_private_renters_across_brmas

    dataset = toy_dataset()
    result = split_private_renters_across_brmas(dataset, k=k, seed=7, weights=weights)
    before, after = dataset.household, result.household
    renter = before.tenure_type.astype(str) == "RENT_PRIVATELY"

    group = after[BRMA_SPLIT_GROUP_COLUMN]
    assert set(group) == set(before.household_id)
    sizes = group.value_counts()
    assert (sizes.loc[before.household_id[renter]] == k).all()
    assert (sizes.loc[before.household_id[~renter]] == 1).all()
    # Each household's weight is conserved across its records.
    weight = after.groupby(group).household_weight.sum()
    assert np.allclose(weight.loc[before.household_id], before.household_weight)
    # Non-renters keep their record exactly.
    kept = after.set_index("household_id").loc[before.household_id[~renter]]
    assert (kept.brma.astype(str) == "INNER_NORTH_LONDON").all()
    # Every record's BRMA is a positive-weight BRMA of its own region.
    supported = set(zip(weights.region, weights.brma))
    renters_after = after[after.tenure_type.astype(str) == "RENT_PRIVATELY"]
    if k > 1:
        assert all(
            (r, b) in supported
            for r, b in zip(renters_after.region.astype(str), renters_after.brma.astype(str))
        )
    for table, column in (
        (result.household, "household_id"),
        (result.benunit, "benunit_id"),
        (result.person, "person_id"),
    ):
        assert table[column].is_unique
    assert set(result.person.person_household_id) == set(after.household_id)
    assert set(result.person.person_benunit_id) == set(result.benunit.benunit_id)
    # The id scheme benunit_id // 100 == household_id survives the copies.
    owner = result.person.drop_duplicates("person_benunit_id").set_index(
        "person_benunit_id"
    ).person_household_id
    assert (result.benunit.benunit_id // 100 == owner.loc[result.benunit.benunit_id].to_numpy()).all()
    assert len(result.person) == len(dataset.person) + (k - 1) * len(
        dataset.person[dataset.person.person_household_id.isin(before.household_id[renter])]
    )


@needs_policyengine
def test_split_is_deterministic(weights):
    from policyengine_uk_data.datasets.brma import split_private_renters_across_brmas

    a = split_private_renters_across_brmas(toy_dataset(), k=4, seed=3, weights=weights)
    b = split_private_renters_across_brmas(toy_dataset(), k=4, seed=3, weights=weights)
    pd.testing.assert_frame_equal(a.household, b.household)
    pd.testing.assert_frame_equal(a.person, b.person)


# --- calibration ties ---


@settings(max_examples=100, deadline=None, derandomize=True)
@given(
    st.lists(st.integers(0, 4), min_size=1, max_size=12),
    st.integers(1, 3),
    st.integers(0, 2**32 - 1),
)
def test_tied_rows_and_weights_are_group_means_and_keep_totals(labels, areas, seed):
    rng = np.random.default_rng(seed)
    groups = np.array(labels)
    rows = rng.random((len(groups), 3))
    tied = tie_rows(pd.DataFrame(rows), groups)
    for g in set(labels):
        m = groups == g
        assert np.allclose(tied.to_numpy()[m], rows[m].mean(axis=0))
    w = rng.random((areas, len(groups)))
    tw = tie_weights(w, groups)
    for g in set(labels):
        m = groups == g
        assert np.allclose(tw[:, m], tw[:, m][:, :1])
        assert np.allclose(tw[:, m].sum(axis=1), w[:, m].sum(axis=1))
    # Predictions w @ rows are unchanged once rows are tied.
    assert np.allclose(tw @ tied.to_numpy(), w @ tied.to_numpy())


def test_split_zero_weight_rows_share_one_households_prior():
    weights = np.array([1_500.0, 0.0, 0.0, 0.0, 625.0, 0.0])
    groups = np.array([1, 2, 2, 2, 3, 4])  # rows 1-3 split one SPI household
    priors = initialize_weight_priors(weights, groups=groups)
    ungrouped = initialize_weight_priors(np.array([1_500.0, 0.0, 625.0, 0.0]))
    assert priors[[1, 2, 3]].sum() == pytest.approx(ungrouped[1])
    assert priors[5] == pytest.approx(ungrouped[3])
    assert np.allclose(priors[[0, 4]], ungrouped[[0, 2]])
    assert priors.sum() == pytest.approx(weights.sum())


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="needs torch")
def test_calibration_keeps_a_groups_records_equal(tmp_path, monkeypatch):
    from policyengine_uk_data.utils import calibrate as calibrate_module
    from policyengine_uk_data.utils.calibrate import calibrate_local_areas

    monkeypatch.setattr(calibrate_module, "STORAGE_FOLDER", tmp_path)
    # Records 0-2 split one household; their rows differ before tying.
    matrix = pd.DataFrame([[1.0, 0.0], [3.0, 0.0], [2.0, 0.0], [0.0, 1.0], [0.0, 2.0]])
    local = pd.DataFrame([[600.0, 300.0], [500.0, 400.0]])
    mask = np.ones((2, 5))

    class Data:
        def __init__(self, w):
            self.household = pd.DataFrame({"household_weight": w})

        def copy(self):
            return Data(self.household.household_weight.to_numpy().copy())

    result = calibrate_local_areas(
        dataset=Data(np.array([100.0, 100.0, 100.0, 300.0, 300.0])),
        matrix_fn=lambda d: (matrix.copy(), local.copy(), mask.copy()),
        national_matrix_fn=lambda d: (pd.DataFrame(np.ones((5, 1))), pd.Series([1_500.0])),
        area_count=2,
        weight_file="tied.h5",
        dataset_key="2024",
        epochs=21,
        groups=np.array([0, 0, 0, 1, 2]),
    )
    w = result.household.household_weight.to_numpy()
    assert w[0] == pytest.approx(w[1]) and w[1] == pytest.approx(w[2])
