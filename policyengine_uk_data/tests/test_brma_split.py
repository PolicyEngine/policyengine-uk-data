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
6. Tying: calibrating a split household's records with ``groups`` is
   calibrating one household on their mean target rows: the same weights as
   an unsplit dataset holding those mean rows, split equally across the records.
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
from policyengine_uk_data.utils.calibrate import RecordGroups


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
        np.repeat(probabilities, k, axis=0),
        np.repeat(keys, k, axis=0),
        quantiles.ravel(),
    ).reshape(-1, k)
    # Outcome = the key itself (any outcome monotone in the walk order).
    outcome = np.where(probabilities > 0, keys, 0)
    mean = np.take_along_axis(outcome, columns, axis=1).mean(axis=1)
    expected = (probabilities * outcome).sum(axis=1)
    support = np.where(probabilities > 0, keys, np.nan)
    spread = np.nanmax(support, axis=1) - np.nanmin(support, axis=1)
    assert (np.abs(mean - expected) <= spread / k + 1e-9).all()


class GridGenerator:
    """Stands in for a Generator: ``random(n)`` returns the midpoints of n equal cells."""

    def random(self, n):
        return (np.arange(n) + 0.5) / n


@settings(max_examples=60, deadline=None, derandomize=True)
@given(distributions(max_rows=3), st.integers(1, 6))
def test_every_split_record_has_its_households_distribution(case, k):
    # Over a uniform phase, record i's quantile (u + i/k) mod 1 is uniform, so
    # its BRMA has exactly the household's distribution.
    probabilities, keys = case
    n = 6_000
    for row in range(len(probabilities)):
        quantiles = spaced_quantiles(n, k, GridGenerator())
        for i in range(k):
            columns = brmas_at_quantiles(
                np.repeat(probabilities[row : row + 1], n, axis=0),
                np.repeat(keys[row : row + 1], n, axis=0),
                quantiles[:, i],
            )
            frequency = np.bincount(columns, minlength=probabilities.shape[1]) / n
            assert np.abs(frequency - probabilities[row]).max() <= 2 / n + 1e-12


@pytest.mark.parametrize(
    "probabilities, keys, quantile, expected",
    [
        ([0, 0.5, 0, 0.5], None, 0.0, 1),
        ([0, 0.5, 0, 0.5], None, np.nextafter(0.5, 0), 1),
        ([0, 0.5, 0, 0.5], None, 0.5, 3),
        ([0, 0.5, 0, 0.5], None, np.nextafter(1.0, 0), 3),
        ([0.2, 0.3, 0.5], [1.0, 1.0, 1.0], 0.0, 0),  # tied keys keep column order
        ([0.2, 0.3, 0.5], [1.0, 1.0, 1.0], 0.2, 1),
        ([0, 0.4, 0, 0.6], [np.inf, 2.0, np.inf, 1.0], 0.0, 3),  # lowest key first
        ([0, 0.4, 0, 0.6], [np.inf, 2.0, np.inf, 1.0], 0.6, 1),
        ([0, 0.4, 0, 0.6], [np.inf, 2.0, np.inf, 1.0], np.nextafter(1.0, 0), 1),
        (
            [0.5, 0.5, 0, 0],
            [1.0, 2.0, np.inf, np.inf],
            np.nextafter(1.0, 0),
            1,
        ),  # padding
    ],
)
def test_quantile_boundaries(probabilities, keys, quantile, expected):
    p = np.array([probabilities], dtype=float)
    k = None if keys is None else np.array([keys], dtype=float)
    assert brmas_at_quantiles(p, k, np.array([quantile]))[0] == expected


def test_quantiles_outside_the_unit_interval_are_rejected():
    with pytest.raises(ValueError):
        brmas_at_quantiles(np.array([[1.0]]), None, np.array([1.0]))


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
    brmas, p = household_brma_probabilities(
        region, benunit_household, category, weights
    )
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


def test_household_distribution_rejects_zero_mass_cells(weights):
    table = weights.copy()
    cell = (table.region == "WALES") & (table.lha_category == "C")
    table.loc[cell, "weight"] = 0.0
    with pytest.raises(ValueError, match="No BRMA weights"):
        household_brma_probabilities(
            np.array(["WALES"]), np.array([0]), np.array(["C"]), table
        )


# --- splitting a dataset (runs PolicyEngine UK on a few synthetic households) ---

needs_policyengine = pytest.mark.skipif(
    importlib.util.find_spec("policyengine_uk") is None,
    reason="policyengine_uk not available",
)


def toy_dataset(id_offset: int = 0):
    from policyengine_uk.data import UKSingleYearDataset

    regions = ["LONDON", "LONDON", "SCOTLAND", "WALES", "NORTH_WEST", "LONDON"] * 2
    tenure = ["RENT_PRIVATELY", "OWNED_OUTRIGHT", "RENT_PRIVATELY"] * 4
    n = len(regions)
    household = pd.DataFrame(
        {
            "household_id": np.arange(1, n + 1) + id_offset,
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
        for b in range(1, 3 if (h - id_offset) % 3 == 0 else 2):
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
            for r, b in zip(
                renters_after.region.astype(str), renters_after.brma.astype(str)
            )
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
    owner = (
        result.person.drop_duplicates("person_benunit_id")
        .set_index("person_benunit_id")
        .person_household_id
    )
    assert (
        result.benunit.benunit_id // 100
        == owner.loc[result.benunit.benunit_id].to_numpy()
    ).all()
    assert len(result.person) == len(dataset.person) + (k - 1) * len(
        dataset.person[
            dataset.person.person_household_id.isin(before.household_id[renter])
        ]
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
def test_groups_collapse_and_expand_without_changing_predictions(labels, areas, seed):
    rng = np.random.default_rng(seed)
    groups = RecordGroups(np.array(labels), len(labels))
    rows = rng.random((len(labels), 3))
    means = groups.means(pd.DataFrame(rows)).to_numpy()
    for g, label in enumerate(pd.unique(np.array(labels))):
        assert np.allclose(means[g], rows[np.array(labels) == label].mean(axis=0))
    w = rng.random((areas, len(groups.sizes)))
    records = groups.expand(w)
    # Equal shares within a group, conserving the group's weight.
    for g in range(len(groups.sizes)):
        m = groups.codes == g
        assert np.allclose(records[:, m], w[:, [g]] / m.sum())
    assert np.allclose(records.sum(axis=1), w.sum(axis=1))
    # Record weights on record rows predict what group weights do on mean rows.
    assert np.allclose(records @ rows, w @ means)
    totals = groups.totals(np.arange(len(labels), dtype=float))
    assert np.allclose(totals.sum(), np.arange(len(labels)).sum())


def test_identity_groups_do_nothing():
    groups = RecordGroups(None, 4)
    w = np.arange(8.0).reshape(2, 4)
    first = w[0]
    assert groups.expand(w) is w
    assert groups.totals(first) is first


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="needs torch")
def test_grouped_records_calibrate_as_one_household(tmp_path, monkeypatch):
    import h5py
    import torch

    from policyengine_uk_data.utils import calibrate as calibrate_module
    from policyengine_uk_data.utils.calibrate import calibrate_local_areas

    monkeypatch.setattr(calibrate_module, "STORAGE_FOLDER", tmp_path)

    class Data:
        def __init__(self, w):
            self.household = pd.DataFrame({"household_weight": np.asarray(w, float)})

        def copy(self):
            return Data(self.household.household_weight.to_numpy().copy())

    local = pd.DataFrame([[600.0, 300.0], [500.0, 400.0]])

    def run(matrix, national, weights, groups):
        torch.manual_seed(0)
        n = len(weights)
        result = calibrate_local_areas(
            dataset=Data(weights),
            matrix_fn=lambda d: (pd.DataFrame(matrix), local.copy(), np.ones((2, n))),
            national_matrix_fn=lambda d: (
                pd.DataFrame(national),
                pd.Series([1_500.0, 900.0]),
            ),
            area_count=2,
            weight_file="w.h5",
            dataset_key="2024",
            epochs=31,
            groups=groups,
        ).household.household_weight.to_numpy()
        with h5py.File(tmp_path / "w.h5") as f:
            saved = f["2024"][...]
        # The saved area weights are the records' weights, in float32 like the optimiser's.
        assert saved.shape == (2, n) and saved.dtype == np.float32
        assert np.allclose(saved.sum(axis=0), result, rtol=1e-5)
        return result

    # Household 0 split into three records (0-2) with different rows; a
    # zero-weight household split into two (5-6).
    split_rows = np.array(
        [
            [1.0, 0.0],
            [3.0, 0.0],
            [2.0, 0.0],
            [0.0, 1.0],
            [0.0, 2.0],
            [1.0, 1.0],
            [0.0, 3.0],
        ]
    )
    # National rows differ within the split groups too (e.g. a BRMA-dependent benefit).
    split_national = np.array(
        [
            [1.0, 0.0],
            [1.0, 4.0],
            [1.0, 2.0],
            [1.0, 1.0],
            [1.0, 0.0],
            [1.0, 3.0],
            [1.0, 1.0],
        ]
    )
    split_weights = np.array([100.0, 100.0, 100.0, 300.0, 300.0, 0.0, 0.0])
    groups = np.array([0, 0, 0, 1, 2, 3, 3])
    grouped = run(split_rows, split_national, split_weights, groups)
    # The same households unsplit, each holding its records' mean rows.
    one_rows = np.array([[2.0, 0.0], [0.0, 1.0], [0.0, 2.0], [0.5, 2.0]])
    one_national = np.array([[1.0, 2.0], [1.0, 1.0], [1.0, 0.0], [1.0, 2.0]])
    unsplit = run(one_rows, one_national, np.array([300.0, 300.0, 300.0, 0.0]), None)
    assert np.allclose(grouped[:3], unsplit[0] / 3)
    assert np.allclose(grouped[3:5], unsplit[1:3])
    assert np.allclose(grouped[5:], unsplit[3] / 2)


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="needs torch")
def test_a_group_spanning_countries_is_rejected(tmp_path, monkeypatch):
    from policyengine_uk_data.utils import calibrate as calibrate_module
    from policyengine_uk_data.utils.calibrate import calibrate_local_areas

    monkeypatch.setattr(calibrate_module, "STORAGE_FOLDER", tmp_path)

    class Data:
        def __init__(self, w):
            self.household = pd.DataFrame({"household_weight": np.asarray(w, float)})

        def copy(self):
            return Data(self.household.household_weight.to_numpy().copy())

    mask = np.array(
        [[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]
    )  # records 0 and 1 in different countries
    with pytest.raises(ValueError, match="different countries"):
        calibrate_local_areas(
            dataset=Data([1.0, 1.0, 1.0]),
            matrix_fn=lambda d: (
                pd.DataFrame(np.eye(3, 2)),
                pd.DataFrame(np.ones((2, 2))),
                mask,
            ),
            national_matrix_fn=lambda d: (
                pd.DataFrame(np.ones((3, 1))),
                pd.Series([3.0]),
            ),
            area_count=2,
            weight_file="w.h5",
            dataset_key="2024",
            epochs=1,
            groups=np.array([0, 0, 1]),
        )


@needs_policyengine
def test_split_records_keep_their_source_household(weights):
    from policyengine_uk_data.datasets.brma import split_private_renters_across_brmas

    result = split_private_renters_across_brmas(
        toy_dataset(), k=3, seed=1, weights=weights
    )
    household = result.household
    assert (household.source_household_id == household[BRMA_SPLIT_GROUP_COLUMN]).all()


@needs_policyengine
def test_a_benefit_unit_without_people_is_rejected(weights):
    from policyengine_uk.data import UKSingleYearDataset

    from policyengine_uk_data.datasets.brma import split_private_renters_across_brmas

    dataset = toy_dataset()
    benunit = pd.concat(
        [dataset.benunit, pd.DataFrame({"benunit_id": [99_999]})], ignore_index=True
    )
    broken = UKSingleYearDataset(
        person=dataset.person,
        benunit=benunit,
        household=dataset.household,
        fiscal_year=2024,
    )
    with pytest.raises(ValueError, match="no people"):
        split_private_renters_across_brmas(broken, k=2, seed=0, weights=weights)


@needs_policyengine
def test_rate_keys_match_fresh_simulations(weights):
    # The keys come from one simulation with the BRMA reset and the rate's cache
    # cleared per slot; each slot must equal a simulation built with that BRMA.
    from policyengine_uk import Microsimulation
    from policyengine_uk.data import UKSingleYearDataset

    from policyengine_uk_data.datasets.brma import lha_rate_keys

    dataset = toy_dataset()
    year = int(dataset.time_period)
    simulation = Microsimulation(dataset=dataset)
    region = dataset.household.region.astype(str).to_numpy()
    position = pd.Series(np.arange(len(region)), index=dataset.household.household_id)
    owner = (
        dataset.person.drop_duplicates("person_benunit_id")
        .set_index("person_benunit_id")
        .person_household_id
    )
    benunit_household = position.loc[owner.loc[dataset.benunit.benunit_id]].to_numpy()
    brmas, _ = household_brma_probabilities(
        region,
        benunit_household,
        np.asarray(simulation.calculate("LHA_category", year)).astype(str),
        weights,
    )
    keys = lha_rate_keys(simulation, year, brmas, region, benunit_household)
    variable = "uncapped_BRMA_LHA_rate"
    if variable not in simulation.tax_benefit_system.variables:
        variable = "BRMA_LHA_rate"
    distinct = set()
    for j in (0, 3, 7):
        household = dataset.household.copy()
        household["brma"] = [brmas[r][min(j, len(brmas[r]) - 1)] for r in region]
        fresh = Microsimulation(
            dataset=UKSingleYearDataset(
                person=dataset.person,
                benunit=dataset.benunit,
                household=household,
                fiscal_year=year,
            )
        )
        rate = np.asarray(fresh.calculate(variable, year))
        units = np.bincount(benunit_household, minlength=len(region))
        mean = (
            np.bincount(benunit_household, weights=rate, minlength=len(region)) / units
        )
        has_slot = np.array([j < len(brmas[r]) for r in region])
        assert np.allclose(keys[has_slot, j], mean[has_slot])
        assert np.isinf(keys[~has_slot, j]).all()
        distinct.add(tuple(np.round(mean[has_slot], 2)))
    assert len(distinct) == 3  # the slots really differ, so a stale cache would show


def _with_tables(dataset, **tables):
    from policyengine_uk.data import UKSingleYearDataset

    parts = {
        "person": dataset.person,
        "benunit": dataset.benunit,
        "household": dataset.household,
    }
    parts.update(tables)
    return UKSingleYearDataset(**parts, fiscal_year=dataset.time_period)


@needs_policyengine
@pytest.mark.parametrize("table", ["benunit", "household"])
def test_tables_out_of_id_order_are_rejected(table, weights):
    # PolicyEngine UK sorts entities by id but reads input columns in table
    # order, so swapping two rows would misalign every input.
    from policyengine_uk_data.datasets.brma import split_private_renters_across_brmas

    dataset = toy_dataset()
    frame = getattr(dataset, table)
    swapped = frame.iloc[[1, 0] + list(range(2, len(frame)))].reset_index(drop=True)
    with pytest.raises(ValueError, match=f"{table} table is not sorted"):
        split_private_renters_across_brmas(
            _with_tables(dataset, **{table: swapped}), k=2, seed=0, weights=weights
        )


@needs_policyengine
def test_ids_beyond_int32_split_like_small_ones(weights):
    # Benefit unit ids above 2**31 must not trip the order check.
    from policyengine_uk_data.datasets.brma import split_private_renters_across_brmas

    offset = 30_000_000
    small = split_private_renters_across_brmas(
        toy_dataset(), k=3, seed=5, weights=weights
    )
    large = split_private_renters_across_brmas(
        toy_dataset(offset), k=3, seed=5, weights=weights
    )
    assert large.benunit.benunit_id.max() > 2**31
    assert len(large.household) == len(small.household)
    assert (
        large.household.brma.astype(str).to_numpy()
        == small.household.brma.astype(str).to_numpy()
    ).all()


@needs_policyengine
def test_encoded_tenure_splits_the_same_renters(weights):
    # A tenure column holding enum codes must select the same private renters
    # as one holding names.
    from policyengine_uk.variables.household.demographic.tenure_type import (
        TenureType,
    )

    from policyengine_uk_data.datasets.brma import split_private_renters_across_brmas

    dataset = toy_dataset()
    household = dataset.household.copy()
    members = list(TenureType)
    household["tenure_type"] = [
        members.index(TenureType[name]) for name in household.tenure_type.astype(str)
    ]
    assert pd.api.types.is_integer_dtype(household.tenure_type)
    named = split_private_renters_across_brmas(dataset, k=2, seed=4, weights=weights)
    coded = split_private_renters_across_brmas(
        _with_tables(dataset, household=household), k=2, seed=4, weights=weights
    )
    assert len(named.household) > len(dataset.household)
    assert len(coded.household) == len(named.household)
    assert (
        coded.household.brma.astype(str).to_numpy()
        == named.household.brma.astype(str).to_numpy()
    ).all()


@settings(max_examples=200, deadline=None, derandomize=True)
@given(
    st.lists(st.integers(0, 3), min_size=1, max_size=10),
    st.integers(1, 4),
    st.integers(1, 3),
    st.integers(0, 2**32 - 1),
)
def test_column_agreement_is_exact(labels, rows, chunk, seed):
    rng = np.random.default_rng(seed)
    groups = RecordGroups(np.array(labels), len(labels))
    matrix = (rng.random((rows, len(labels))) < 0.5).astype(float)
    if rng.random() < 0.5:  # often make the groups agree
        matrix = matrix[:, groups.first[groups.codes]]
    expected = all(
        np.array_equal(matrix[:, i], matrix[:, j])
        for i in range(len(labels))
        for j in range(len(labels))
        if labels[i] == labels[j]
    )
    assert groups.columns_agree(matrix, chunk=chunk) == expected


def test_one_hot_area_masks_that_differ_never_agree():
    # Two records of a group placed in different single areas of 650 (masks
    # whose seeded random projections nearly coincide) must be told apart.
    mask = np.zeros((650, 2))
    mask[506, 0] = mask[173, 1] = 1.0
    assert not RecordGroups(np.array([0, 0]), 2).columns_agree(mask)
    assert RecordGroups(np.array([0, 1]), 2).columns_agree(mask)


@needs_policyengine
def test_a_dataset_without_private_renters_is_returned_unsplit(weights):
    from policyengine_uk_data.datasets.brma import split_private_renters_across_brmas

    dataset = toy_dataset()
    household = dataset.household.copy()
    household["tenure_type"] = "OWNED_OUTRIGHT"
    owners = _with_tables(dataset, household=household)
    result = split_private_renters_across_brmas(owners, k=3, seed=0, weights=weights)
    assert len(result.household) == len(household)
    assert (result.household[BRMA_SPLIT_GROUP_COLUMN] == household.household_id).all()
    assert (result.household.brma.astype(str) == "INNER_NORTH_LONDON").all()


@needs_policyengine
@pytest.mark.parametrize("problem", ["no_tenure", "duplicate_household"])
def test_split_refuses_ambiguous_tables(problem, weights):
    from policyengine_uk_data.datasets.brma import split_private_renters_across_brmas

    dataset = toy_dataset()
    household = dataset.household.copy()
    if problem == "no_tenure":
        household = household.drop(columns="tenure_type")
        match = "needs tenure_type"
    else:
        household.loc[1, "household_id"] = household.household_id[0]
        match = "duplicate household_id"
    with pytest.raises(ValueError, match=match):
        split_private_renters_across_brmas(
            _with_tables(dataset, household=household), k=2, seed=0, weights=weights
        )
