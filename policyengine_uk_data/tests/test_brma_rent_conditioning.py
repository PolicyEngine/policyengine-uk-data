"""Rent-conditioned BRMA assignment for private renters.

Invariants of ``brma_probabilities`` and ``draw_brmas``, for any census table,
list-of-rents table, model and households with data for their cells:

1. Support: each row of probabilities sums to 1 and is zero outside the
   household's region × bedrooms cell, and a draw never leaves that support.
2. Determinism: the same generator state gives the same BRMAs.
3. Fail closed: a household without a positive rent, a known region × bedrooms
   cell with households in it, a list of rents for every BRMA in that cell, or
   a region shift raises; so does a census table that mixes ``all`` and banded
   rows in one region.
4. Level invariance: scaling a region's rents (or its BRMAs' list medians) by
   a constant and moving the region's shift to match changes nothing, so the
   list of rents needs no further uprating once each region has its own shift.
5. No information, no change: if a cell's BRMAs have the same list of rents,
   the probabilities are the census shares.
6. Agreement with a scalar reference implementation of the same formula,
   computed in log space so it never underflows.
7. Dearer rents, dearer BRMAs: without the below-market component, the
   probability of the dearer of two otherwise equal BRMAs rises with rent.
   With it, the probability is not monotone: rents well below the list move
   back towards the census shares, and implausibly low ones towards the
   cheapest BRMA again. Intended: such rents say little about location.
8. Calibration: when rents come from the model, households' probabilities
   average to the census shares, and the fit recovers the model.
9. Region isolation: a household's probabilities depend only on its own
   region's census rows, also when a BRMA spans two regions.
10. The fit's likelihood is the same with and without dropping each
    household's unsupported BRMAs, and weights act as frequency weights: weight
    2 equals a duplicate household and weight 0 equals leaving it out.
"""

import math

import numpy as np
import pandas as pd
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from policyengine_uk.variables.household.demographic.locations import BRMAName

from policyengine_uk.data import UKSingleYearDataset

from policyengine_uk_data.datasets.brma import (
    BEDROOM_BAND_CATEGORY,
    BRMA_HOUSEHOLDS_PATH,
    BRMA_RENTS_PATH,
    ReportedRentModel,
    assign_private_renter_brmas,
    bedroom_band,
    brma_probabilities,
    draw_brmas,
    _fit_inputs,
    _negative_log_likelihood,
    fit_reported_rent_model,
)
from policyengine_uk_data.datasets.frs_release import CURRENT_FRS_RELEASE
from policyengine_uk_data.storage import STORAGE_FOLDER

REGIONS = ["LONDON", "WALES", "SCOTLAND", "NORTHERN_IRELAND"]
BANDS = list(BEDROOM_BAND_CATEGORY)
CATEGORIES = ["A", "B", "C", "D", "E"]


@pytest.fixture(scope="module")
def households():
    return pd.read_csv(BRMA_HOUSEHOLDS_PATH, dtype={"bedrooms": str})


@pytest.fixture(scope="module")
def rents():
    return pd.read_csv(BRMA_RENTS_PATH)


def test_rents_table_covers_every_brma_and_category(rents):
    assert list(rents.columns)[:4] == [
        "brma",
        "lha_category",
        "median_weekly_rent",
        "log_sd",
    ]
    assert not rents.duplicated(["brma", "lha_category"]).any()
    assert set(rents.brma) == set(BRMAName.__members__)
    assert (rents.groupby("brma").lha_category.agg(set) == set(CATEGORIES)).all()
    assert rents.median_weekly_rent.between(40, 2_000).all()
    assert rents.log_sd.between(0.03, 0.6).all()


def test_each_nation_uses_its_best_source(households, rents):
    by_region = households.groupby(["brma", "region"]).households.sum().reset_index()
    region = by_region.sort_values("households").groupby("brma").region.last()
    nation = region.where(
        region.isin(["WALES", "SCOTLAND", "NORTHERN_IRELAND"]), "ENGLAND"
    )
    basis = rents.groupby("brma").basis.agg(set).map(lambda b: b.pop())
    assert basis.groupby(nation).agg(set).to_dict() == {
        "ENGLAND": {"list"},
        "WALES": {"list"},
        "SCOTLAND": {"30th percentile and list spread"},
        "NORTHERN_IRELAND": {"30th percentile and typical spread"},
    }
    # Every cell with its own spread has enough rents to estimate it.
    assert (
        rents[rents.basis != "30th percentile and typical spread"].rents >= 25
    ).all()


def test_larger_homes_cost_more_in_nearly_every_brma(rents):
    median = rents.pivot(
        index="brma", columns="lha_category", values="median_weekly_rent"
    )
    for small, large in zip("ABCD", "BCDE"):
        assert (median[large] > median[small]).mean() > 0.97, (small, large)


def test_bedroom_bands():
    assert (
        bedroom_band([1, 2, 3, 4, 5, 6]).tolist() == ["1", "2", "3", "4+"] + ["4+"] * 2
    )
    for bad in (0, -1, np.nan):
        with pytest.raises(ValueError, match="at least one bedroom"):
            bedroom_band([2, bad])


def test_bedroom_bands_map_to_their_list_categories():
    # Census bands count a home's bedrooms; list categories B-E are
    # self-contained homes with one to four or more bedrooms.
    assert BEDROOM_BAND_CATEGORY == {"1": "B", "2": "C", "3": "D", "4+": "E"}


@st.composite
def tables(draw):
    """A census table and a list-of-rents table for a few made-up BRMAs."""
    regions = draw(
        st.lists(st.sampled_from(REGIONS), min_size=1, max_size=3, unique=True)
    )
    count = draw(st.integers(1, 4))
    # Like the 37 real BRMAs that cross a region boundary, a BRMA may be named
    # in more than one region; zero-household rows may be kept.
    shared = draw(st.booleans())
    keep_zero = draw(st.booleans())

    def name(region, i):
        return f"SHARED_{i}" if shared else f"{region}_{i}"

    census, names = [], []
    for region in regions:
        # Northern Ireland's census has one band, ``all``.
        for band in ["all"] if region == "NORTHERN_IRELAND" else BANDS:
            sizes = [draw(st.sampled_from([0, 1, 7, 1000])) for _ in range(count)]
            sizes[draw(st.integers(0, count - 1))] = 5  # a non-empty cell
            census += [
                (region, name(region, i), band, size)
                for i, size in enumerate(sizes)
                if size or keep_zero
            ]
        names += [name(region, i) for i in range(count)]
    lists = [
        (brma, category, draw(st.floats(40, 2_000)), draw(st.floats(0.05, 0.6)))
        for brma in dict.fromkeys(names)
        for category in BEDROOM_BAND_CATEGORY.values()
    ]
    return (
        pd.DataFrame(census, columns=["region", "brma", "bedrooms", "households"]),
        pd.DataFrame(
            lists, columns=["brma", "lha_category", "median_weekly_rent", "log_sd"]
        ),
    )


@st.composite
def cases(draw, below_market=True):
    census, lists = draw(tables())
    regions = sorted(census.region.unique())
    n = draw(st.integers(1, 25))
    region = np.array([draw(st.sampled_from(regions)) for _ in range(n)])
    bedrooms = np.array([draw(st.integers(1, 6)) for _ in range(n)])
    rent = np.array([draw(st.floats(1, 5_000, allow_nan=False)) for _ in range(n)])
    noise = draw(st.floats(0.01, 1))
    model = ReportedRentModel(
        region_shift={r: draw(st.floats(-1, 1)) for r in regions},
        noise=noise,
        below_market_share=draw(st.floats(0.001, 0.5)) if below_market else 0.0,
        below_market_discount=draw(st.floats(0, 2)),
        below_market_noise=noise + draw(st.floats(0.05, 2)),
    )
    return census, lists, region, bedrooms, rent, model


def supported(census, region, band, brma):
    cell = census[
        (census.region == region)
        & census.bedrooms.isin([band, "all"])
        & (census.households > 0)
    ]
    return brma in set(cell.brma)


@settings(max_examples=200, deadline=None, derandomize=True)
@given(cases(), st.integers(0, 2**32 - 1))
def test_probabilities_and_draws_stay_on_the_census_support(case, seed):
    census, lists, region, bedrooms, rent, model = case
    brmas, p = brma_probabilities(region, bedrooms, rent, model, census, lists)
    assert p.shape == (len(region), len(brmas))
    assert np.isfinite(p).all() and (p >= 0).all()
    assert np.allclose(p.sum(axis=1), 1)
    band = bedroom_band(bedrooms)
    for i, j in zip(*np.nonzero(p)):
        assert supported(census, region[i], band[i], brmas[j])
    first = draw_brmas(brmas, p, np.random.default_rng(seed))
    second = draw_brmas(brmas, p, np.random.default_rng(seed))
    assert list(first) == list(second)
    for i, brma in enumerate(first):
        assert p[i, list(brmas).index(brma)] > 0
        assert supported(census, region[i], band[i], brma)


@settings(max_examples=100, deadline=None, derandomize=True)
@given(
    cases(),
    st.sampled_from(["rent", "cell", "empty", "mixed", "list", "shift", "bedrooms"]),
)
def test_missing_inputs_fail_closed(case, what):
    census, lists, region, bedrooms, rent, model = case
    if what == "rent":
        rent = rent.copy()
        rent[0] = [0.0, -5.0, np.nan][len(rent) % 3]
        match = "positive weekly rent"
    elif what == "cell":
        census = census[census.region != region[0]]
        assume(len(census))
        match = "No BRMA weights"
    elif what == "empty":
        band = bedroom_band(bedrooms)[0]
        in_cell = (census.region == region[0]) & census.bedrooms.isin([band, "all"])
        census = census.assign(households=census.households.where(~in_cell, 0))
        match = "No BRMA weights"
    elif what == "mixed":
        banded = census[census.bedrooms != "all"]
        assume(len(banded))
        extra = banded.iloc[[0]].assign(bedrooms="all")
        census = pd.concat([census, extra], ignore_index=True)
        match = "both `all` and banded rows"
    elif what == "list":
        band = bedroom_band(bedrooms)[0]
        cell = census[
            (census.region == region[0])
            & census.bedrooms.isin([band, "all"])
            & (census.households > 0)
        ]
        lists = lists[
            ~(
                (lists.brma == cell.brma.iloc[0])
                & (lists.lha_category == BEDROOM_BAND_CATEGORY[band])
            )
        ]
        match = "No list of rents"
    elif what == "shift":
        shifts = {r: s for r, s in model.region_shift.items() if r != region[0]}
        model = ReportedRentModel(shifts, *list(model.__dict__.values())[1:])
        match = "No reported-rent shift"
    else:
        bedrooms = bedrooms.astype(float)
        bedrooms[0] = 0
        match = "at least one bedroom"
    with pytest.raises(ValueError, match=match):
        brma_probabilities(region, bedrooms, rent, model, census, lists)


@settings(max_examples=100, deadline=None, derandomize=True)
@given(cases(), st.floats(0.25, 4))
def test_a_regions_rent_level_is_absorbed_by_its_shift(case, factor):
    census, lists, region, bedrooms, rent, model = case
    _, base = brma_probabilities(region, bedrooms, rent, model, census, lists)
    target = region[0]
    shifted = ReportedRentModel(
        {
            r: s + (math.log(factor) if r == target else 0)
            for r, s in model.region_shift.items()
        },
        *list(model.__dict__.values())[1:],
    )
    scaled_rent = np.where(region == target, rent * factor, rent)
    _, a = brma_probabilities(region, bedrooms, scaled_rent, shifted, census, lists)
    assert np.allclose(a, base, atol=1e-9)
    # Equivalently, uprate the region's lists and lower its shift, when no
    # other region shares its BRMAs.
    own = set(census.brma[census.region == target])
    assume(not own & set(census.brma[census.region != target]))
    uprated = lists.copy()
    in_target = uprated.brma.isin(own)
    uprated.loc[in_target, "median_weekly_rent"] *= factor
    lowered = ReportedRentModel(
        {
            r: s - (math.log(factor) if r == target else 0)
            for r, s in model.region_shift.items()
        },
        *list(model.__dict__.values())[1:],
    )
    _, b = brma_probabilities(region, bedrooms, rent, lowered, census, uprated)
    assert np.allclose(b, base, atol=1e-9)


@settings(max_examples=100, deadline=None, derandomize=True)
@given(cases(), st.floats(40, 2_000), st.floats(0.05, 0.6))
def test_identical_lists_leave_the_census_shares(case, median, log_sd):
    census, lists, region, bedrooms, rent, model = case
    lists = lists.assign(median_weekly_rent=median, log_sd=log_sd)
    brmas, p = brma_probabilities(region, bedrooms, rent, model, census, lists)
    band = bedroom_band(bedrooms)
    for i in range(len(region)):
        cell = census[
            (census.region == region[i]) & census.bedrooms.isin([band[i], "all"])
        ]
        share = cell.set_index("brma").households / cell.households.sum()
        assert np.allclose(p[i], share.reindex(brmas, fill_value=0).to_numpy())


def reference_probabilities(census, lists, region, band, rent, model):
    """The same formula for one household, written out with scalars in log space."""
    cell = census[(census.region == region) & census.bedrooms.isin([band, "all"])]
    cell = cell[cell.households > 0]
    log_terms = {}
    for _, row in cell.iterrows():
        entry = lists[
            (lists.brma == row.brma)
            & (lists.lha_category == BEDROOM_BAND_CATEGORY[band])
        ].iloc[0]
        error = (
            math.log(rent)
            - model.region_shift[region]
            - math.log(entry.median_weekly_rent)
        )

        def log_normal(x, variance):
            return -0.5 * x * x / variance - 0.5 * math.log(variance)

        main = math.log1p(-model.below_market_share) + log_normal(
            error, entry.log_sd**2 + model.noise**2
        )
        below = math.log(model.below_market_share) + log_normal(
            error + model.below_market_discount,
            entry.log_sd**2 + model.below_market_noise**2,
        )
        top = max(main, below)
        log_density = top + math.log(math.exp(main - top) + math.exp(below - top))
        log_terms[row.brma] = math.log(row.households) + log_density
    top = max(log_terms.values())
    total = sum(math.exp(v - top) for v in log_terms.values())
    return {brma: math.exp(v - top) / total for brma, v in log_terms.items()}


def test_reference_and_probabilities_survive_rents_far_from_every_list():
    # Both mixture components underflow in linear space here (exponents of
    # about -1,683 and -888); review of uk-data#521 found the old scalar
    # reference divided by zero on this case.
    census = pd.DataFrame(
        [("LONDON", "LONDON_0", "2", 5), ("LONDON", "LONDON_1", "2", 3)],
        columns=["region", "brma", "bedrooms", "households"],
    )
    lists = pd.DataFrame(
        [("LONDON_0", "C", 40.0, 0.05), ("LONDON_1", "C", 45.0, 0.05)],
        columns=["brma", "lha_category", "median_weekly_rent", "log_sd"],
    )
    model = ReportedRentModel({"LONDON": 0.5304839057541509}, 0.01, 0.001, 1 / 3, 0.06)
    for rent in (1_309.28, 1.0):
        brmas, p = brma_probabilities(
            np.array(["LONDON"]), np.array([2]), np.array([rent]), model, census, lists
        )
        want = reference_probabilities(census, lists, "LONDON", "2", rent, model)
        assert np.isfinite(p).all() and p.sum() == pytest.approx(1)
        for brma, value in want.items():
            assert p[0, list(brmas).index(brma)] == pytest.approx(value, abs=1e-12)


@settings(max_examples=100, deadline=None, derandomize=True)
@given(cases())
def test_probabilities_match_the_scalar_reference(case):
    census, lists, region, bedrooms, rent, model = case
    brmas, p = brma_probabilities(region, bedrooms, rent, model, census, lists)
    band = bedroom_band(bedrooms)
    for i in range(len(region)):
        want = reference_probabilities(
            census, lists, region[i], band[i], rent[i], model
        )
        got = dict(zip(brmas, p[i]))
        for brma, value in want.items():
            assert got[brma] == pytest.approx(value, rel=1e-6, abs=1e-12)
        assert sum(got[b] for b in want) == pytest.approx(1)


@settings(max_examples=100, deadline=None, derandomize=True)
@given(
    st.floats(60, 400),
    st.floats(1.05, 3),
    st.floats(0.05, 0.6),
    st.floats(0.01, 1),
    st.lists(st.floats(20, 3_000), min_size=2, max_size=12),
)
def test_higher_rents_favour_the_dearer_brma(median, premium, log_sd, noise, rent):
    census = pd.DataFrame(
        [("WALES", "CHEAP", "2", 300), ("WALES", "DEAR", "2", 100)],
        columns=["region", "brma", "bedrooms", "households"],
    )
    lists = pd.DataFrame(
        [("CHEAP", "C", median, log_sd), ("DEAR", "C", median * premium, log_sd)],
        columns=["brma", "lha_category", "median_weekly_rent", "log_sd"],
    )
    model = ReportedRentModel({"WALES": 0.0}, noise, 0.0, 0.5, noise + 0.5)
    rent = np.sort(np.array(rent))
    brmas, p = brma_probabilities(
        np.full(len(rent), "WALES"), np.full(len(rent), 2), rent, model, census, lists
    )
    dear = p[:, list(brmas).index("DEAR")]
    assert (np.diff(dear) >= -1e-12).all()


@settings(max_examples=100, deadline=None, derandomize=True)
@given(cases(), st.floats(0.01, 100))
def test_a_households_probabilities_use_only_its_regions_census_rows(case, factor):
    census, lists, region, bedrooms, rent, model = case
    _, base = brma_probabilities(region, bedrooms, rent, model, census, lists)
    target = region[0]
    others = census.region != target
    # Rescale every other region's households, including BRMAs it shares.
    rescaled = census.assign(
        households=census.households.where(~others, census.households * factor)
    )
    _, p = brma_probabilities(region, bedrooms, rent, model, rescaled, lists)
    rows = region == target
    assert np.allclose(p[rows], base[rows], atol=1e-12)


@st.composite
def parameters(draw, regions):
    """A parameter vector for the fit's likelihood: shifts, then the shape."""
    return np.array(
        [draw(st.floats(-1, 1)) for _ in regions]
        + [
            draw(st.floats(0.01, 1)),
            draw(st.floats(0.001, 0.5)),
            draw(st.floats(0, 2)),
            draw(st.floats(0.05, 2)),
        ]
    )


@settings(max_examples=100, deadline=None, derandomize=True)
@given(cases(), st.data())
def test_dropping_unsupported_brmas_leaves_the_likelihood_unchanged(case, data):
    census, lists, region, bedrooms, rent, _ = case
    weight = np.array([data.draw(st.floats(0.1, 10)) for _ in region])
    args = (region, bedrooms, rent, weight, census, lists)
    names, compact = _fit_inputs(*args)
    _, full = _fit_inputs(*args, compact=False)
    assert compact["prior"].shape[1] <= full["prior"].shape[1]
    x = data.draw(parameters(names))
    assert _negative_log_likelihood(x, **compact) == pytest.approx(
        _negative_log_likelihood(x, **full), rel=1e-12, abs=1e-12
    )


@settings(max_examples=100, deadline=None, derandomize=True)
@given(cases(), st.data())
def test_weights_act_as_frequency_weights(case, data):
    census, lists, region, bedrooms, rent, _ = case
    n = len(region)
    weight = np.array([data.draw(st.floats(0.1, 10)) for _ in region])
    k = data.draw(st.integers(0, n - 1))
    names, base = _fit_inputs(region, bedrooms, rent, weight, census, lists)
    x = data.draw(parameters(names))

    def loss(rows, w):
        _, inputs = _fit_inputs(
            region[rows], bedrooms[rows], rent[rows], w, census, lists
        )
        return _negative_log_likelihood(x, **inputs)

    # Doubling a household's weight is the same as listing it twice.
    doubled = weight.copy()
    doubled[k] *= 2
    twice = np.r_[np.arange(n), k]
    assert loss(np.arange(n), doubled) == pytest.approx(
        loss(twice, weight[twice]), rel=1e-12
    )
    # A zero weight is the same as leaving the household out, when its region
    # keeps other households (so the regions' shifts line up).
    rest = np.delete(np.arange(n), k)
    assume(len(rest) and set(region[rest]) == set(region))
    zeroed = weight.copy()
    zeroed[k] = 0
    assert loss(np.arange(n), zeroed) == pytest.approx(
        loss(rest, weight[rest]), rel=1e-12
    )


def simulate(households, rents, model, n, seed):
    """Households drawn from the census, with rents drawn from ``model``."""
    rng = np.random.default_rng(seed)
    banded = households[households.bedrooms != "all"]
    cells = banded.sample(n, weights="households", replace=True, random_state=rng)
    bedrooms = cells.bedrooms.map({"1": 1, "2": 2, "3": 3, "4+": 5}).to_numpy()
    lists = rents.set_index(["brma", "lha_category"])
    key = list(zip(cells.brma, cells.bedrooms.map(BEDROOM_BAND_CATEGORY)))
    below = rng.random(n) < model.below_market_share
    log_rent = (
        np.log(lists.median_weekly_rent.loc[key].to_numpy())
        + lists.log_sd.loc[key].to_numpy() * rng.standard_normal(n)
        + cells.region.map(model.region_shift).to_numpy()
        + np.where(
            below,
            -model.below_market_discount
            + model.below_market_noise * rng.standard_normal(n),
            model.noise * rng.standard_normal(n),
        )
    )
    return cells.region.to_numpy(), bedrooms, np.exp(log_rent), cells.brma.to_numpy()


def test_model_draws_average_to_the_census_and_the_fit_recovers_the_model(
    households, rents
):
    regions = sorted(set(households.region) - {"NORTHERN_IRELAND"})
    truth = ReportedRentModel(
        {r: -0.05 - 0.01 * i for i, r in enumerate(regions)}, 0.13, 0.16, 0.6, 0.7
    )
    region, bedrooms, rent, true_brma = simulate(households, rents, truth, 40_000, 3)
    weight = np.ones(len(rent))
    fitted = fit_reported_rent_model(region, bedrooms, rent, weight)
    assert fitted.noise == pytest.approx(truth.noise, abs=0.03)
    assert fitted.below_market_share == pytest.approx(
        truth.below_market_share, abs=0.03
    )
    assert fitted.below_market_discount == pytest.approx(0.6, abs=0.1)
    assert fitted.below_market_noise == pytest.approx(0.7, abs=0.1)
    for r in regions:
        assert fitted.region_shift[r] == pytest.approx(truth.region_shift[r], abs=0.03)
    brmas, p = brma_probabilities(region, bedrooms, rent, truth)
    drawn = draw_brmas(brmas, p, np.random.default_rng(4))
    medians = rents[rents.lha_category == "C"].set_index("brma").median_weekly_rent
    for r in regions:
        rows = region == r
        expected = p[rows].mean(axis=0)
        actual = pd.Series(true_brma[rows]).value_counts(normalize=True)
        actual = actual.reindex(brmas, fill_value=0).to_numpy()
        # Probabilities average to where the simulated households really live.
        assert 0.5 * np.abs(expected - actual).sum() < 0.06, r
        # The drawn BRMAs' rent levels track rents as the true BRMAs' do.
        want = np.corrcoef(np.log(rent[rows]), np.log(medians[true_brma[rows]]))[0, 1]
        got = np.corrcoef(np.log(rent[rows]), np.log(medians[drawn[rows]]))[0, 1]
        assert got == pytest.approx(want, abs=0.06), r


def test_assignment_is_deterministic_and_fit_needs_valid_inputs(households, rents):
    truth = ReportedRentModel(
        {r: -0.1 for r in households.region.unique()}, 0.15, 0.1, 0.5, 0.6
    )
    region, bedrooms, rent, _ = simulate(households, rents, truth, 3_000, 5)
    weight = np.linspace(1, 2, len(rent))
    first = assign_private_renter_brmas(
        region, bedrooms, rent, weight, np.random.default_rng(0)
    )
    second = assign_private_renter_brmas(
        region, bedrooms, rent, weight, np.random.default_rng(0)
    )
    assert list(first) == list(second)
    with pytest.raises(ValueError, match="positive and weights non-negative"):
        fit_reported_rent_model(region, bedrooms, rent, -weight)
    with pytest.raises(ValueError, match="positive and weights non-negative"):
        fit_reported_rent_model(region, bedrooms, np.zeros(len(rent)), weight)


@pytest.fixture(scope="module")
def built_renters():
    """Private renters who report a rent, from the raw survey and the built FRS.

    The dataset does not keep bedrooms, so they are read from the raw household
    table, as the build reads them.
    """
    raw_path = STORAGE_FOLDER / CURRENT_FRS_RELEASE.name / "househol.tab"
    built_path = STORAGE_FOLDER / CURRENT_FRS_RELEASE.base_dataset_file
    if not (raw_path.exists() and built_path.exists()):
        pytest.skip("Raw FRS household table or built FRS dataset not available")
    raw = pd.read_csv(raw_path, sep="\t", low_memory=False)
    raw.columns = raw.columns.str.lower()
    raw = raw[["sernum", "ptentyp2", "hhrent", "bedroom6", "gross4"]]
    raw = raw.apply(pd.to_numeric, errors="coerce")
    raw = raw[raw.ptentyp2.isin([3, 4]) & (raw.hhrent > 0)]
    built = UKSingleYearDataset(built_path).household.set_index("household_id")
    built = built.loc[raw.sernum]
    assert (built.tenure_type.astype(str) == "RENT_PRIVATELY").all()
    renters = pd.DataFrame(
        {
            "region": built.region.astype(str).to_numpy(),
            "bedrooms": raw.bedroom6.to_numpy(),
            "weekly_rent": raw.hhrent.to_numpy(),
            "weight": raw.gross4.to_numpy(),
            "brma": built.brma.astype(str).to_numpy(),
        }
    )
    args = (renters.region, renters.bedrooms, renters.weekly_rent)
    model = fit_reported_rent_model(*args, renters.weight)
    brmas, p = brma_probabilities(*args, model)
    flat = ReportedRentModel(model.region_shift, 50.0, 0.0, 0.0, 51.0)
    _, census = brma_probabilities(*args, flat)
    return renters, brmas, p, census


def test_built_private_renters_match_the_census_within_each_region(built_renters):
    """Within each region, the built private renters' weighted mean probability
    of each BRMA stays within 5 percentage points (total variation) of the
    census shares for their homes' bedroom bands."""
    renters, brmas, p, census = built_renters
    for region in renters.region.unique():
        rows = (renters.region == region).to_numpy()
        w = renters.weight[rows].to_numpy()
        w = w / w.sum()
        gap = 0.5 * np.abs(w @ p[rows] - w @ census[rows]).sum()
        assert gap < 0.05, (region, gap)


def test_built_private_renters_brmas_are_draws_from_their_probabilities(
    built_renters,
):
    """The BRMAs in the built dataset look like independent draws from the
    probabilities, scored by their log probability.

    For independent draws the log score's mean and variance are exact sums
    over households, so the standardised score is close to normal. A draw
    from the census shares instead (uk-data#516's) scores about -19.
    """
    renters, brmas, p, _ = built_renters
    drawn = pd.Index(brmas).get_indexer(renters.brma)
    rows = np.arange(len(renters))
    assert (p[rows, drawn] > 0).all()
    log_p = np.log(np.where(p > 0, p, 1))
    score = np.log(p[rows, drawn]).sum()
    expected = (p * log_p).sum()
    variance = ((p * log_p**2).sum(axis=1) - (p * log_p).sum(axis=1) ** 2).sum()
    assert abs(score - expected) / np.sqrt(variance) < 4


def test_built_private_renters_rents_track_their_brmas_rent_levels(
    built_renters, rents
):
    """Dearer rents sit in dearer BRMAs, within region and bedroom band."""
    renters, *_ = built_renters
    category = pd.Series(bedroom_band(renters.bedrooms)).map(BEDROOM_BAND_CATEGORY)
    listed = rents.set_index(["brma", "lha_category"]).median_weekly_rent
    frame = pd.DataFrame(
        {
            "cell": renters.region + category.to_numpy(),
            "rent": np.log(renters.weekly_rent),
            "level": np.log(listed.loc[list(zip(renters.brma, category))].to_numpy()),
            "w": renters.weight,
        }
    )
    for column in ("rent", "level"):
        mean = frame.groupby("cell").apply(
            lambda g: np.average(g[column], weights=g.w), include_groups=False
        )
        frame[column] -= frame.cell.map(mean)
    covariance = np.average(frame.rent * frame.level, weights=frame.w)
    correlation = covariance / np.sqrt(
        np.average(frame.rent**2, weights=frame.w)
        * np.average(frame.level**2, weights=frame.w)
    )
    assert correlation > 0.15, correlation
