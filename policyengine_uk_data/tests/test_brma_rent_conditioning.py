"""Rent-conditioned BRMA assignment for private renters.

Invariants of ``brma_probabilities`` and ``draw_brmas``, for any census table,
list-of-rents table, model and households with data for their cells:

1. Support: each row of probabilities sums to 1 and is zero outside the
   household's region × bedrooms cell, and a draw never leaves that support.
2. Determinism: the same generator state gives the same BRMAs.
3. Fail closed: a household without a positive rent, a known region × bedrooms
   cell, a list of rents for every BRMA in that cell, or a region shift raises.
4. Level invariance: scaling a region's rents (or its BRMAs' list medians) by
   a constant and moving the region's shift to match changes nothing, so the
   list of rents needs no further uprating once each region has its own shift.
5. No information, no change: if a cell's BRMAs have the same list of rents,
   the probabilities are the census shares.
6. Agreement with a scalar reference implementation of the same formula.
7. Dearer rents, dearer BRMAs: without the below-market component, the
   probability of the dearer of two otherwise equal BRMAs rises with rent.
   (With it, very low rents revert towards the census shares: intended.)
8. Calibration: when rents come from the model, households' probabilities
   average to the census shares, and the fit recovers the model.
"""

import math

import numpy as np
import pandas as pd
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from policyengine_uk.variables.household.demographic.locations import BRMAName

from policyengine_uk_data.datasets.brma import (
    BEDROOM_BAND_CATEGORY,
    BRMA_HOUSEHOLDS_PATH,
    BRMA_RENTS_PATH,
    ReportedRentModel,
    assign_private_renter_brmas,
    bedroom_band,
    brma_probabilities,
    draw_brmas,
    fit_reported_rent_model,
)

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


@st.composite
def tables(draw):
    """A census table and a list-of-rents table for a few made-up BRMAs."""
    regions = draw(
        st.lists(st.sampled_from(REGIONS), min_size=1, max_size=3, unique=True)
    )
    count = draw(st.integers(1, 4))
    census, lists = [], []
    for region in regions:
        # Northern Ireland's census has one band, ``all``.
        for band in ["all"] if region == "NORTHERN_IRELAND" else BANDS:
            sizes = [draw(st.sampled_from([0, 1, 7, 1000])) for _ in range(count)]
            sizes[draw(st.integers(0, count - 1))] = 5  # a non-empty cell
            census += [
                (region, f"{region}_{i}", band, size)
                for i, size in enumerate(sizes)
                if size
            ]
        for i in range(count):
            for category in BEDROOM_BAND_CATEGORY.values():
                median = draw(st.floats(40, 2_000, allow_nan=False))
                lists.append(
                    (f"{region}_{i}", category, median, draw(st.floats(0.05, 0.6)))
                )
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
@given(cases(), st.sampled_from(["rent", "cell", "list", "shift", "bedrooms"]))
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
    elif what == "list":
        band = bedroom_band(bedrooms)[0]
        cell = census[
            (census.region == region[0]) & census.bedrooms.isin([band, "all"])
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
    # Equivalently, uprate the region's lists and lower its shift.
    uprated = lists.copy()
    in_target = uprated.brma.str.startswith(target)
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
    """The same formula for one household, written out with scalars."""
    cell = census[(census.region == region) & census.bedrooms.isin([band, "all"])]
    result = {}
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

        def normal(x, variance):
            return math.exp(-0.5 * x * x / variance) / math.sqrt(variance)

        density = (1 - model.below_market_share) * normal(
            error, entry.log_sd**2 + model.noise**2
        ) + model.below_market_share * normal(
            error + model.below_market_discount,
            entry.log_sd**2 + model.below_market_noise**2,
        )
        result[row.brma] = row.households * density
    total = sum(result.values())
    return {brma: value / total for brma, value in result.items()}


@settings(max_examples=100, deadline=None, derandomize=True)
@given(cases())
def test_probabilities_match_the_scalar_reference(case):
    census, lists, region, bedrooms, rent, model = case
    # Keep rents where the scalar densities do not underflow.
    rent = np.clip(rent, 30, 3_000)
    brmas, p = brma_probabilities(region, bedrooms, rent, model, census, lists)
    band = bedroom_band(bedrooms)
    for i in range(len(region)):
        want = reference_probabilities(
            census, lists, region[i], band[i], rent[i], model
        )
        assume(all(math.isfinite(v) for v in want.values()))
        got = dict(zip(brmas, p[i]))
        for brma, value in want.items():
            assert got[brma] == pytest.approx(value, rel=1e-6, abs=1e-12)


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


def private_renters(dataset):
    household = dataset.household
    rows = (
        (household.tenure_type.astype(str) == "RENT_PRIVATELY")
        & (household.rent > 0)
        & (household.household_weight > 0)
    )
    return household[rows]


def test_built_private_renters_match_the_census_within_each_region(
    enhanced_frs, households
):
    """Calibration check on the built dataset.

    The rent model is refitted to the built private renters; their weighted
    mean probability of each BRMA must stay within 5 percentage points (total
    variation) of the census shares of their region and bedroom bands, and the
    BRMAs actually drawn must lie in those cells.
    """
    renters = private_renters(enhanced_frs)
    region = renters.region.astype(str).to_numpy()
    bedrooms = renters.num_bedrooms.to_numpy()
    weekly_rent = renters.rent.to_numpy() / 52
    weight = renters.household_weight.to_numpy()
    model = fit_reported_rent_model(region, bedrooms, weekly_rent, weight)
    brmas, p = brma_probabilities(region, bedrooms, weekly_rent, model)
    flat = ReportedRentModel(model.region_shift, 50.0, 0.0, 0.0, 51.0)
    _, census = brma_probabilities(region, bedrooms, weekly_rent, flat)
    position = {brma: j for j, brma in enumerate(brmas)}
    drawn = renters.brma.astype(str).map(position).to_numpy()
    assert (census[np.arange(len(renters)), drawn] > 0).all()
    for r in np.unique(region):
        rows = region == r
        w = weight[rows, None] / weight[rows].sum()
        gap = (
            0.5
            * np.abs((w * p[rows]).sum(axis=0) - (w * census[rows]).sum(axis=0)).sum()
        )
        assert gap < 0.05, (r, gap)


def test_built_private_renters_rents_track_their_brmas_rent_levels(enhanced_frs, rents):
    """Dearer rents sit in dearer BRMAs, within region and bedroom band."""
    renters = private_renters(enhanced_frs)
    category = pd.Series(bedroom_band(renters.num_bedrooms.to_numpy())).map(
        BEDROOM_BAND_CATEGORY
    )
    listed = rents.set_index(["brma", "lha_category"]).median_weekly_rent
    brma_level = np.log(
        listed.loc[list(zip(renters.brma.astype(str), category))].to_numpy()
    )
    frame = pd.DataFrame(
        {
            "region": renters.region.astype(str).to_numpy(),
            "category": category.to_numpy(),
            "rent": np.log(renters.rent.to_numpy()),
            "level": brma_level,
            "w": renters.household_weight.to_numpy(),
        }
    )
    cell = frame.groupby(["region", "category"])
    for column in ("rent", "level"):
        mean = cell.apply(lambda g: np.average(g[column], weights=g.w))
        frame[column] -= mean.loc[list(zip(frame.region, frame.category))].to_numpy()
    covariance = np.average(frame.rent * frame.level, weights=frame.w)
    correlation = covariance / np.sqrt(
        np.average(frame.rent**2, weights=frame.w)
        * np.average(frame.level**2, weights=frame.w)
    )
    assert correlation > 0.1, correlation
