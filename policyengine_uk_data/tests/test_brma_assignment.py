"""BRMA assignment: census private-rented weights and the draw.

Invariants of ``assign_brmas``, for any weights table and any benefit units
whose region × LHA category cell has a positive weight:

1. Support: each benefit unit's BRMA has a positive weight in its cell.
2. Determinism: the same generator state gives the same BRMAs.
3. Fail closed: a benefit unit in a cell with no positive weight raises.
4. Proportionality: draws converge on the cell's weights.
"""

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from policyengine_uk.variables.household.demographic.locations import BRMAName

from policyengine_uk_data.datasets.brma import (
    BRMA_HOUSEHOLDS_PATH,
    LHA_CATEGORY_BEDROOMS,
    assign_brmas,
    load_brma_weights,
)

REGIONS = [
    "NORTH_EAST",
    "NORTH_WEST",
    "YORKSHIRE",
    "EAST_MIDLANDS",
    "WEST_MIDLANDS",
    "EAST_OF_ENGLAND",
    "LONDON",
    "SOUTH_EAST",
    "SOUTH_WEST",
    "WALES",
    "SCOTLAND",
    "NORTHERN_IRELAND",
]
CATEGORIES = list(LHA_CATEGORY_BEDROOMS)


@pytest.fixture(scope="module")
def households():
    return pd.read_csv(BRMA_HOUSEHOLDS_PATH, dtype={"bedrooms": str})


@pytest.fixture(scope="module")
def weights():
    return load_brma_weights()


def test_table_covers_every_region_band_and_brma(households):
    assert list(households.columns) == ["region", "brma", "bedrooms", "households"]
    assert (households.households > 0).all()
    assert not households.duplicated(["region", "brma", "bedrooms"]).any()
    assert set(households.region) == set(REGIONS)
    assert set(households.brma) == set(BRMAName.__members__)
    bands = households.groupby("region").bedrooms.agg(set)
    # Northern Ireland's 2021 census has no bedrooms question.
    assert bands.pop("NORTHERN_IRELAND") == {"all"}
    assert all(b == {"1", "2", "3", "4+"} for b in bands)


# Published private-rented households (private landlord or letting agency plus
# other private rented): Census 2021 TS054 (England, Wales); Scotland's Census
# 2022 tenure by bedrooms, national cells; NISRA Census 2021 tenure, 7 categories.
CENSUS_PRIVATE_RENTED = {
    "ENGLAND": 4_794_889,
    "WALES": 228_642,
    "SCOTLAND": 323_042,
    "NORTHERN_IRELAND": 132_436,
}


def test_nations_reconcile_to_published_census_totals(households):
    nation = households.region.where(
        households.region.isin(["WALES", "SCOTLAND", "NORTHERN_IRELAND"]), "ENGLAND"
    )
    totals = households.groupby(nation).households.sum()
    for name, published in CENSUS_PRIVATE_RENTED.items():
        # Small-area cells are perturbed for disclosure control, so sums
        # differ slightly from national tables.
        assert abs(totals[name] / published - 1) < 0.001, name


def test_edinburgh_and_glasgow_hold_the_most_scottish_private_renters(households):
    # Census 2022: Lothian 19.3%, Greater Glasgow 15.9%. The 2019-20 list of
    # rents gave Greater Glasgow the fewest entries of any Scottish BRMA.
    scotland = (
        households[households.region == "SCOTLAND"].groupby("brma").households.sum()
    )
    assert set(scotland.nlargest(2).index) == {"LOTHIAN", "GREATER_GLASGOW"}


def test_every_region_category_cell_has_weights(weights):
    cells = weights.groupby(["region", "lha_category"]).weight.sum()
    assert len(cells) == len(REGIONS) * len(CATEGORIES)
    assert (cells > 0).all()


def test_shared_and_one_bedroom_categories_use_the_same_weights(weights):
    a = weights[weights.lha_category == "A"].drop(columns="lha_category")
    b = weights[weights.lha_category == "B"].drop(columns="lha_category")
    pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True))


def test_large_draw_matches_scottish_two_bedroom_weights(weights):
    cell = weights[(weights.region == "SCOTLAND") & (weights.lha_category == "C")]
    n = 400_000
    drawn = assign_brmas(
        np.full(n, "SCOTLAND"), np.full(n, "C"), np.random.default_rng(1), weights
    )
    observed = pd.Series(drawn).value_counts(normalize=True)
    expected = cell.set_index("brma").weight / cell.weight.sum()
    assert observed.index.isin(expected.index).all()
    assert (
        observed.reindex(expected.index, fill_value=0) - expected
    ).abs().max() < 0.004


@st.composite
def weights_and_units(draw):
    regions = draw(
        st.lists(st.sampled_from(REGIONS), min_size=1, max_size=3, unique=True)
    )
    brmas = [f"B{i}" for i in range(draw(st.integers(1, 5)))]
    rows = [
        (region, category, brma, draw(st.sampled_from([0, 0.5, 1, 7, 1000])))
        for region in regions
        for category in CATEGORIES
        for brma in brmas
    ]
    table = pd.DataFrame(rows, columns=["region", "lha_category", "brma", "weight"])
    table = table[table.weight > 0]
    cells = sorted(set(zip(table.region, table.lha_category)))
    units = draw(st.lists(st.sampled_from(cells), max_size=40)) if cells else []
    region = np.array([u[0] for u in units], dtype=str)
    category = np.array([u[1] for u in units], dtype=str)
    return table, region, category, draw(st.integers(0, 2**32 - 1))


@settings(max_examples=200, deadline=None, derandomize=True)
@given(weights_and_units())
def test_draws_stay_on_positive_weights_and_are_deterministic(case):
    table, region, category, seed = case
    first = assign_brmas(region, category, np.random.default_rng(seed), table)
    second = assign_brmas(region, category, np.random.default_rng(seed), table)
    assert list(first) == list(second)
    positive = set(zip(table.region, table.lha_category, table.brma))
    assert all((r, c, b) in positive for r, c, b in zip(region, category, first))


@settings(max_examples=100, deadline=None, derandomize=True)
@given(weights_and_units(), st.sampled_from(REGIONS), st.sampled_from(CATEGORIES))
def test_cells_without_positive_weight_fail_closed(case, region, category):
    table, *_ = case
    if ((table.region == region) & (table.lha_category == category)).any():
        return
    with pytest.raises(ValueError, match="No BRMA weights"):
        assign_brmas(
            np.array([region]), np.array([category]), np.random.default_rng(0), table
        )


def test_built_dataset_brmas_lie_in_their_regions_weights(enhanced_frs, weights):
    household = enhanced_frs.household
    supported = set(zip(weights.region, weights.brma))
    pairs = zip(household.region.astype(str), household.brma.astype(str))
    assert all(pair in supported for pair in pairs)
