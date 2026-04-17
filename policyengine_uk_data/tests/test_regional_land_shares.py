"""Tests for region-specific land-to-property ratios (#357)."""

import pandas as pd
import pytest

from policyengine_uk_data.targets.sources.mhclg_regional_land import (
    _regional_shares_from_frame,
)


def _frame(
    regions=("NORTH", "LONDON", "SOUTH"),
    dwellings=(1_000, 1_000, 1_000),
    prices=(100_000, 500_000, 200_000),
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "region": list(regions),
            "dwellings": list(dwellings),
            "avg_house_price": list(prices),
        }
    )


def test_default_reproduces_property_wealth_shares():
    """Pre-#357 behaviour: shares ∝ property wealth when no ratios passed."""
    df = _frame()
    shares = _regional_shares_from_frame(df)
    total = 100_000 + 500_000 + 200_000  # × 1000 dwellings each, cancels
    assert shares["NORTH"] == pytest.approx(100_000 / total)
    assert shares["LONDON"] == pytest.approx(500_000 / total)
    assert shares["SOUTH"] == pytest.approx(200_000 / total)


def test_shares_sum_to_one():
    df = _frame()
    shares = _regional_shares_from_frame(df)
    assert sum(shares.values()) == pytest.approx(1.0)


def test_shares_sum_to_one_with_ratios():
    df = _frame()
    ratios = {"NORTH": 0.3, "LONDON": 0.8, "SOUTH": 0.5}
    shares = _regional_shares_from_frame(df, ratios)
    assert sum(shares.values()) == pytest.approx(1.0)


def test_uniform_ratio_is_equivalent_to_default():
    """Passing the same ratio for every region must cancel out."""
    df = _frame()
    uniform = {"NORTH": 0.6, "LONDON": 0.6, "SOUTH": 0.6}
    default = _regional_shares_from_frame(df)
    scaled = _regional_shares_from_frame(df, uniform)
    for region in df["region"]:
        assert default[region] == pytest.approx(scaled[region])


def test_london_heavier_ratio_raises_london_share():
    """The substantive motivation of #357: London land share must rise when
    London's land-to-property ratio is higher than other regions."""
    df = _frame()
    default = _regional_shares_from_frame(df)
    ratios = {"NORTH": 0.4, "LONDON": 0.8, "SOUTH": 0.5}
    weighted = _regional_shares_from_frame(df, ratios)
    assert weighted["LONDON"] > default["LONDON"]
    # And the other two must fall to compensate (shares sum to 1).
    assert weighted["NORTH"] < default["NORTH"]
    assert weighted["SOUTH"] < default["SOUTH"]


def test_ratio_maths_is_correct():
    """Lock the arithmetic with a hand-computed example.

    Two equal-property-wealth regions with ratios 0.2 and 0.8 must split
    the total 20/80.
    """
    df = _frame(regions=("A", "B"), dwellings=(1_000, 1_000), prices=(100_000, 100_000))
    shares = _regional_shares_from_frame(df, {"A": 0.2, "B": 0.8})
    assert shares["A"] == pytest.approx(0.2)
    assert shares["B"] == pytest.approx(0.8)


def test_missing_region_in_ratio_map_raises():
    df = _frame()
    incomplete = {"NORTH": 0.4, "LONDON": 0.8}  # SOUTH missing
    with pytest.raises(KeyError, match="SOUTH"):
        _regional_shares_from_frame(df, incomplete)


def test_extra_region_in_ratio_map_is_tolerated():
    """A ratio mapping may carry extras (e.g. Wales) that this CSV lacks."""
    df = _frame()
    extras = {
        "NORTH": 0.4,
        "LONDON": 0.8,
        "SOUTH": 0.5,
        "WALES": 0.3,  # not in df
    }
    shares = _regional_shares_from_frame(df, extras)
    assert set(shares) == {"NORTH", "LONDON", "SOUTH"}


def test_all_zero_ratios_raise_rather_than_divide_by_zero():
    df = _frame()
    zeros = {"NORTH": 0.0, "LONDON": 0.0, "SOUTH": 0.0}
    with pytest.raises(ValueError, match="sum to zero"):
        _regional_shares_from_frame(df, zeros)


def test_ratio_of_zero_for_one_region_zeros_only_that_region():
    df = _frame()
    ratios = {"NORTH": 0.4, "LONDON": 0.0, "SOUTH": 0.5}
    shares = _regional_shares_from_frame(df, ratios)
    assert shares["LONDON"] == pytest.approx(0.0)
    assert shares["NORTH"] > 0
    assert shares["SOUTH"] > 0
    assert sum(shares.values()) == pytest.approx(1.0)


def test_real_csv_default_still_sums_to_one():
    """Smoke test against the shipped CSV that we didn't break the module."""
    from policyengine_uk_data.targets.sources.mhclg_regional_land import (
        _compute_regional_shares,
    )

    try:
        shares = _compute_regional_shares()
    except FileNotFoundError:
        pytest.skip("regional_land_values.csv not available")
    assert sum(shares.values()) == pytest.approx(1.0)
    # The 9 English regions must all be present.
    assert set(shares) >= {
        "NORTH_EAST",
        "NORTH_WEST",
        "YORKSHIRE",
        "EAST_MIDLANDS",
        "WEST_MIDLANDS",
        "EAST_OF_ENGLAND",
        "LONDON",
        "SOUTH_EAST",
        "SOUTH_WEST",
    }
