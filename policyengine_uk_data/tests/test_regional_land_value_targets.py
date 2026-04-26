"""Tests for regional household land value calibration targets.

See: https://github.com/PolicyEngine/policyengine-uk-data/issues/314
"""

import pandas as pd

from policyengine_uk_data.targets.sources._land import HOUSEHOLD_LAND_VALUES
from policyengine_uk_data.targets.sources.mhclg_regional_land import (
    _compute_regional_targets,
    get_targets,
)

REGIONAL_TARGETS = _compute_regional_targets()

_GB_REGIONS = {
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
}


# ── CSV data quality ─────────────────────────────────────────────────


def test_csv_has_all_gb_regions():
    """CSV should contain exactly the 11 GB regions."""
    assert set(REGIONAL_TARGETS.keys()) == _GB_REGIONS


def test_csv_no_northern_ireland():
    """Northern Ireland should be excluded (not in FRS sample frame)."""
    assert "NORTHERN_IRELAND" not in REGIONAL_TARGETS


def test_all_targets_positive():
    """Every regional target should be a positive value."""
    for region, values in REGIONAL_TARGETS.items():
        for year, value in values.items():
            assert value > 0, f"{region} {year} has non-positive target: {value}"


# ── Target value constraints ─────────────────────────────────────────


def test_regional_targets_sum_to_national():
    """Regional targets should sum to the ONS national household land total."""
    for year in (2021, 2023, 2025):
        regional_sum = sum(values[year] for values in REGIONAL_TARGETS.values())
        national = HOUSEHOLD_LAND_VALUES[year]
        rel_error = abs(regional_sum / national - 1)
        assert rel_error < 0.01, (
            f"{year}: regional sum £{regional_sum / 1e12:.2f}tn != "
            f"national £{national / 1e12:.2f}tn"
        )


def test_london_highest_land_value():
    """London should have the highest regional land value target."""
    london = REGIONAL_TARGETS["LONDON"][2025]
    for region, values in REGIONAL_TARGETS.items():
        if region != "LONDON":
            value = values[2025]
            assert london > value, (
                f"London (£{london / 1e9:.0f}bn) should exceed "
                f"{region} (£{value / 1e9:.0f}bn)"
            )


def test_london_to_north_east_ratio():
    """London/NE ratio should be at least 3x.

    UK HPI shows London avg house price ~3.3x North East, and London
    has ~3x more dwellings, so the ratio should be substantial.
    """
    ratio = REGIONAL_TARGETS["LONDON"][2025] / REGIONAL_TARGETS["NORTH_EAST"][2025]
    assert ratio >= 3.0, f"London/NE ratio = {ratio:.1f}x, expected >= 3x"


def test_south_east_above_south_west():
    """South East should have higher land value than South West."""
    assert REGIONAL_TARGETS["SOUTH_EAST"][2025] > REGIONAL_TARGETS["SOUTH_WEST"][2025]


def test_east_of_england_above_east_midlands():
    """East of England should have higher land value than East Midlands."""
    assert (
        REGIONAL_TARGETS["EAST_OF_ENGLAND"][2025]
        > REGIONAL_TARGETS["EAST_MIDLANDS"][2025]
    )


# ── Target registry integration ──────────────────────────────────────


def test_get_targets_returns_11():
    """get_targets() should return exactly 11 regional targets."""
    targets = get_targets()
    assert len(targets) == 11


def test_target_names_match_regions():
    """Target names should follow the ons/household_land_value/{REGION} pattern."""
    targets = get_targets()
    names = {t.name for t in targets}
    expected = {f"ons/household_land_value/{r}" for r in _GB_REGIONS}
    assert names == expected


def test_targets_have_values_for_2025():
    """All targets should have a value for 2025."""
    for t in get_targets():
        assert 2025 in t.values, f"{t.name} missing value for 2025"


def test_targets_have_values_for_2021_to_2026():
    """Regional targets should cover the full backfilled annual range."""
    expected_years = set(range(2021, 2027))
    for target in get_targets():
        assert set(target.values) == expected_years


def test_target_registry_includes_regional():
    """Regional land targets should appear in the global registry."""
    from policyengine_uk_data.targets import get_all_targets

    targets = get_all_targets(year=2025)
    regional = [t for t in targets if t.name.startswith("ons/household_land_value/")]
    assert len(regional) == 11, (
        f"Expected 11 regional land targets, got {len(regional)}"
    )


# ── House price data verification ────────────────────────────────────


def test_hpi_prices_within_range():
    """All house prices should be between £100k and £600k."""
    from policyengine_uk_data.targets.sources._common import STORAGE

    df = pd.read_csv(STORAGE / "regional_land_values.csv")
    for _, row in df.iterrows():
        assert 100_000 <= row["avg_house_price"] <= 600_000, (
            f"{row['region']}: avg_house_price £{row['avg_house_price']:,.0f} "
            f"outside plausible range"
        )


def test_london_highest_house_price():
    """London should have the highest average house price."""
    from policyengine_uk_data.targets.sources._common import STORAGE

    df = pd.read_csv(STORAGE / "regional_land_values.csv")
    london = df.loc[df["region"] == "LONDON", "avg_house_price"].iloc[0]
    for _, row in df.iterrows():
        if row["region"] != "LONDON":
            assert london >= row["avg_house_price"], (
                f"London price (£{london:,.0f}) should be >= "
                f"{row['region']} (£{row['avg_house_price']:,.0f})"
            )
