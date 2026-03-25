"""Tests for MHCLG regional household land value calibration targets.

See: https://github.com/PolicyEngine/policyengine-uk-data/issues/314
"""

import pandas as pd
import pytest

from policyengine_uk_data.targets.sources.mhclg_regional_land import (
    _compute_regional_targets,
    _ONS_2024_HOUSEHOLD,
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
    for region, value in REGIONAL_TARGETS.items():
        assert value > 0, f"{region} has non-positive target: {value}"


# ── Target value constraints ─────────────────────────────────────────


def test_regional_targets_sum_to_national():
    """Regional targets should sum to the ONS national household land total."""
    regional_sum = sum(REGIONAL_TARGETS.values())
    rel_error = abs(regional_sum / _ONS_2024_HOUSEHOLD - 1)
    assert rel_error < 0.01, (
        f"Regional sum £{regional_sum / 1e12:.2f}tn != "
        f"national £{_ONS_2024_HOUSEHOLD / 1e12:.2f}tn"
    )


def test_london_highest_land_value():
    """London should have the highest regional land value target."""
    london = REGIONAL_TARGETS["LONDON"]
    for region, value in REGIONAL_TARGETS.items():
        if region != "LONDON":
            assert london > value, (
                f"London (£{london / 1e9:.0f}bn) should exceed "
                f"{region} (£{value / 1e9:.0f}bn)"
            )


def test_london_to_north_east_ratio():
    """London/NE ratio should be at least 3x.

    MHCLG shows 20x in land per hectare; even after adjusting for
    dwelling counts the ratio should far exceed the old model's 1.2x.
    """
    ratio = REGIONAL_TARGETS["LONDON"] / REGIONAL_TARGETS["NORTH_EAST"]
    assert ratio >= 3.0, (
        f"London/NE ratio = {ratio:.1f}x, expected >= 3x"
    )


def test_south_east_above_south_west():
    """South East should have higher land value than South West."""
    assert REGIONAL_TARGETS["SOUTH_EAST"] > REGIONAL_TARGETS["SOUTH_WEST"]


def test_east_of_england_above_east_midlands():
    """East of England should have higher land value than East Midlands."""
    assert REGIONAL_TARGETS["EAST_OF_ENGLAND"] > REGIONAL_TARGETS["EAST_MIDLANDS"]


# ── Target registry integration ──────────────────────────────────────


def test_get_targets_returns_11():
    """get_targets() should return exactly 11 regional targets."""
    targets = get_targets()
    assert len(targets) == 11


def test_target_names_match_regions():
    """Target names should follow the mhclg/household_land_value/{REGION} pattern."""
    targets = get_targets()
    names = {t.name for t in targets}
    expected = {f"mhclg/household_land_value/{r}" for r in _GB_REGIONS}
    assert names == expected


def test_targets_have_values_for_2025():
    """All targets should have a value for 2025."""
    for t in get_targets():
        assert 2025 in t.values, f"{t.name} missing value for 2025"


def test_target_registry_includes_mhclg():
    """MHCLG regional targets should appear in the global registry."""
    from policyengine_uk_data.targets import get_all_targets

    targets = get_all_targets(year=2025)
    mhclg = [t for t in targets if t.source == "mhclg"]
    assert len(mhclg) == 11, f"Expected 11 MHCLG regional targets, got {len(mhclg)}"


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


def test_intensity_ratios_within_range():
    """Intensity ratios should be between 0.3 and 0.95."""
    from policyengine_uk_data.targets.sources._common import STORAGE

    df = pd.read_csv(STORAGE / "regional_land_values.csv")
    for _, row in df.iterrows():
        assert 0.3 <= row["land_intensity"] <= 0.95, (
            f"{row['region']}: land_intensity {row['land_intensity']} "
            f"outside plausible range"
        )


def test_london_highest_intensity():
    """London should have the highest land intensity."""
    from policyengine_uk_data.targets.sources._common import STORAGE

    df = pd.read_csv(STORAGE / "regional_land_values.csv")
    london = df.loc[df["region"] == "LONDON", "land_intensity"].iloc[0]
    for _, row in df.iterrows():
        if row["region"] != "LONDON":
            assert london >= row["land_intensity"], (
                f"London intensity ({london}) should be >= "
                f"{row['region']} ({row['land_intensity']})"
            )
