"""Tests for LA-level household land value calibration targets."""

import pandas as pd

from policyengine_uk_data.targets.schema import GeographicLevel
from policyengine_uk_data.targets.sources._common import STORAGE
from policyengine_uk_data.targets.sources._land import HOUSEHOLD_LAND_VALUES
from policyengine_uk_data.targets.sources.la_land import (
    _compute_la_shares,
    _compute_la_targets,
    _load_inputs,
    get_targets,
)

LA_TARGETS = _compute_la_targets()
LA_SHARES = _compute_la_shares()
LA_INPUTS = _load_inputs()


# ── CSV data quality ─────────────────────────────────────────────────


def test_csv_row_count_matches_la_code_list():
    """la_land_values.csv should have the same 360 LAs as local_authorities_2021.csv."""
    la_codes = pd.read_csv(STORAGE / "local_authorities_2021.csv")
    assert len(LA_INPUTS) == len(la_codes)
    assert set(LA_INPUTS["code"]) == set(la_codes["code"])


def test_csv_columns_match_schema():
    """CSV should have exactly the columns code, name, households, avg_house_price."""
    assert list(LA_INPUTS.columns) == ["code", "name", "households", "avg_house_price"]


def test_csv_no_missing_values():
    """No LA should have NaN in any column."""
    assert not LA_INPUTS.isna().any().any()


def test_csv_covers_all_four_countries():
    """All four UK countries (E/W/S/NI) should appear."""
    prefixes = LA_INPUTS["code"].str[0].unique()
    assert set(prefixes) == {"E", "W", "S", "N"}


def test_house_prices_within_plausible_range():
    """Avg house prices should be between £50k and £2m per LA."""
    for _, row in LA_INPUTS.iterrows():
        assert 50_000 <= row["avg_house_price"] <= 2_000_000, (
            f"{row['name']}: avg_house_price £{row['avg_house_price']:,} "
            "outside plausible range"
        )


def test_households_positive():
    """Every LA should have a positive implied household count."""
    assert (LA_INPUTS["households"] > 0).all()


def test_households_within_plausible_range():
    """Smallest UK billing authority (Isles of Scilly) has ~1,100
    households; largest (Birmingham) has ~450,000. A 1000x outlier — like
    the regional-total fallback that leaked into the IoS row pre-review —
    must be caught by bounds, not spotted by eye.
    """
    out_of_range = LA_INPUTS[~LA_INPUTS["households"].between(500, 500_000)]
    assert out_of_range.empty, (
        "Households out of plausible [500, 500_000] range: "
        f"{out_of_range[['code', 'name', 'households']].to_dict('records')}"
    )


def test_isles_of_scilly_households_are_thousands_not_millions():
    """Explicit regression for the IoS fallback leak (was 2,492,115).

    Real IoS has ~1,115 households per ONS mid-2023 estimate (pop ~2,000).
    Anything outside [500, 5,000] indicates the fallback path has
    regressed again.
    """
    ios = LA_INPUTS[LA_INPUTS["code"] == "E06000053"]
    assert len(ios) == 1
    hh = int(ios["households"].iloc[0])
    assert 500 <= hh <= 5_000, (
        f"Isles of Scilly households = {hh:,}; ONS mid-2023 estimate is ~1,115"
    )


# ── Share constraints ────────────────────────────────────────────────


def test_shares_sum_to_one():
    """LA shares should sum to exactly 1."""
    assert abs(LA_SHARES["share"].sum() - 1.0) < 1e-9


def test_all_shares_positive():
    """Every LA share should be positive."""
    assert (LA_SHARES["share"] > 0).all()


# ── Target value constraints ─────────────────────────────────────────


def test_all_targets_positive():
    """Every LA target should be a positive value for every year."""
    for code, values in LA_TARGETS.items():
        for year, value in values.items():
            assert value > 0, f"{code} {year}: non-positive target {value}"


def test_targets_sum_to_national():
    """LA targets should sum to the ONS national household land total."""
    for year in (2021, 2023, 2024):
        la_sum = sum(values[year] for values in LA_TARGETS.values())
        national = HOUSEHOLD_LAND_VALUES[year]
        rel_error = abs(la_sum / national - 1)
        assert rel_error < 1e-6, (
            f"{year}: LA sum £{la_sum / 1e12:.3f}tn != "
            f"national £{national / 1e12:.3f}tn"
        )


def test_kensington_and_chelsea_above_blackpool():
    """K&C avg household land value should exceed Blackpool's."""
    kc_code = LA_INPUTS.loc[LA_INPUTS["name"] == "Kensington and Chelsea", "code"].iloc[
        0
    ]
    blackpool_code = LA_INPUTS.loc[LA_INPUTS["name"] == "Blackpool", "code"].iloc[0]
    kc_hh = LA_INPUTS.set_index("code").loc[kc_code, "households"]
    bp_hh = LA_INPUTS.set_index("code").loc[blackpool_code, "households"]
    kc_per_hh = LA_TARGETS[kc_code][2024] / kc_hh
    bp_per_hh = LA_TARGETS[blackpool_code][2024] / bp_hh
    assert kc_per_hh > bp_per_hh * 3, (
        f"K&C avg household land (£{kc_per_hh:,.0f}) should be at least "
        f"3x Blackpool (£{bp_per_hh:,.0f})"
    )


def test_london_prime_dominates_top_quintile():
    """Top quintile of LAs by avg household land value should be London-heavy."""
    totals = pd.Series(
        {code: values[2024] for code, values in LA_TARGETS.items()}, name="total"
    )
    inputs = LA_INPUTS.set_index("code")
    avg_per_hh = (totals / inputs["households"]).sort_values(ascending=False)
    top_quintile = avg_per_hh.head(len(avg_per_hh) // 5).index
    london_codes = set(inputs.loc[inputs.index.str.startswith("E09"), :].index)
    london_in_top = len(set(top_quintile) & london_codes)
    assert london_in_top >= 15, (
        f"Expected London LAs to dominate top quintile, found only {london_in_top}"
    )


def test_london_total_land_dwarfs_north_east():
    """Sum of London LA targets should exceed sum of North-East LA targets."""
    inputs = LA_INPUTS.set_index("code")
    london_codes = inputs.loc[inputs.index.str.startswith("E09"), :].index
    ne_prefixes = (
        "E06000001",
        "E06000002",
        "E06000003",
        "E06000004",
        "E06000005",
        "E06000047",
        "E08000021",
        "E08000022",
        "E08000023",
        "E08000024",
        "E08000037",
        "E06000057",
    )
    ne_codes = [c for c in inputs.index if c in ne_prefixes]
    london_total = sum(LA_TARGETS[c][2024] for c in london_codes)
    ne_total = sum(LA_TARGETS[c][2024] for c in ne_codes)
    assert london_total > ne_total * 3, (
        f"London total (£{london_total / 1e9:.0f}bn) should exceed "
        f"NE total (£{ne_total / 1e9:.0f}bn) by at least 3x"
    )


# ── Target registry integration ──────────────────────────────────────


def test_get_targets_returns_360():
    """get_targets() should return exactly 360 LA targets."""
    targets = get_targets()
    assert len(targets) == 360


def test_target_names_follow_code_pattern():
    """Target names should follow the ons/household_land_value/{code} pattern."""
    targets = get_targets()
    for t in targets:
        assert t.name.startswith("ons/household_land_value/")
        assert t.name.removeprefix("ons/household_land_value/") == t.geo_code


def test_targets_declare_local_authority_geographic_level():
    """All LA targets should be tagged with GeographicLevel.LOCAL_AUTHORITY."""
    for t in get_targets():
        assert t.geographic_level == GeographicLevel.LOCAL_AUTHORITY


def test_targets_have_values_for_all_known_years():
    """LA targets should carry every year in the backfilled series."""
    expected_years = set(HOUSEHOLD_LAND_VALUES)
    for t in get_targets():
        assert set(t.values) == expected_years, (
            f"{t.name} missing years: {expected_years - set(t.values)}"
        )


def test_target_registry_includes_la_targets():
    """LA land targets should appear in the global registry."""
    from policyengine_uk_data.targets import get_all_targets

    targets = get_all_targets(
        year=2024, geographic_level=GeographicLevel.LOCAL_AUTHORITY
    )
    la_land = [t for t in targets if t.name.startswith("ons/household_land_value/")]
    assert len(la_land) == 360, (
        f"Expected 360 LA household-land targets, got {len(la_land)}"
    )
