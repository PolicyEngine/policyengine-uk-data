"""Tests for LA-level main residence value calibration targets.

Targets are built from directly observed LA-level housing indicators
(HMLR avg house price × English Housing Survey ownership share × Census
household count), mirroring the existing private-rent target. No
national-total apportionment.
"""

import pandas as pd

from policyengine_uk_data.targets.schema import GeographicLevel
from policyengine_uk_data.targets.sources._common import STORAGE
from policyengine_uk_data.targets.sources.la_land import (
    _compute_la_targets,
    get_targets,
    load_la_avg_prices,
)
from policyengine_uk_data.targets.sources.local_la_extras import (
    load_household_counts,
    load_tenure_data,
)


LA_PRICES = load_la_avg_prices()
LA_TARGETS = _compute_la_targets()


# ── CSV data quality ─────────────────────────────────────────────────


def test_csv_row_count_matches_la_code_list():
    """la_land_values.csv should have the same 360 LAs as local_authorities_2021.csv."""
    la_codes = pd.read_csv(STORAGE / "local_authorities_2021.csv")
    raw = pd.read_csv(STORAGE / "la_land_values.csv")
    assert len(raw) == len(la_codes)
    assert set(raw["code"]) == set(la_codes["code"])


def test_csv_columns_match_schema():
    """CSV should have exactly the columns code, name, households, avg_house_price."""
    raw = pd.read_csv(STORAGE / "la_land_values.csv")
    assert list(raw.columns) == ["code", "name", "households", "avg_house_price"]


def test_csv_no_missing_values():
    """No LA should have NaN in any column."""
    raw = pd.read_csv(STORAGE / "la_land_values.csv")
    assert not raw.isna().any().any()


def test_csv_covers_all_four_countries():
    """All four UK countries (E/W/S/NI) should appear."""
    prefixes = LA_PRICES["code"].str[0].unique()
    assert set(prefixes) == {"E", "W", "S", "N"}


def test_house_prices_within_plausible_range():
    """Avg house prices should be between £50k and £2m per LA."""
    for _, row in LA_PRICES.iterrows():
        assert 50_000 <= row["avg_house_price"] <= 2_000_000, (
            f"{row['name']}: avg_house_price £{row['avg_house_price']:,} "
            "outside plausible range"
        )


def test_csv_households_within_plausible_range():
    """Smallest UK billing authority (Isles of Scilly) has ~1,100
    households; largest (Birmingham) has ~450,000. The CSV `households`
    column is retained as a regression fixture for the IoS fallback leak
    even though the calibration target uses Census counts.
    """
    raw = pd.read_csv(STORAGE / "la_land_values.csv")
    out_of_range = raw[~raw["households"].between(500, 500_000)]
    assert out_of_range.empty, (
        "CSV households out of plausible [500, 500_000] range: "
        f"{out_of_range[['code', 'name', 'households']].to_dict('records')}"
    )


def test_isles_of_scilly_households_are_thousands_not_millions():
    """Explicit regression for the IoS fallback leak (was 2,492,115)."""
    raw = pd.read_csv(STORAGE / "la_land_values.csv")
    ios = raw[raw["code"] == "E06000053"]
    assert len(ios) == 1
    hh = int(ios["households"].iloc[0])
    assert 500 <= hh <= 5_000, (
        f"Isles of Scilly households = {hh:,}; ONS mid-2023 estimate is ~1,115"
    )


# ── Target value constraints ─────────────────────────────────────────


def test_targets_match_observed_product():
    """Every target equals avg_price × ownership_share × n_households exactly.

    No national-total apportionment, no rescaling: just the directly
    observed product, identical in shape to the rent target.
    """
    prices = LA_PRICES.set_index("code")["avg_house_price"]
    tenure = load_tenure_data().set_index("la_code")
    households = load_household_counts().set_index("la_code")["households"]

    for code, target in LA_TARGETS.items():
        if code not in tenure.index or code not in households.index:
            continue
        ownership = (
            tenure.loc[code, "owned_outright_pct"]
            + tenure.loc[code, "owned_mortgage_pct"]
        ) / 100
        expected = prices.loc[code] * ownership * households.loc[code]
        assert abs(target - expected) < 1e-3, (
            f"{code}: target {target:,.2f} != expected {expected:,.2f}"
        )


def test_all_targets_positive():
    """Every per-LA target should be positive."""
    assert all(value > 0 for value in LA_TARGETS.values())


def test_explicit_targets_cover_english_las():
    """Direct-formula targets are produced for LAs with EHS tenure data
    (England). Wales, Scotland and Northern Ireland LAs are handled by
    the national-share fallback in loss.py — same as the existing
    tenure target, by construction."""
    prefixes = {code[0] for code in LA_TARGETS}
    assert prefixes == {"E"}, (
        f"Expected English-only targets from EHS data, got {sorted(prefixes)}"
    )


def test_kensington_and_chelsea_above_blackpool():
    """K&C aggregate main-residence-value target should exceed Blackpool's."""
    name_to_code = dict(zip(LA_PRICES["name"], LA_PRICES["code"]))
    kc = LA_TARGETS[name_to_code["Kensington and Chelsea"]]
    bp = LA_TARGETS[name_to_code["Blackpool"]]
    assert kc > bp, f"K&C target (£{kc / 1e9:.1f}bn) should exceed Blackpool (£{bp / 1e9:.1f}bn)"


def test_london_total_exceeds_north_east():
    """Sum of London LA targets should exceed sum of North-East LA targets."""
    london_codes = [c for c in LA_TARGETS if c.startswith("E09")]
    ne_prefixes = {
        "E06000001", "E06000002", "E06000003", "E06000004", "E06000005",
        "E06000047", "E08000021", "E08000022", "E08000023", "E08000024",
        "E08000037", "E06000057",
    }
    ne_codes = [c for c in LA_TARGETS if c in ne_prefixes]
    london_total = sum(LA_TARGETS[c] for c in london_codes)
    ne_total = sum(LA_TARGETS[c] for c in ne_codes)
    assert london_total > ne_total * 3, (
        f"London total (£{london_total / 1e9:.0f}bn) should exceed "
        f"NE total (£{ne_total / 1e9:.0f}bn) by at least 3x"
    )


# ── Target registry integration ──────────────────────────────────────


def test_get_targets_returns_targets_for_covered_las():
    """get_targets() returns one Target per LA with all inputs available."""
    targets = get_targets()
    assert len(targets) == len(LA_TARGETS)
    assert {t.geo_code for t in targets} == set(LA_TARGETS)


def test_target_names_follow_code_pattern():
    """Target names should follow the housing/main_residence_value/{code} pattern."""
    for t in get_targets():
        assert t.name.startswith("housing/main_residence_value/")
        assert t.name.removeprefix("housing/main_residence_value/") == t.geo_code


def test_targets_declare_local_authority_geographic_level():
    """All LA targets should be tagged with GeographicLevel.LOCAL_AUTHORITY."""
    for t in get_targets():
        assert t.geographic_level == GeographicLevel.LOCAL_AUTHORITY


def test_targets_declare_hmlr_source():
    """LA property-value targets are sourced from HMLR UK HPI."""
    for t in get_targets():
        assert t.source == "hmlr"


def test_targets_have_calibration_year_values():
    """LA targets should carry values for the supported calibration years."""
    for t in get_targets():
        assert {2024, 2025, 2026} <= set(t.values)


def test_target_registry_includes_la_targets():
    """LA property-value targets should appear in the global registry."""
    from policyengine_uk_data.targets import get_all_targets

    targets = get_all_targets(
        year=2024, geographic_level=GeographicLevel.LOCAL_AUTHORITY
    )
    la_property = [
        t for t in targets if t.name.startswith("housing/main_residence_value/")
    ]
    assert len(la_property) == len(LA_TARGETS), (
        f"Expected {len(LA_TARGETS)} LA property-value targets, got {len(la_property)}"
    )
