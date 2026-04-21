"""Tests for LA-level council-tax targets.

Covers both the canonical CSV (``storage/la_council_tax.csv``) and the
``get_targets`` module output. Includes bound checks to guard against
the kind of outlier that slipped into #371 (Isles of Scilly household
count inflated 2000x by a silently leaked national-fallback).
"""

from __future__ import annotations

import pandas as pd
import pytest

from policyengine_uk_data.targets.schema import GeographicLevel, Unit
from policyengine_uk_data.targets.sources._common import STORAGE
from policyengine_uk_data.targets.sources.la_council_tax import (
    _BAND_COUNT_COLUMNS,
    get_targets,
)


_CSV_NAME = "la_council_tax.csv"
_REFERENCE_LIST = "local_authorities_2021.csv"


@pytest.fixture(scope="module")
def la_ct_df() -> pd.DataFrame:
    path = STORAGE / _CSV_NAME
    if not path.exists():
        pytest.skip(f"{path} not present in this checkout")
    return pd.read_csv(path)


@pytest.fixture(scope="module")
def la_reference() -> pd.DataFrame:
    return pd.read_csv(STORAGE / _REFERENCE_LIST)


# -- CSV structure ---------------------------------------------------------


def test_csv_row_count_matches_local_authorities(la_ct_df, la_reference):
    assert len(la_ct_df) == len(la_reference), (
        f"la_council_tax.csv has {len(la_ct_df)} rows but "
        f"local_authorities_2021.csv has {len(la_reference)}."
    )


def test_csv_has_every_expected_column(la_ct_df):
    expected = {
        "code",
        "name",
        "country",
        "band_d_amount",
        "has_council_tax",
        "band_A",
        "band_B",
        "band_C",
        "band_D_count",
        "band_E",
        "band_F",
        "band_G",
        "band_H",
        "total_dwellings",
    }
    assert expected.issubset(la_ct_df.columns), (
        f"Missing columns: {expected - set(la_ct_df.columns)}"
    )


def test_country_column_covers_four_uk_countries(la_ct_df):
    assert set(la_ct_df["country"].unique()) == {
        "ENGLAND",
        "WALES",
        "SCOTLAND",
        "NORTHERN_IRELAND",
    }


def test_every_code_matches_reference(la_ct_df, la_reference):
    assert set(la_ct_df["code"]) == set(la_reference["code"]), (
        "LA codes in la_council_tax.csv differ from the reference list"
    )


# -- Value plausibility (the #371 lesson) ----------------------------------


def test_band_d_amount_within_plausible_range(la_ct_df):
    """Real UK Band D ranges from ~£1,000 (Westminster / Wandsworth) to
    ~£3,000 (some parts of England). A 1000× outlier like the #371 IoS
    households bug would blow through these bounds instantly.
    """
    ok = la_ct_df.dropna(subset=["band_d_amount"])
    assert ok["band_d_amount"].between(900, 3_500).all(), (
        "Band D amount outliers: "
        f"{ok.loc[~ok['band_d_amount'].between(900, 3_500), ['code', 'name', 'band_d_amount']].to_dict('records')}"
    )


def test_total_dwellings_within_plausible_range(la_ct_df):
    """Smallest UK billing authority (Isles of Scilly) has ~1,000 dwellings;
    largest (Birmingham) has ~470,000. Anything outside [200, 800_000]
    is a data-pipeline bug.
    """
    ok = la_ct_df.dropna(subset=["total_dwellings"])
    assert ok["total_dwellings"].between(200, 800_000).all(), (
        "Total-dwelling outliers: "
        f"{ok.loc[~ok['total_dwellings'].between(200, 800_000), ['code', 'name', 'total_dwellings']].to_dict('records')}"
    )


def test_isles_of_scilly_dwellings_are_thousands_not_millions(la_ct_df):
    """Explicit regression test for the #371 Isles of Scilly bug.

    IoS had 2,492,115 households in #371's la_land_values.csv because a
    national-total fallback leaked into one row. Real IoS has ~1,100
    dwellings.
    """
    ios = la_ct_df[la_ct_df["code"] == "E06000053"]
    assert len(ios) == 1
    total = float(ios["total_dwellings"].iloc[0])
    assert 500 <= total <= 5_000, (
        f"IoS total_dwellings = {total:.0f}; real figure is ~1,100"
    )


def test_band_counts_sum_to_total(la_ct_df):
    """For rows with band data, Σ(bands A-H) should equal total_dwellings
    (within rounding from the 10-property suppression at the lowest counts).
    """
    have_bands = la_ct_df.dropna(subset=["total_dwellings"]).copy()
    band_cols = list(_BAND_COUNT_COLUMNS.values())
    have_bands["sum_bands"] = have_bands[band_cols].fillna(0).sum(axis=1)
    diff = (have_bands["total_dwellings"] - have_bands["sum_bands"]).abs()
    # Allow up to 20 dwellings of slack for VOA rounding / suppression.
    assert diff.max() <= 20, (
        f"Band totals disagree by up to {int(diff.max())}; worst rows: "
        f"{have_bands.loc[diff == diff.max(), ['code', 'name', 'total_dwellings', 'sum_bands']].head(3).to_dict('records')}"
    )


# -- Country-level coverage expectations -----------------------------------


def test_all_english_las_have_band_d(la_ct_df):
    """Every English LA should have a Band D figure from DLUHC."""
    eng = la_ct_df[la_ct_df["country"] == "ENGLAND"]
    missing = eng[eng["band_d_amount"].isna()][["code", "name"]].to_dict("records")
    assert not missing, f"English LAs missing Band D: {missing}"


def test_all_welsh_las_have_band_d(la_ct_df):
    wa = la_ct_df[la_ct_df["country"] == "WALES"]
    missing = wa[wa["band_d_amount"].isna()][["code", "name"]].to_dict("records")
    assert not missing, f"Welsh LAs missing Band D: {missing}"


def test_all_scottish_las_have_band_d(la_ct_df):
    sc = la_ct_df[la_ct_df["country"] == "SCOTLAND"]
    missing = sc[sc["band_d_amount"].isna()][["code", "name"]].to_dict("records")
    assert not missing, f"Scottish LAs missing Band D: {missing}"


def test_northern_ireland_has_no_council_tax(la_ct_df):
    """NI uses domestic rates, not council tax. The CSV must reflect that."""
    ni = la_ct_df[la_ct_df["country"] == "NORTHERN_IRELAND"]
    assert (ni["has_council_tax"] == False).all()
    assert ni["band_d_amount"].isna().all()


# -- Spot-check published values -------------------------------------------


def test_wandsworth_and_westminster_are_lowest_in_england(la_ct_df):
    """Well-known political fact: Wandsworth and Westminster have the
    lowest Band D in the UK. Catches data-join mistakes that swap rows.
    """
    eng = (
        la_ct_df[la_ct_df["country"] == "ENGLAND"]
        .dropna(subset=["band_d_amount"])
        .sort_values("band_d_amount")
    )
    lowest_two = set(eng.head(2)["code"].tolist())
    assert "E09000032" in lowest_two, (
        "Wandsworth (E09000032) should be in the bottom two"
    )
    assert "E09000033" in lowest_two, (
        "Westminster (E09000033) should be in the bottom two"
    )


def test_scottish_band_d_is_lower_than_english_on_average(la_ct_df):
    """Scotland's Band D is typically ~£1,500, well below England ~£2,400."""
    en_mean = la_ct_df[la_ct_df["country"] == "ENGLAND"]["band_d_amount"].mean()
    sc_mean = la_ct_df[la_ct_df["country"] == "SCOTLAND"]["band_d_amount"].mean()
    assert sc_mean < en_mean - 500


# -- get_targets() output --------------------------------------------------


def test_get_targets_runs_without_network():
    targets = get_targets()
    assert targets, "Expected get_targets() to return a non-empty list"


def test_band_d_target_count_matches_csv(la_ct_df):
    targets = get_targets()
    band_d_targets = [t for t in targets if "council_tax_band_d/" in t.name]
    expected = int(la_ct_df["band_d_amount"].notna().sum())
    assert len(band_d_targets) == expected


def test_band_count_target_count_matches_csv(la_ct_df):
    targets = get_targets()
    bc_targets = [t for t in targets if "council_tax_band_count/" in t.name]
    # Expected = sum across all 8 bands of non-NaN counts.
    expected = int(la_ct_df[list(_BAND_COUNT_COLUMNS.values())].notna().sum().sum())
    assert len(bc_targets) == expected


def test_every_target_carries_local_authority_geo_level():
    for target in get_targets():
        assert target.geographic_level == GeographicLevel.LOCAL_AUTHORITY
        assert target.geo_code is not None


def test_band_d_targets_use_gbp_unit():
    for target in get_targets():
        if "council_tax_band_d/" in target.name:
            assert target.unit == Unit.GBP


def test_band_count_targets_use_count_unit_and_is_count_flag():
    for target in get_targets():
        if "council_tax_band_count/" in target.name:
            assert target.unit == Unit.COUNT
            assert target.is_count is True


def test_every_target_has_at_least_one_value_year():
    for target in get_targets():
        assert target.values, f"Target {target.name} has no values"


def test_every_band_count_target_value_within_sensible_range():
    """No single band within one LA should exceed the largest LA's total
    dwelling stock (Birmingham ≈ 470k). This catches a fallback-leak
    where a national total leaked into a row, à la #371.
    """
    for target in get_targets():
        if "council_tax_band_count/" not in target.name:
            continue
        for year, value in target.values.items():
            assert 0 <= value <= 500_000, (
                f"{target.name} has band count {value} in {year} — "
                "out of plausible [0, 500k] range"
            )
