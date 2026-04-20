"""Tests for the ONS age-specific fertility rates loader.

Hermetic: builds a synthetic xlsx matching the ONS Table_10 schema so CI
never hits the network.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest


def _synthetic_workbook(tmp_path: Path) -> Path:
    """Write a minimal xlsx matching the ONS Table_10 schema.

    The real file has a five-row rubric above the column header at row 5
    (zero-indexed). We replicate that so the loader's ``header=5`` logic
    is exercised.
    """
    path = tmp_path / "ons_asfr_fake.xlsx"
    header = [
        "Year",
        "Country",
        "Parent",
        "Age group (years)",
        "Number of live births",
        "Age-specific fertility rate",
    ]
    rows: list[list] = [
        ["Table 10: Live births..."] + [""] * 5,
        [""] * 6,
        [""] * 6,
        [""] * 6,
        [""] * 6,
        header,
    ]
    bands = [
        ("Under 20", 7.0),
        ("20 to 24", 35.0),
        ("25 to 29", 70.0),
        ("30 to 34", 90.0),
        ("35 to 39", 50.0),
        ("40 and over", 15.0),
    ]
    for parent in ["Mother", "Father"]:
        for year in [2022, 2024]:
            for country in ["England", "Wales", "England, Wales and Elsewhere"]:
                for band, rate in bands:
                    scale = 1.0 if year == 2024 else 0.9
                    rows.append([year, country, parent, band, 10_000, rate * scale])

    df = pd.DataFrame(rows)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Table_10", index=False, header=False)
    return path


def _clear_cache():
    from policyengine_uk_data.targets.sources.ons_fertility import (
        load_ons_fertility_rates,
    )

    load_ons_fertility_rates.cache_clear()


def test_load_ons_fertility_rates_returns_long_frame(tmp_path):
    from policyengine_uk_data.targets.sources.ons_fertility import (
        load_ons_fertility_rates,
    )

    path = _synthetic_workbook(tmp_path)
    _clear_cache()
    t = load_ons_fertility_rates(path=str(path))

    expected_cols = {"year", "country", "age_low", "age_high", "rate_per_1000"}
    assert expected_cols.issubset(t.columns)
    # Fathers filtered out.
    assert {2022, 2024} == set(t["year"].unique())
    # Three countries × two years × six bands = 36 Mother rows.
    assert len(t) == 36


def test_open_band_expands_to_five_years_only(tmp_path):
    """The "40 and over" row must not inflate ages 45+."""
    from policyengine_uk_data.targets.sources.ons_fertility import (
        get_fertility_rates,
        load_ons_fertility_rates,
    )

    path = _synthetic_workbook(tmp_path)
    _clear_cache()
    t = load_ons_fertility_rates(path=str(path))

    r = get_fertility_rates(2024, tables=t)
    # 40-44 populated at 15/1000 = 0.015; 45+ absent (so 0 on lookup).
    assert r[40] == pytest.approx(0.015)
    assert r[44] == pytest.approx(0.015)
    assert 45 not in r
    assert 50 not in r


def test_rates_are_per_woman_probability_not_per_1000(tmp_path):
    """Sanity: ONS writes per-1000 but the module converts to per-1 probability."""
    from policyengine_uk_data.targets.sources.ons_fertility import (
        get_fertility_rates,
        load_ons_fertility_rates,
    )

    path = _synthetic_workbook(tmp_path)
    _clear_cache()
    t = load_ons_fertility_rates(path=str(path))

    r = get_fertility_rates(2024, tables=t)
    # Peak band in the fixture is 30-34 @ 90/1000 = 0.090.
    assert r[32] == pytest.approx(0.090)
    # Every single-year age in the band shares the band rate.
    assert r[30] == r[31] == r[32] == r[33] == r[34]


def test_under_20_starts_at_15(tmp_path):
    """'Under 20' is mapped to single-year ages 15-19 (not 0-19)."""
    from policyengine_uk_data.targets.sources.ons_fertility import (
        get_fertility_rates,
        load_ons_fertility_rates,
    )

    path = _synthetic_workbook(tmp_path)
    _clear_cache()
    t = load_ons_fertility_rates(path=str(path))

    r = get_fertility_rates(2024, tables=t)
    assert r[15] == pytest.approx(0.007)
    assert r[19] == pytest.approx(0.007)
    assert 14 not in r
    assert 0 not in r


def test_year_resolution_prefers_nearest_past(tmp_path):
    """Asking for a year not in the data falls back to the most recent earlier year."""
    from policyengine_uk_data.targets.sources.ons_fertility import (
        get_fertility_rates,
        load_ons_fertility_rates,
    )

    path = _synthetic_workbook(tmp_path)
    _clear_cache()
    t = load_ons_fertility_rates(path=str(path))

    # 2023 isn't in the fixture; should fall back to 2022 (scaled 0.9×).
    r = get_fertility_rates(2023, tables=t)
    assert r[32] == pytest.approx(0.090 * 0.9)


def test_year_before_earliest_raises(tmp_path):
    from policyengine_uk_data.targets.sources.ons_fertility import (
        get_fertility_rates,
        load_ons_fertility_rates,
    )

    path = _synthetic_workbook(tmp_path)
    _clear_cache()
    t = load_ons_fertility_rates(path=str(path))

    with pytest.raises(KeyError, match="No ONS ASFR data for 1900"):
        get_fertility_rates(1900, tables=t)


def test_country_filter(tmp_path):
    """Wales rates should differ from England if the fixture makes them do so."""
    from policyengine_uk_data.targets.sources.ons_fertility import (
        get_fertility_rates,
        load_ons_fertility_rates,
    )

    path = _synthetic_workbook(tmp_path)
    _clear_cache()
    t = load_ons_fertility_rates(path=str(path))

    # Our fixture has the same rates across countries. Just check filter works.
    r_wales = get_fertility_rates(2024, country="Wales", tables=t)
    r_ew = get_fertility_rates(2024, country="England, Wales and Elsewhere", tables=t)
    # Same rates in the fixture, but both should return a map of the same shape.
    assert set(r_wales.keys()) == set(r_ew.keys())


def test_unknown_country_raises(tmp_path):
    from policyengine_uk_data.targets.sources.ons_fertility import (
        get_fertility_rates,
        load_ons_fertility_rates,
    )

    path = _synthetic_workbook(tmp_path)
    _clear_cache()
    t = load_ons_fertility_rates(path=str(path))

    with pytest.raises(KeyError, match="No ASFR rows for country"):
        get_fertility_rates(2024, country="Scotland", tables=t)


def test_age_dataset_accepts_real_fertility_rates(tmp_path):
    """End-to-end: real ONS rates plug into age_dataset without shape errors."""
    from policyengine_uk.data import UKSingleYearDataset
    from policyengine_uk_data.targets.sources.ons_fertility import (
        get_fertility_rates,
        load_ons_fertility_rates,
    )
    from policyengine_uk_data.utils.demographic_ageing import (
        AGE_COLUMN,
        FEMALE_VALUE,
        age_dataset,
    )

    path = _synthetic_workbook(tmp_path)
    _clear_cache()
    tables = load_ons_fertility_rates(path=str(path))
    fertility = get_fertility_rates(2024, tables=tables)

    # All-female population aged 30: uses the fixture's peak rate of 0.090.
    n = 1_000
    person = pd.DataFrame(
        {
            "person_id": list(range(1, n + 1)),
            "person_benunit_id": list(range(1, n + 1)),
            "person_household_id": list(range(1, n + 1)),
            AGE_COLUMN: [30] * n,
            "gender": [FEMALE_VALUE] * n,
            "employment_income": [0.0] * n,
        }
    )
    benunit = pd.DataFrame({"benunit_id": list(range(1, n + 1))})
    household = pd.DataFrame(
        {
            "household_id": list(range(1, n + 1)),
            "household_weight": [1.0] * n,
        }
    )
    base = UKSingleYearDataset(
        person=person, benunit=benunit, household=household, fiscal_year=2023
    )

    aged = age_dataset(
        base,
        years=1,
        seed=42,
        mortality_rates={},
        fertility_rates=fertility,
    )
    births = len(aged.person) - n
    # With p=0.090 on 1,000 mothers we expect ~90 births; confidence
    # interval ~±3*sqrt(90*0.91) ≈ ±29. Anchor the test comfortably.
    assert 50 <= births <= 140
