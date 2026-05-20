"""Tests for the road-fuel volume override applied to petrol_spending /
diesel_spending uprating. See #402."""

import pandas as pd

from policyengine_uk_data.sources.hmrc_hydrocarbon_oils import (
    ROAD_FUEL_VOLUME_MLITRES,
    road_fuel_volume_index,
)
from policyengine_uk_data.storage import STORAGE_FOLDER
from policyengine_uk_data.utils.uprating import (
    START_YEAR,
    END_YEAR,
    VOLUME_OVERRIDDEN_VARIABLES,
    _apply_road_fuel_volume_override,
)


def test_road_fuel_volume_series_covers_uprating_window():
    """The HMRC road-fuel volume series must span the full uprating window."""
    for year in range(START_YEAR, END_YEAR + 1):
        assert year in ROAD_FUEL_VOLUME_MLITRES, (
            f"road_fuel_volume series missing year {year}; the uprating window "
            f"requires coverage from {START_YEAR} to {END_YEAR}"
        )


def test_road_fuel_volume_index_is_normalised_to_base():
    """Index at the base year must equal 1.0 by construction."""
    idx = road_fuel_volume_index(base_year=START_YEAR)
    assert idx[START_YEAR] == 1.0


def test_road_fuel_volume_does_not_track_cpi():
    """Cumulative volume growth from 2020 to 2029 must be flat or negative,
    contradicting CPI which inflates ~25-30% over the same window. Guards
    against a regression where the override stops being applied."""
    idx = road_fuel_volume_index(base_year=START_YEAR)
    growth_2020_2029 = idx[2029] - idx[2020]
    assert growth_2020_2029 < 0.10, (
        f"road_fuel_volume_index 2020->2029 grew by {growth_2020_2029:.2%}; "
        f"expected near-flat or negative growth consistent with OBR EV-adoption "
        f"forecasts. CPI over the same window grows ~25-30%, so a value much "
        f"above zero suggests the override has been bypassed."
    )


def test_apply_override_replaces_target_variables():
    """The override should replace exactly the petrol/diesel rows and leave
    other rows untouched."""
    df = pd.DataFrame(
        {year: [1.0, 1.0, 1.0] for year in range(START_YEAR, END_YEAR + 1)},
        index=["petrol_spending", "diesel_spending", "some_other_variable"],
    )
    out = _apply_road_fuel_volume_override(df.copy())

    # Other variable untouched
    for year in range(START_YEAR, END_YEAR + 1):
        assert out.loc["some_other_variable", year] == 1.0

    # Petrol/diesel rows now match the volume index
    expected = road_fuel_volume_index(base_year=START_YEAR)
    for variable in VOLUME_OVERRIDDEN_VARIABLES:
        for year in range(START_YEAR, END_YEAR + 1):
            assert out.loc[variable, year] == expected[year], (
                f"{variable} {year}: {out.loc[variable, year]} != {expected[year]}"
            )


def test_storage_csv_reflects_override():
    """The committed ``uprating_factors.csv`` must show the volume-based
    growth for petrol/diesel, not the CPI growth."""
    df = pd.read_csv(STORAGE_FOLDER / "uprating_factors.csv").set_index("Variable")
    expected = road_fuel_volume_index(base_year=START_YEAR)
    for variable in VOLUME_OVERRIDDEN_VARIABLES:
        assert variable in df.index, f"{variable} missing from uprating_factors.csv"
        for year in range(START_YEAR, END_YEAR + 1):
            assert abs(df.loc[variable, str(year)] - expected[year]) < 1e-3, (
                f"{variable} {year}: csv shows {df.loc[variable, str(year)]} "
                f"vs expected volume index {expected[year]}"
            )
