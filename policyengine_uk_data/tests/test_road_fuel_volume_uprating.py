"""Tests for road-fuel volume uprating. See #402."""

import pandas as pd

from policyengine_uk_data.datasets.imputations.consumption import (
    IMPUTATIONS,
    uprate_lcfs_table,
)
from policyengine_uk_data.sources.road_fuel_volume import (
    FISCAL_YEAR_AVERAGE_DUTY_RATE,
    HMRC_ROAD_FUEL_CLEARANCES_MLITRES,
    NON_ROAD_FUEL_RECEIPTS_GBP_BN,
    OBR_FUEL_DUTY_RECEIPTS_GBP_BN,
    forecast_road_fuel_clearances_mlitres,
    road_fuel_clearances_mlitres,
    road_fuel_volume_index,
    road_fuel_volume_uprating,
)
from policyengine_uk_data.storage import STORAGE_FOLDER
from policyengine_uk_data.utils.uprating import (
    END_YEAR,
    START_YEAR,
    VOLUME_OVERRIDDEN_VARIABLES,
    _apply_road_fuel_volume_override,
)


def test__given_hmrc_outturn__then_road_fuel_volume_matches_petrol_plus_diesel():
    # Given/When
    volume_2024 = HMRC_ROAD_FUEL_CLEARANCES_MLITRES[2024]

    # Then
    assert volume_2024 == 18_033.7097453447 + 28_293.3873251369


def test__given_obr_receipts__then_forecast_volume_uses_road_fuel_receipts():
    # Given
    year = 2026
    road_fuel_receipts = (
        OBR_FUEL_DUTY_RECEIPTS_GBP_BN[year] - NON_ROAD_FUEL_RECEIPTS_GBP_BN[year]
    )

    # When
    volume = forecast_road_fuel_clearances_mlitres()[year]

    # Then
    assert volume == road_fuel_receipts * 1_000 / FISCAL_YEAR_AVERAGE_DUTY_RATE[year]


def test__given_uprating_window__then_volume_index_covers_all_years():
    # When
    index = road_fuel_volume_index(base_year=START_YEAR, end_year=END_YEAR)

    # Then
    assert set(range(START_YEAR, END_YEAR + 1)).issubset(index)
    assert index[START_YEAR] == 1.0
    assert index[2031] == index[2030]


def test__given_pre_pandemic_base__then_obr_forecast_volume_declines():
    # When
    index = road_fuel_volume_index(base_year=2024, end_year=END_YEAR)

    # Then
    assert index[2030] < 0.85
    assert index[2034] == index[2030]


def test__given_uprating_table__then_only_fuel_rows_are_overridden():
    # Given
    df = pd.DataFrame(
        {year: [1.0, 1.0, 1.0] for year in range(START_YEAR, END_YEAR + 1)},
        index=["petrol_spending", "diesel_spending", "some_other_variable"],
    )

    # When
    out = _apply_road_fuel_volume_override(df.copy())
    expected = road_fuel_volume_index(base_year=START_YEAR, end_year=END_YEAR)

    # Then
    for year in range(START_YEAR, END_YEAR + 1):
        assert out.loc["some_other_variable", year] == 1.0
    for variable in VOLUME_OVERRIDDEN_VARIABLES:
        for year in range(START_YEAR, END_YEAR + 1):
            assert out.loc[variable, year] == round(expected[year], 3)


def test__given_storage_csv__then_fuel_rows_reflect_volume_index():
    # Given
    df = pd.read_csv(STORAGE_FOLDER / "uprating_factors.csv").set_index("Variable")
    expected = road_fuel_volume_index(base_year=START_YEAR, end_year=END_YEAR)

    # Then
    for variable in VOLUME_OVERRIDDEN_VARIABLES:
        assert variable in df.index
        for year in range(START_YEAR, END_YEAR + 1):
            assert df.loc[variable, str(year)] == round(expected[year], 3)


def test__given_lcfs_training_table__then_fuel_uprating_uses_volume_index():
    # Given
    household = pd.DataFrame({variable: [1.0] for variable in IMPUTATIONS})
    household["hbai_household_net_income"] = 1.0
    household["household_gross_income"] = 1.0
    household["employment_income"] = 1.0
    household["self_employment_income"] = 1.0
    household["private_pension_income"] = 1.0

    # When
    out = uprate_lcfs_table(household.copy(), "2024")
    expected = road_fuel_volume_uprating(start_year=2021, end_year=2024)

    # Then
    assert out["petrol_spending"].iloc[0] == expected
    assert out["diesel_spending"].iloc[0] == expected
    assert expected != 1.3


def test__given_year_after_obr_horizon__then_last_forecast_is_carried_forward():
    # When
    series = road_fuel_clearances_mlitres(end_year=END_YEAR)

    # Then
    assert series[2031] == series[2030]
    assert series[2034] == series[2030]
