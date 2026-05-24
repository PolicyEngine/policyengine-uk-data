"""Tests for road-fuel volume uprating. See #402."""

import pandas as pd
import pytest
from policyengine_uk.data import UKSingleYearDataset

from policyengine_uk_data.datasets.imputations.consumption import (
    CONSUMPTION_MODEL_FILENAME,
    IMPUTATIONS,
    _fuel_litre_proxy_mlitres,
    calibrate_dataset_fuel_litre_proxies_to_road_fuel,
    calibrate_fuel_litre_proxies_to_road_fuel,
    fuel_spending_litre_proxy_uprating,
    uprate_lcfs_table,
)
from policyengine_uk_data.datasets.private_releases import (
    CURRENT_LCFS_RELEASE,
    CURRENT_WAS_RELEASE,
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
    HOUSEHOLD_WEIGHT_UPRATING_INDEX,
    START_YEAR,
    VOLUME_OVERRIDDEN_VARIABLES,
    _apply_household_weight_uprating_override,
    _apply_road_fuel_litre_proxy_override,
    fuel_spending_litre_proxy_index,
    uprate_dataset,
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
        {year: [1.0, 1.0, 1.0, 1.0] for year in range(START_YEAR, END_YEAR + 1)},
        index=[
            "petrol_spending",
            "diesel_spending",
            "household_weight",
            "some_other_variable",
        ],
    )
    df.loc["household_weight", 2024] = 1.2

    # When
    out = _apply_road_fuel_litre_proxy_override(df.copy())

    # Then
    for year in range(START_YEAR, END_YEAR + 1):
        assert out.loc["some_other_variable", year] == 1.0
        assert out.loc["household_weight", year] == df.loc["household_weight", year]
    for variable in VOLUME_OVERRIDDEN_VARIABLES:
        expected = fuel_spending_litre_proxy_index(
            variable=variable,
            base_year=START_YEAR,
            end_year=END_YEAR,
            household_weight_index=df.loc["household_weight"],
        )
        for year in range(START_YEAR, END_YEAR + 1):
            assert out.loc[variable, year] == round(expected[year], 3)


def test__given_generated_uprating_table__then_household_weight_row_is_restored():
    # Given
    df = pd.DataFrame(
        {year: [1.0, 2.0] for year in range(START_YEAR, END_YEAR + 1)},
        index=["household_weight", "some_other_variable"],
    )

    # When
    out = _apply_household_weight_uprating_override(df.copy())

    # Then
    for year in range(START_YEAR, END_YEAR + 1):
        assert (
            out.loc["household_weight", year] == HOUSEHOLD_WEIGHT_UPRATING_INDEX[year]
        )
        assert out.loc["some_other_variable", year] == 2.0


def test__given_storage_csv__then_fuel_rows_reflect_litre_proxy_index():
    # Given
    df = pd.read_csv(STORAGE_FOLDER / "uprating_factors.csv").set_index("Variable")

    # Then
    for variable in VOLUME_OVERRIDDEN_VARIABLES:
        expected = fuel_spending_litre_proxy_index(
            variable=variable,
            base_year=START_YEAR,
            end_year=END_YEAR,
            household_weight_index=df.loc["household_weight"],
        )
        assert variable in df.index
        for year in range(START_YEAR, END_YEAR + 1):
            assert df.loc[variable, str(year)] == round(expected[year], 3)


def test__given_storage_csv__then_household_weight_row_is_unchanged():
    # Given
    df = pd.read_csv(STORAGE_FOLDER / "uprating_factors.csv").set_index("Variable")

    # Then
    assert df.loc[
        "household_weight", [str(year) for year in range(START_YEAR, END_YEAR + 1)]
    ].tolist() == [
        HOUSEHOLD_WEIGHT_UPRATING_INDEX[year]
        for year in range(START_YEAR, END_YEAR + 1)
    ]


def test__given_lcfs_training_table__then_fuel_uprating_preserves_litre_proxy():
    # Given
    household = pd.DataFrame({variable: [1.0] for variable in IMPUTATIONS})
    household["hbai_household_net_income"] = 1.0
    household["household_gross_income"] = 1.0
    household["employment_income"] = 1.0
    household["self_employment_income"] = 1.0
    household["private_pension_income"] = 1.0

    # When
    out = uprate_lcfs_table(household.copy(), "2024")
    start_year = CURRENT_LCFS_RELEASE.fuel_price_year
    petrol_expected = fuel_spending_litre_proxy_uprating(
        variable="petrol_spending",
        start_year=start_year,
        end_year=2024,
    )
    diesel_expected = fuel_spending_litre_proxy_uprating(
        variable="diesel_spending",
        start_year=start_year,
        end_year=2024,
    )
    volume_only = road_fuel_volume_uprating(start_year=start_year, end_year=2024)

    # Then
    assert out["petrol_spending"].iloc[0] == petrol_expected
    assert out["diesel_spending"].iloc[0] == diesel_expected
    assert petrol_expected != volume_only
    assert diesel_expected != volume_only
    assert petrol_expected != 1.3


def test__given_fuel_method_change__then_consumption_model_filename_is_versioned():
    # Then
    assert CONSUMPTION_MODEL_FILENAME != "consumption.pkl"
    assert "fuel_litre_proxy" in CONSUMPTION_MODEL_FILENAME
    assert CURRENT_LCFS_RELEASE.name in CONSUMPTION_MODEL_FILENAME
    assert CURRENT_WAS_RELEASE.name in CONSUMPTION_MODEL_FILENAME


def test__given_obr_2027_volume__then_rate_difference_matches_cost_benchmark():
    # Given
    road_fuel_bn_litres = road_fuel_clearances_mlitres()[2027] / 1_000
    broad_2027_rate_gap_gbp_per_litre = FISCAL_YEAR_AVERAGE_DUTY_RATE[2027] - 0.5295

    # When
    benchmark_cost_gbp_bn = road_fuel_bn_litres * broad_2027_rate_gap_gbp_per_litre

    # Then
    assert round(benchmark_cost_gbp_bn, 2) == 3.12


def test__given_imputed_fuel_proxies__then_calibration_matches_road_fuel_litres():
    # Given
    from policyengine_uk.system import system

    year = 2027
    target_mlitres = road_fuel_clearances_mlitres()[year]
    petrol_price = system.parameters.household.consumption.fuel.prices.petrol(year)
    diesel_price = system.parameters.household.consumption.fuel.prices.diesel(year)
    household = pd.DataFrame(
        {
            "household_weight": [1.0, 1.0],
            "petrol_spending": [
                target_mlitres * 1_000_000 * petrol_price,
                0.0,
            ],
            "diesel_spending": [
                0.0,
                target_mlitres * 1_000_000 * diesel_price,
            ],
        }
    )

    # When
    scale = calibrate_fuel_litre_proxies_to_road_fuel(household, year)

    # Then
    assert scale == pytest.approx(0.5)
    assert _fuel_litre_proxy_mlitres(household, year) == pytest.approx(target_mlitres)


def test__given_final_weight_changes__then_dataset_calibration_matches_litres():
    # Given
    from policyengine_uk.system import system

    year = 2023
    target_mlitres = road_fuel_clearances_mlitres()[year]
    petrol_price = system.parameters.household.consumption.fuel.prices.petrol(year)
    dataset = _minimal_fuel_dataset(
        fiscal_year=year,
        petrol_spending=target_mlitres * 1_000_000 * petrol_price,
        diesel_spending=0.0,
        household_weight=2.0,
    )

    # When
    scale = calibrate_dataset_fuel_litre_proxies_to_road_fuel(dataset)

    # Then
    assert scale == pytest.approx(0.5)
    assert _fuel_litre_proxy_mlitres(dataset.household, year) == pytest.approx(
        target_mlitres
    )


def test__given_calibrated_dataset_uprated__then_litres_track_forecast_target():
    # Given
    from policyengine_uk.system import system

    start_year = 2023
    target_year = 2027
    start_mlitres = road_fuel_clearances_mlitres()[start_year]
    petrol_price = system.parameters.household.consumption.fuel.prices.petrol(
        start_year
    )
    dataset = _minimal_fuel_dataset(
        fiscal_year=start_year,
        petrol_spending=start_mlitres * 1_000_000 * petrol_price,
        diesel_spending=0.0,
    )
    calibrate_dataset_fuel_litre_proxies_to_road_fuel(dataset)

    # When
    uprated = uprate_dataset(dataset, target_year)

    # Then
    assert _fuel_litre_proxy_mlitres(
        uprated.household,
        target_year,
    ) == pytest.approx(road_fuel_clearances_mlitres()[target_year], rel=1e-3)


def test__given_year_after_obr_horizon__then_last_forecast_is_carried_forward():
    # When
    series = road_fuel_clearances_mlitres(end_year=END_YEAR)

    # Then
    assert series[2031] == series[2030]
    assert series[2034] == series[2030]


def _minimal_fuel_dataset(
    *,
    fiscal_year: int,
    petrol_spending: float,
    diesel_spending: float,
    household_weight: float = 1.0,
) -> UKSingleYearDataset:
    return UKSingleYearDataset(
        person=pd.DataFrame(
            {
                "person_id": [0],
                "benunit_id": [0],
                "household_id": [0],
            }
        ),
        benunit=pd.DataFrame(
            {
                "benunit_id": [0],
                "household_id": [0],
            }
        ),
        household=pd.DataFrame(
            {
                "household_id": [0],
                "household_weight": [household_weight],
                "petrol_spending": [petrol_spending],
                "diesel_spending": [diesel_spending],
            }
        ),
        fiscal_year=fiscal_year,
    )
