"""Tests for regional property value calibration."""

import numpy as np
import pandas as pd
import pytest

from policyengine_uk_data.datasets.imputations.regional_property_uprating import (
    _load_regional_house_prices,
    _calibrate_property_to_hpi,
)


def test_csv_loads_all_regions():
    prices = _load_regional_house_prices()
    assert "LONDON" in prices
    assert "NORTH_EAST" in prices
    assert len(prices) == 11  # 11 GB regions, no NI


def test_london_highest_price():
    prices = _load_regional_house_prices()
    assert prices["LONDON"] == max(prices.values())


def test_all_prices_positive():
    prices = _load_regional_house_prices()
    for region, price in prices.items():
        assert price > 0, f"{region} has non-positive price {price}"


def test_calibration_rescales_means():
    """After calibration, owner means should match HPI targets."""
    prices = _load_regional_house_prices()
    rng = np.random.default_rng(42)

    rows = []
    for region, hpi_price in prices.items():
        n = 50
        # Deliberately compress: all regions get ~250k mean
        values = rng.normal(250_000, 50_000, size=n).clip(0)
        rows.extend(
            {
                "region": region,
                "main_residence_value": v,
                "property_wealth": v * 1.1,
                "household_weight": 1.0,
            }
            for v in values
        )
    # Add some renters (zero main_residence_value)
    for region in prices:
        rows.extend(
            {
                "region": region,
                "main_residence_value": 0.0,
                "property_wealth": 0.0,
                "household_weight": 1.0,
            }
            for _ in range(10)
        )

    df = pd.DataFrame(rows)
    calibrated = _calibrate_property_to_hpi(df)

    for region, hpi_price in prices.items():
        mask = (calibrated["region"] == region) & (
            calibrated["main_residence_value"] > 0
        )
        calibrated_mean = calibrated.loc[mask, "main_residence_value"].mean()
        assert abs(calibrated_mean - hpi_price) / hpi_price < 0.01, (
            f"{region}: expected ~{hpi_price}, got {calibrated_mean:.0f}"
        )


def test_renters_unchanged():
    """Households with zero main_residence_value should not be scaled."""
    prices = _load_regional_house_prices()
    df = pd.DataFrame(
        {
            "region": ["LONDON", "LONDON", "LONDON"],
            "main_residence_value": [500_000.0, 0.0, 0.0],
            "property_wealth": [500_000.0, 0.0, 0.0],
            "household_weight": [1.0, 1.0, 1.0],
        }
    )
    calibrated = _calibrate_property_to_hpi(df)
    assert calibrated.iloc[1]["main_residence_value"] == 0.0
    assert calibrated.iloc[2]["main_residence_value"] == 0.0


def test_property_wealth_scales_proportionally():
    """property_wealth should scale by the same factor as main_residence_value."""
    prices = _load_regional_house_prices()
    df = pd.DataFrame(
        {
            "region": ["LONDON", "LONDON"],
            "main_residence_value": [400_000.0, 300_000.0],
            "property_wealth": [500_000.0, 400_000.0],
            "household_weight": [1.0, 1.0],
        }
    )
    original_ratio = df["property_wealth"].values / df["main_residence_value"].values
    calibrated = _calibrate_property_to_hpi(df)
    new_ratio = (
        calibrated["property_wealth"].values / calibrated["main_residence_value"].values
    )
    np.testing.assert_allclose(original_ratio, new_ratio, rtol=1e-10)
