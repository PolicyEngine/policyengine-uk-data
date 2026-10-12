"""Purchase inputs must retain their stock caps during year materialisation."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from policyengine_uk_data.storage import STORAGE_FOLDER
from policyengine_uk_data.utils.uprating import (
    _apply_property_purchase_uprating_override,
    uprate_dataset,
)

PROPERTY_INPUTS = (
    ("additional_residential_property_purchased", "other_residential_property_value"),
    ("non_residential_property_purchased", "non_residential_property_value"),
)


@pytest.mark.unit
@pytest.mark.parametrize("purchase_name,stock_name", PROPERTY_INPUTS)
@pytest.mark.parametrize("purchase_exists", [False, True])
def test_uprating_generator_adds_or_replaces_purchase_index(
    purchase_name, stock_name, purchase_exists
):
    factors = pd.DataFrame([[1.2, 1.3]], index=[stock_name], columns=[2024, 2025])
    if purchase_exists:
        factors.loc[purchase_name] = [9.0, 10.0]
    updated = _apply_property_purchase_uprating_override(factors)
    np.testing.assert_array_equal(updated.loc[purchase_name], [1.2, 1.3])
    np.testing.assert_array_equal(updated.loc[stock_name], [1.2, 1.3])


@pytest.mark.unit
@pytest.mark.parametrize(
    "filename", ["uprating_factors.csv", "uprating_growth_factors.csv"]
)
@pytest.mark.parametrize("purchase_name,stock_name", PROPERTY_INPUTS)
def test_purchase_uprating_matches_stock(filename, purchase_name, stock_name):
    factors = pd.read_csv(STORAGE_FOLDER / filename, index_col="Variable")
    np.testing.assert_array_equal(factors.loc[purchase_name], factors.loc[stock_name])


@dataclass
class _SyntheticDataset:
    household: pd.DataFrame
    time_period: int

    @property
    def tables(self):
        return [self.household]

    def copy(self):
        return _SyntheticDataset(self.household.copy(), self.time_period)


@pytest.mark.unit
@pytest.mark.parametrize("purchase_name,stock_name", PROPERTY_INPUTS)
def test_base_year_materialisation_preserves_purchase_stock_cap(
    purchase_name, stock_name
):
    stock = np.array([50_000.0, 250_000.0, 1_000_000.0])
    dataset = _SyntheticDataset(
        pd.DataFrame({stock_name: stock, purchase_name: stock}), time_period=2025
    )
    saved = uprate_dataset(dataset, 2024)
    assert saved.time_period == 2024
    assert np.all(saved.household[purchase_name] <= saved.household[stock_name])
    assert np.all(saved.household[purchase_name] < dataset.household[purchase_name])
    restored = uprate_dataset(saved, 2025)
    np.testing.assert_allclose(restored.household[purchase_name], stock)
