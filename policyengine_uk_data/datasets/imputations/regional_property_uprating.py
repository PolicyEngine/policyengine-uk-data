"""
Regional calibration of main_residence_value after WAS imputation.

The WAS QRF imputation compresses regional property values — London is
undervalued by ~41% and the North East overvalued by ~39% relative to
UK HPI Dec 2025.  This module rescales main_residence_value (and
property_wealth proportionally) so that regional means match official
average house prices, following the same pattern as
``_calibrate_energy_to_need`` in ``consumption.py``.
"""

import numpy as np
import pandas as pd
from policyengine_uk.data import UKSingleYearDataset

from policyengine_uk_data.storage import STORAGE_FOLDER

REGIONAL_LAND_VALUES_CSV = STORAGE_FOLDER / "regional_land_values.csv"


def _load_regional_house_prices() -> dict[str, float]:
    """Return {REGION: avg_house_price} from the CSV used by targets."""
    df = pd.read_csv(REGIONAL_LAND_VALUES_CSV)
    return dict(zip(df["region"], df["avg_house_price"]))


def _calibrate_property_to_hpi(household: pd.DataFrame) -> pd.DataFrame:
    """Rescale imputed main_residence_value to match UK HPI regional means.

    For each region, computes the ratio of the HPI average house price
    to the imputed mean main_residence_value (among owner-occupiers) and
    applies it multiplicatively.  This preserves within-region
    distributional shape while anchoring the level to admin data.

    property_wealth is adjusted by the same factor so that the land-to-
    property ratio stays consistent.
    """
    hpi_prices = _load_regional_house_prices()
    region = household["region"].values
    main_res = household["main_residence_value"].values.astype(float)

    household = household.copy()

    for reg, hpi_price in hpi_prices.items():
        mask = region == reg
        if mask.sum() == 0:
            continue

        owners_mask = mask & (main_res > 0)
        if owners_mask.sum() == 0:
            continue

        imputed_mean = household["main_residence_value"][owners_mask].mean()
        if imputed_mean <= 0:
            continue

        factor = hpi_price / imputed_mean
        household.loc[owners_mask, "main_residence_value"] *= factor
        household.loc[owners_mask, "property_wealth"] *= factor

    return household


def uprate_property_by_region(
    dataset: UKSingleYearDataset,
) -> UKSingleYearDataset:
    """Scale main_residence_value so regional means match UK HPI prices.

    Args:
        dataset: Dataset with wealth already imputed.

    Returns:
        Dataset with regionally calibrated property values.
    """
    dataset = dataset.copy()
    household = pd.DataFrame(dataset.household)
    household = _calibrate_property_to_hpi(household)

    dataset.household["main_residence_value"] = household[
        "main_residence_value"
    ].values
    dataset.household["property_wealth"] = household["property_wealth"].values

    dataset.validate()
    return dataset
