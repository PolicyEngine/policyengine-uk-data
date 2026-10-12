"""Broad Rental Market Area (BRMA) assignment for FRS benefit units.

The FRS identifies only a household's region, but Local Housing Allowance
rates vary by BRMA. Each benefit unit is given a BRMA drawn within its
region, in proportion to the number of private-rented households in each
BRMA with the number of bedrooms matching its LHA category (shared
accommodation and one-bedroom categories both use one-bedroom homes).

The counts come from the censuses (England and Wales 2021, Scotland 2022,
Northern Ireland 2021) mapped to BRMAs; ``storage/BRMA_DATA_SOURCES.md``
gives the sources, method and validation. A BRMA that crosses a region
boundary appears once per region, holding the households on that side.

These weights replace the row counts of the 2019-20 LHA list of rents. Its
Scottish, Welsh and Northern Ireland rows were copies of English BRMAs'
lists, so their counts said nothing about those nations' rental markets.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from policyengine_uk_data.storage import STORAGE_FOLDER

BRMA_HOUSEHOLDS_PATH = STORAGE_FOLDER / "brma_private_rented_households.csv"

# Census bedroom band whose private-rented households weight each LHA category.
LHA_CATEGORY_BEDROOMS = {"A": "1", "B": "1", "C": "2", "D": "3", "E": "4+"}


def load_brma_weights(path: Path = BRMA_HOUSEHOLDS_PATH) -> pd.DataFrame:
    """Return BRMA sampling weights by region and LHA category.

    Columns: ``region``, ``lha_category``, ``brma``, ``weight`` (private-rented
    households). Every row has a positive weight. Northern Ireland's census
    has no bedrooms question, so its rows (bedrooms ``all``) weight every
    category.
    """
    households = pd.read_csv(path, dtype={"bedrooms": str})
    bands = pd.DataFrame(
        LHA_CATEGORY_BEDROOMS.items(), columns=["lha_category", "bedrooms"]
    )
    by_band = bands.merge(households, on="bedrooms")
    any_band = households[households.bedrooms == "all"].merge(
        bands[["lha_category"]], how="cross"
    )
    weights = pd.concat([by_band, any_band], ignore_index=True)
    weights = weights[weights.households > 0]
    return weights.rename(columns={"households": "weight"})[
        ["region", "lha_category", "brma", "weight"]
    ].reset_index(drop=True)


def assign_brmas(
    region: np.ndarray,
    lha_category: np.ndarray,
    rng: np.random.Generator,
    weights: pd.DataFrame | None = None,
) -> np.ndarray:
    """Draw a BRMA for each benefit unit from its region × LHA category cell.

    Args:
        region: Region name of each benefit unit's household.
        lha_category: LHA category (A-E) of each benefit unit.
        rng: Generator for the draws.
        weights: Output of ``load_brma_weights``; loaded from storage if omitted.

    Raises:
        ValueError: If a benefit unit's cell has no BRMA with a positive weight.
    """
    if weights is None:
        weights = load_brma_weights()
    if hasattr(region, "decode_to_str"):  # EnumArray of region codes
        region = region.decode_to_str()
    region = np.asarray(region).astype(str)
    lha_category = np.asarray(lha_category).astype(str)
    brma = np.empty(len(region), dtype=object)
    for (cell_region, category), cell in weights.groupby(
        ["region", "lha_category"], sort=True
    ):
        mask = (region == cell_region) & (lha_category == category)
        if mask.any():
            p = cell.weight.to_numpy(dtype=float)
            brma[mask] = rng.choice(
                cell.brma.to_numpy(), size=mask.sum(), p=p / p.sum()
            )
    missing = pd.isna(brma)
    if missing.any():
        cells = sorted(set(zip(region[missing], lha_category[missing])))
        raise ValueError(f"No BRMA weights for region × LHA category cells {cells}.")
    return brma


def pick_household_brmas(
    brma: np.ndarray, household_id: np.ndarray, rng: np.random.Generator
) -> pd.Series:
    """Give each household the BRMA of one of its benefit units, chosen at random.

    ``brma`` and ``household_id`` are per benefit unit; returns a Series
    indexed by household ID.
    """
    units = pd.DataFrame({"brma": brma, "household_id": household_id})
    return units.groupby("household_id").brma.aggregate(
        lambda x: x.sample(n=1, random_state=rng).iloc[0]
    )
