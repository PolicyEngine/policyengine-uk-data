"""Clone-and-assign: duplicate FRS households and assign OA geography.

Each FRS household is cloned N times. Each clone gets a different
Output Area (population-weighted, country-constrained, with
constituency collision avoidance). Weights are divided by N so
population totals are preserved.

This is the UK equivalent of policyengine-us-data's clone-and-assign
approach (PRs #457, #531).

Inserted into the pipeline after imputations and before calibration.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from policyengine_uk.data import UKSingleYearDataset

from policyengine_uk_data.calibration.oa_assignment import (
    assign_random_geography,
    GeographyAssignment,
)

logger = logging.getLogger(__name__)

# FRS region values that map to each country
_REGION_TO_COUNTRY_CODE = {
    "NORTH_EAST": 1,
    "NORTH_WEST": 1,
    "YORKSHIRE": 1,
    "EAST_MIDLANDS": 1,
    "WEST_MIDLANDS": 1,
    "EAST_OF_ENGLAND": 1,
    "LONDON": 1,
    "SOUTH_EAST": 1,
    "SOUTH_WEST": 1,
    "WALES": 2,
    "SCOTLAND": 3,
    "NORTHERN_IRELAND": 4,
    "UNKNOWN": 1,  # Default to England
}


def _household_country_codes(dataset: UKSingleYearDataset) -> np.ndarray:
    """Extract FRS country codes (1-4) from household region."""
    regions = dataset.household["region"].values
    codes = np.array(
        [_REGION_TO_COUNTRY_CODE.get(str(r), 1) for r in regions],
        dtype=np.int32,
    )
    return codes


def _remap_ids(
    old_ids: np.ndarray,
    clone_idx: int,
    id_multiplier: int,
) -> np.ndarray:
    """Create new unique IDs for a clone by adding an offset.

    Uses clone_idx * id_multiplier as the offset to ensure
    no collisions between clones. Clone 0 keeps original IDs.
    """
    if clone_idx == 0:
        return old_ids.copy()
    offset = clone_idx * id_multiplier
    return old_ids + offset


def clone_and_assign(
    dataset: UKSingleYearDataset,
    n_clones: int = 10,
    seed: int = 42,
    crosswalk_path: Optional[str] = None,
) -> UKSingleYearDataset:
    """Clone each FRS household N times and assign OA geography.

    Each clone gets a population-weighted random Output Area,
    constrained to its country, with constituency collision
    avoidance across clones.

    Household weights are divided by n_clones so aggregate
    population totals are preserved.

    Args:
        dataset: Input FRS dataset (post-imputation).
        n_clones: Number of clones per household.
        seed: Random seed for reproducibility.
        crosswalk_path: Override path to OA crosswalk file.

    Returns:
        New dataset with n_clones * n_households households.
    """
    hh = dataset.household
    person = dataset.person
    benunit = dataset.benunit

    n_households = len(hh)
    n_persons = len(person)
    n_benunits = len(benunit)

    logger.info(
        "Cloning %d households x %d = %d total records",
        n_households,
        n_clones,
        n_households * n_clones,
    )

    # Get country codes for OA assignment
    country_codes = _household_country_codes(dataset)

    # Assign OA geography to all clones
    geography = assign_random_geography(
        household_countries=country_codes,
        n_clones=n_clones,
        seed=seed,
        crosswalk_path=crosswalk_path,
    )

    # ID offset must be larger than any existing ID to avoid collisions.
    # person_id = household_id * 1000 + person_idx (largest IDs)
    # benunit_id = household_id * 100 + benunit_idx
    max_id = max(
        int(hh["household_id"].max()),
        int(person["person_id"].max()),
        int(benunit["benunit_id"].max()),
    )
    # Round up to next power of 10 for clean offsets
    id_multiplier = 10 ** len(str(max_id))

    hh_id_col = hh["household_id"].values
    person_hh_id_col = person["person_household_id"].values
    person_id_col = person["person_id"].values
    benunit_id_col = benunit["benunit_id"].values

    # Build mapping from household_id to person/benunit indices
    # for efficient cloning
    hh_to_person_idx = {}
    for idx, hh_id in enumerate(person_hh_id_col):
        hh_to_person_idx.setdefault(hh_id, []).append(idx)

    # benunit_id = household_id * 100 + benunit_idx, so we can
    # derive household_id from benunit_id
    benunit_hh_ids = benunit_id_col // 100
    hh_to_benunit_idx = {}
    for idx, hh_id in enumerate(benunit_hh_ids):
        hh_to_benunit_idx.setdefault(hh_id, []).append(idx)

    # Build cloned dataframes
    hh_dfs = []
    person_dfs = []
    benunit_dfs = []

    for clone_idx in range(n_clones):
        # Remap IDs
        new_hh_ids = _remap_ids(hh_id_col, clone_idx, id_multiplier)

        # Clone household table
        hh_clone = hh.copy()
        hh_clone["household_id"] = new_hh_ids
        hh_clone["household_weight"] = hh["household_weight"].values / n_clones

        # Add geography columns from assignment
        start = clone_idx * n_households
        end = start + n_households
        hh_clone["oa_code"] = geography.oa_code[start:end]
        hh_clone["lsoa_code"] = geography.lsoa_code[start:end]
        hh_clone["msoa_code"] = geography.msoa_code[start:end]
        hh_clone["la_code_oa"] = geography.la_code[start:end]
        hh_clone["constituency_code_oa"] = geography.constituency_code[start:end]
        hh_clone["region_code_oa"] = geography.region_code[start:end]
        hh_clone["clone_index"] = clone_idx

        hh_dfs.append(hh_clone)

        # Clone person table
        person_clone = person.copy()
        person_clone["person_id"] = _remap_ids(person_id_col, clone_idx, id_multiplier)
        person_clone["person_household_id"] = _remap_ids(
            person_hh_id_col, clone_idx, id_multiplier
        )
        # person_benunit_id needs remapping too
        person_clone["person_benunit_id"] = _remap_ids(
            person["person_benunit_id"].values,
            clone_idx,
            id_multiplier,
        )
        person_dfs.append(person_clone)

        # Clone benunit table
        benunit_clone = benunit.copy()
        benunit_clone["benunit_id"] = _remap_ids(
            benunit_id_col, clone_idx, id_multiplier
        )
        benunit_dfs.append(benunit_clone)

    # Concatenate all clones
    new_hh = pd.concat(hh_dfs, ignore_index=True)
    new_person = pd.concat(person_dfs, ignore_index=True)
    new_benunit = pd.concat(benunit_dfs, ignore_index=True)

    logger.info(
        "Cloned dataset: %d households, %d persons, %d benunits",
        len(new_hh),
        len(new_person),
        len(new_benunit),
    )

    result = UKSingleYearDataset(
        person=new_person,
        benunit=new_benunit,
        household=new_hh,
        fiscal_year=dataset.time_period,
    )

    return result
