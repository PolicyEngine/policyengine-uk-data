"""Output Area assignment for cloned FRS records.

Assigns population-weighted random Output Areas to household
clones, with constituency collision avoidance (each clone of
the same household gets a different constituency where
possible) and country constraints (English households get
English OAs only, etc.).

Analogous to policyengine-us-data's clone_and_assign.py.
"""

import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from policyengine_uk_data.calibration.oa_crosswalk import (
    CROSSWALK_PATH,
    load_oa_crosswalk,
)

logger = logging.getLogger(__name__)

# Country name → OA code prefix mapping
COUNTRY_PREFIXES = {
    "England": "E",
    "Wales": "W",
    "Scotland": "S",
    "Northern Ireland": "N",  # NI DZ codes start with digits
}

# Map FRS country codes to country names.
# FRS uses numeric codes: 1=England, 2=Wales, 3=Scotland,
# 4=Northern Ireland
FRS_COUNTRY_MAP = {
    1: "England",
    2: "Wales",
    3: "Scotland",
    4: "Northern Ireland",
}


@dataclass
class GeographyAssignment:
    """Random geography assignment for cloned FRS records.

    All arrays have length n_records * n_clones.
    Index i corresponds to clone (i // n_records),
    record (i % n_records).
    """

    oa_code: np.ndarray  # str, OA/DZ codes
    lsoa_code: np.ndarray  # str, LSOA/DZ codes
    msoa_code: np.ndarray  # str, MSOA/IZ codes
    la_code: np.ndarray  # str, LA codes
    constituency_code: np.ndarray  # str, constituency codes
    region_code: np.ndarray  # str, region codes
    country: np.ndarray  # str, country names
    n_records: int
    n_clones: int


@lru_cache(maxsize=1)
def _load_country_distributions(
    crosswalk_path: Optional[str] = None,
) -> Dict[str, Dict]:
    """Load OA distributions grouped by country.

    Returns:
        Dict mapping country name to dict with keys:
        - oa_codes: np.ndarray of OA code strings
        - constituencies: np.ndarray of constituency codes
        - probs: np.ndarray of population-weighted
          probabilities (sum to 1 within country)
        - crosswalk_idx: np.ndarray of indices into the
          full crosswalk DataFrame
    """
    path = Path(crosswalk_path) if crosswalk_path else None
    xw = load_oa_crosswalk(path)

    # Ensure population is numeric
    xw["population"] = pd.to_numeric(xw["population"], errors="coerce").fillna(0)

    distributions = {}
    for country_name in [
        "England",
        "Wales",
        "Scotland",
        "Northern Ireland",
    ]:
        mask = xw["country"] == country_name
        subset = xw[mask].copy()

        if len(subset) == 0:
            logger.warning(f"No OAs found for {country_name}")
            continue

        pop = subset["population"].values.astype(np.float64)
        total = pop.sum()
        if total == 0:
            # Uniform if no population data
            probs = np.ones(len(subset)) / len(subset)
        else:
            probs = pop / total

        distributions[country_name] = {
            "oa_codes": subset["oa_code"].values,
            "constituencies": subset["constituency_code"].values,
            "lsoa_codes": subset["lsoa_code"].values,
            "msoa_codes": subset["msoa_code"].values,
            "la_codes": subset["la_code"].values,
            "region_codes": subset["region_code"].values,
            "probs": probs,
        }

    return distributions


def assign_random_geography(
    household_countries: np.ndarray,
    n_clones: int = 10,
    seed: int = 42,
    crosswalk_path: Optional[str] = None,
) -> GeographyAssignment:
    """Assign random OA geography to cloned FRS records.

    Each of n_records * n_clones total records gets a random
    Output Area sampled from the country-specific
    population-weighted distribution. LA, constituency,
    region are derived from the OA.

    Constituency collision avoidance: each clone of the same
    household gets a different constituency where possible
    (up to 50 retry iterations).

    Args:
        household_countries: Array of length n_records with
            FRS country codes (1-4) or country name strings.
        n_clones: Number of clones per household.
        seed: Random seed for reproducibility.
        crosswalk_path: Override crosswalk file path.

    Returns:
        GeographyAssignment with arrays of length
        n_records * n_clones.
    """
    n_records = len(household_countries)
    n_total = n_records * n_clones

    # Normalise country codes to names
    if np.issubdtype(household_countries.dtype, np.integer):
        countries = np.array(
            [FRS_COUNTRY_MAP.get(int(c), "England") for c in household_countries]
        )
    else:
        countries = np.asarray(household_countries, dtype=str)

    distributions = _load_country_distributions(
        str(crosswalk_path) if crosswalk_path else None
    )

    rng = np.random.default_rng(seed)

    # Output arrays
    oa_codes = np.empty(n_total, dtype=object)
    constituency_codes = np.empty(n_total, dtype=object)
    lsoa_codes = np.empty(n_total, dtype=object)
    msoa_codes = np.empty(n_total, dtype=object)
    la_codes = np.empty(n_total, dtype=object)
    region_codes = np.empty(n_total, dtype=object)
    country_names = np.empty(n_total, dtype=object)

    # Group households by country for efficient sampling
    unique_countries = np.unique(countries)

    # Track assigned constituencies per record for
    # collision avoidance
    assigned_const = np.empty((n_clones, n_records), dtype=object)

    for clone_idx in range(n_clones):
        start = clone_idx * n_records
        end = start + n_records

        for country_name in unique_countries:
            if country_name not in distributions:
                logger.warning(f"No distribution for {country_name}, skipping")
                continue

            dist = distributions[country_name]
            hh_mask = countries == country_name
            n_hh = hh_mask.sum()

            if n_hh == 0:
                continue

            # Sample OAs
            indices = rng.choice(
                len(dist["oa_codes"]),
                size=n_hh,
                p=dist["probs"],
            )

            sampled_const = dist["constituencies"][indices]

            # Constituency collision avoidance
            if clone_idx > 0:
                # Find records where we've seen this
                # constituency before
                hh_positions = np.where(hh_mask)[0]
                collisions = np.zeros(n_hh, dtype=bool)
                for prev in range(clone_idx):
                    prev_const = assigned_const[prev, hh_positions]
                    collisions |= sampled_const == prev_const

                for _ in range(50):
                    n_bad = collisions.sum()
                    if n_bad == 0:
                        break
                    new_idx = rng.choice(
                        len(dist["oa_codes"]),
                        size=n_bad,
                        p=dist["probs"],
                    )
                    indices[collisions] = new_idx
                    sampled_const = dist["constituencies"][indices]
                    collisions = np.zeros(n_hh, dtype=bool)
                    for prev in range(clone_idx):
                        prev_const = assigned_const[prev, hh_positions]
                        collisions |= sampled_const == prev_const

            # Store results
            positions = np.where(hh_mask)[0]
            for i, pos in enumerate(positions):
                idx = start + pos
                oa_codes[idx] = dist["oa_codes"][indices[i]]
                constituency_codes[idx] = dist["constituencies"][indices[i]]
                lsoa_codes[idx] = dist["lsoa_codes"][indices[i]]
                msoa_codes[idx] = dist["msoa_codes"][indices[i]]
                la_codes[idx] = dist["la_codes"][indices[i]]
                region_codes[idx] = dist["region_codes"][indices[i]]
                country_names[idx] = country_name

            assigned_const[clone_idx, positions] = sampled_const

    return GeographyAssignment(
        oa_code=oa_codes,
        lsoa_code=lsoa_codes,
        msoa_code=msoa_codes,
        la_code=la_codes,
        constituency_code=constituency_codes,
        region_code=region_codes,
        country=country_names,
        n_records=n_records,
        n_clones=n_clones,
    )


def save_geography(geography: GeographyAssignment, path: Path) -> None:
    """Save a GeographyAssignment to a compressed .npz file.

    Args:
        geography: The geography assignment to save.
        path: Output file path (should end in .npz).
    """
    np.savez_compressed(
        path,
        oa_code=geography.oa_code,
        lsoa_code=geography.lsoa_code,
        msoa_code=geography.msoa_code,
        la_code=geography.la_code,
        constituency_code=geography.constituency_code,
        region_code=geography.region_code,
        country=geography.country,
        n_records=np.array([geography.n_records]),
        n_clones=np.array([geography.n_clones]),
    )


def load_geography(path: Path) -> GeographyAssignment:
    """Load a GeographyAssignment from a .npz file.

    Args:
        path: Path to the .npz file.

    Returns:
        GeographyAssignment with all fields restored.
    """
    data = np.load(path, allow_pickle=True)
    return GeographyAssignment(
        oa_code=data["oa_code"],
        lsoa_code=data["lsoa_code"],
        msoa_code=data["msoa_code"],
        la_code=data["la_code"],
        constituency_code=data["constituency_code"],
        region_code=data["region_code"],
        country=data["country"],
        n_records=int(data["n_records"][0]),
        n_clones=int(data["n_clones"][0]),
    )
