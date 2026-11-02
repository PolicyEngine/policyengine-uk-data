"""Output Area assignment for cloned FRS records.

Assigns population-weighted random Output Areas to household
clones, with constituency collision avoidance (each clone of
the same household gets a different constituency where
possible) and region constraints (a London household gets a
London OA, a Welsh household a Welsh OA, and so on). The FRS
region is the finest geography the survey records, so it bounds
where a household's OA can be.

Analogous to policyengine-us-data's clone_and_assign.py.
"""

import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from policyengine_uk_data.calibration.oa_crosswalk import load_oa_crosswalk

logger = logging.getLogger(__name__)

# Map FRS country codes to country names.
# FRS uses numeric codes: 1=England, 2=Wales, 3=Scotland,
# 4=Northern Ireland
FRS_COUNTRY_MAP = {
    1: "England",
    2: "Wales",
    3: "Scotland",
    4: "Northern Ireland",
}

COUNTRY_NAME_MAP = {
    "ENGLAND": "England",
    "WALES": "Wales",
    "SCOTLAND": "Scotland",
    "NORTHERN_IRELAND": "Northern Ireland",
}

# FRS regions (the policyengine-uk ``region`` enum) to the region codes
# the OA crosswalk carries: ONS RGN codes in England, and the
# crosswalk's country-level pseudo-codes elsewhere.
FRS_REGION_TO_CODE = {
    "NORTH_EAST": "E12000001",
    "NORTH_WEST": "E12000002",
    "YORKSHIRE": "E12000003",
    "EAST_MIDLANDS": "E12000004",
    "WEST_MIDLANDS": "E12000005",
    "EAST_OF_ENGLAND": "E12000006",
    "LONDON": "E12000007",
    "SOUTH_EAST": "E12000008",
    "SOUTH_WEST": "E12000009",
    "WALES": "W99999999",
    "SCOTLAND": "S99999999",
    "NORTHERN_IRELAND": "N99999999",
}
_REGION_CODE_PREFIX_TO_COUNTRY = {
    "E": "England",
    "W": "Wales",
    "S": "Scotland",
    "N": "Northern Ireland",
}
# Region values that carry no location below the country.
_UNKNOWN_REGIONS = {"", "UNKNOWN"}


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


def _distribution(subset: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Population-weighted sampling frame for a set of OAs."""
    pop = subset["population"].values.astype(np.float64)
    total = pop.sum()
    if total == 0:
        # Uniform if no population data
        probs = np.ones(len(subset)) / len(subset)
    else:
        probs = pop / total

    return {
        "oa_codes": subset["oa_code"].values,
        "constituencies": subset["constituency_code"].values,
        "lsoa_codes": subset["lsoa_code"].values,
        "msoa_codes": subset["msoa_code"].values,
        "la_codes": subset["la_code"].values,
        "region_codes": subset["region_code"].values,
        "probs": probs,
    }


def _load_numeric_crosswalk(crosswalk_path: Optional[str]) -> pd.DataFrame:
    path = Path(crosswalk_path) if crosswalk_path else None
    xw = load_oa_crosswalk(path)
    # Ensure population is numeric
    xw["population"] = pd.to_numeric(xw["population"], errors="coerce").fillna(0)
    return xw


@lru_cache(maxsize=1)
def _load_region_distributions(
    crosswalk_path: Optional[str] = None,
) -> Dict[str, Dict]:
    """Load OA distributions grouped by crosswalk region code.

    Returns:
        Dict mapping region code (e.g. ``E12000007``, ``W99999999``)
        to the same sampling frame as ``_load_country_distributions``,
        with probabilities summing to 1 within the region.
    """
    xw = _load_numeric_crosswalk(crosswalk_path)
    region_codes = xw["region_code"].fillna("").astype(str).str.strip()
    return {
        code: _distribution(xw[region_codes == code])
        for code in sorted(set(region_codes) - {""})
    }


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
    xw = _load_numeric_crosswalk(crosswalk_path)

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

        distributions[country_name] = _distribution(subset)

    return distributions


def _normalise_country(value) -> Optional[str]:
    """Normalise FRS country codes and repo country labels."""
    if pd.isna(value):
        return None

    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return FRS_COUNTRY_MAP.get(int(value))

    text = str(value).strip()
    if not text:
        return None
    if text.isdigit():
        return FRS_COUNTRY_MAP.get(int(text))

    return COUNTRY_NAME_MAP.get(text.upper().replace("-", "_").replace(" ", "_"))


def _normalise_region(value) -> Optional[str]:
    """Map an FRS region name or crosswalk region code to a region code.

    Returns None when the region says nothing below the country
    (missing or ``UNKNOWN``). Raises on anything unrecognised.
    """
    if isinstance(value, bytes):
        value = value.decode()
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return None
    text = str(value).strip().upper()
    if text in _UNKNOWN_REGIONS:
        return None
    if text in FRS_REGION_TO_CODE:
        return FRS_REGION_TO_CODE[text]
    if text in FRS_REGION_TO_CODE.values():
        return text
    raise ValueError(f"Unrecognised household region value: {value!r}")


def assign_random_geography(
    household_countries: np.ndarray,
    n_clones: int = 10,
    seed: int = 42,
    crosswalk_path: Optional[str] = None,
    household_regions: Optional[np.ndarray] = None,
) -> GeographyAssignment:
    """Assign random OA geography to cloned FRS records.

    Each of n_records * n_clones total records gets a random
    Output Area sampled from the population-weighted
    distribution of its household's region (or of its country,
    when no region is given). LA, constituency and region are
    derived from the OA, so they always agree with the
    household's region.

    Constituency collision avoidance: each clone of the same
    household gets a different constituency where possible
    (up to 50 retry iterations), drawing only from the
    household's own region.

    Args:
        household_countries: Array of length n_records with
            FRS country codes (1-4) or country name strings.
        n_clones: Number of clones per household.
        seed: Random seed for reproducibility.
        crosswalk_path: Override crosswalk file path.
        household_regions: Optional array of length n_records
            with FRS region names (e.g. ``LONDON``) or crosswalk
            region codes (e.g. ``E12000007``). Households with a
            region draw OAs from that region only; missing or
            ``UNKNOWN`` regions fall back to the country.

    Returns:
        GeographyAssignment with arrays of length
        n_records * n_clones.
    """
    n_records = len(household_countries)
    n_total = n_records * n_clones

    # Normalise country codes to canonical country names
    countries = np.array(
        [_normalise_country(value) for value in household_countries],
        dtype=object,
    )
    invalid_mask = pd.isna(countries)
    if invalid_mask.any():
        bad_values = sorted(
            {str(value) for value in np.asarray(household_countries)[invalid_mask]}
        )
        raise ValueError(
            "Unrecognised household country values: " + ", ".join(bad_values)
        )

    if household_regions is None:
        regions = np.full(n_records, None, dtype=object)
    else:
        if len(household_regions) != n_records:
            raise ValueError(
                f"household_regions has {len(household_regions)} values, "
                f"expected {n_records}."
            )
        regions = np.array(
            [_normalise_region(value) for value in household_regions],
            dtype=object,
        )
        mismatched = sorted(
            {
                f"{region} in {country}"
                for region, country in zip(regions, countries)
                if region is not None
                and _REGION_CODE_PREFIX_TO_COUNTRY[region[0]] != country
            }
        )
        if mismatched:
            raise ValueError(
                "Household regions disagree with household countries: "
                + ", ".join(mismatched)
            )
        n_unknown = sum(region is None for region in regions)
        if n_unknown:
            logger.warning(
                "%d households have no region below the country; "
                "sampling their OAs country-wide",
                n_unknown,
            )

    path_key = str(crosswalk_path) if crosswalk_path else None
    country_distributions = _load_country_distributions(path_key)
    region_distributions = (
        _load_region_distributions(path_key) if household_regions is not None else {}
    )
    # A region that holds every OA of its country (Wales and Scotland in
    # the crosswalk) samples from the same pool as the country, so it
    # shares the country's stratum: its households then draw exactly as a
    # country-only call would draw them.
    sole_region_country = {}
    for country, dist in country_distributions.items():
        codes = set(pd.Series(dist["region_codes"]).fillna("").astype(str).str.strip())
        if len(codes) == 1:
            (code,) = codes
            region_dist = region_distributions.get(code)
            if region_dist is not None and len(region_dist["oa_codes"]) == len(
                dist["oa_codes"]
            ):
                sole_region_country[code] = country

    # Sampling stratum per household: its region code where known,
    # otherwise its country name.
    strata = np.array(
        [
            country if region is None else sole_region_country.get(region, region)
            for region, country in zip(regions, countries)
        ],
        dtype=object,
    )
    distributions = {}
    missing_distributions = []
    for stratum in sorted(set(strata)):
        is_country = stratum in COUNTRY_NAME_MAP.values()
        source = country_distributions if is_country else region_distributions
        if stratum in source:
            distributions[stratum] = source[stratum]
        elif is_country:
            missing_distributions.append(stratum)
        else:
            country = _REGION_CODE_PREFIX_TO_COUNTRY[stratum[0]]
            missing_distributions.append(f"{stratum} ({country})")
    if missing_distributions:
        raise ValueError(
            "No OA distribution available for: " + ", ".join(missing_distributions)
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

    # Track assigned constituencies per record for
    # collision avoidance
    assigned_const = np.full((n_clones, n_records), "", dtype=object)

    for clone_idx in range(n_clones):
        start = clone_idx * n_records

        for stratum, dist in distributions.items():
            hh_mask = strata == stratum
            n_hh = hh_mask.sum()

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
            store_idx = start + positions
            oa_codes[store_idx] = dist["oa_codes"][indices]
            constituency_codes[store_idx] = dist["constituencies"][indices]
            lsoa_codes[store_idx] = dist["lsoa_codes"][indices]
            msoa_codes[store_idx] = dist["msoa_codes"][indices]
            la_codes[store_idx] = dist["la_codes"][indices]
            region_codes[store_idx] = dist["region_codes"][indices]
            country_names[store_idx] = countries[positions]

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
