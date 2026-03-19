"""Publish per-area H5 files from sparse L0-calibrated weights.

After L0 calibration (Phase 3) produces a sparse weight vector over the
cloned dataset, this module extracts per-area subsets — each H5 contains
only the households assigned to that area, with their calibrated weights.

Supports two modes:
- Local: sequential generation to ``storage/local_h5s/{area_type}/``
- Modal: parallel generation via Modal for ~180K OA files

Phase 6 of the OA calibration pipeline.
US reference: policyengine-us-data PR #465 (modal).
"""

import logging
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import pandas as pd

from policyengine_uk_data.storage import STORAGE_FOLDER

logger = logging.getLogger(__name__)

LOCAL_H5_DIR = STORAGE_FOLDER / "local_h5s"


def _get_area_household_indices(
    dataset,
    area_type: str,
    area_codes: list[str],
) -> dict[str, np.ndarray]:
    """Map each area code to its household indices in the cloned dataset.

    Args:
        dataset: Cloned dataset with OA geography columns.
        area_type: "constituency" or "la".
        area_codes: List of canonical area codes.

    Returns:
        Dict mapping area code to array of household row indices.
    """
    geo_col = "constituency_code_oa" if area_type == "constituency" else "la_code_oa"
    hh_codes = dataset.household[geo_col].values

    area_to_indices: dict[str, list[int]] = {code: [] for code in area_codes}
    for j, code in enumerate(hh_codes):
        code_str = str(code)
        if code_str in area_to_indices:
            area_to_indices[code_str].append(j)

    return {
        code: np.array(indices, dtype=np.int64)
        for code, indices in area_to_indices.items()
    }


def _extract_entity_subset(
    df: pd.DataFrame,
    hh_indices: np.ndarray,
    id_col: str,
    fk_col: Optional[str],
    hh_ids: np.ndarray,
) -> pd.DataFrame:
    """Extract rows from an entity table for a subset of households.

    Args:
        df: Full entity DataFrame (person or benunit).
        hh_indices: Household row indices in the area.
        id_col: Primary key column (unused but kept for clarity).
        fk_col: Foreign key column linking to household_id.
        hh_ids: Household IDs corresponding to hh_indices.

    Returns:
        Filtered DataFrame with only rows belonging to those households.
    """
    if fk_col is None:
        return df.iloc[hh_indices].reset_index(drop=True)

    hh_id_set = set(hh_ids)
    mask = df[fk_col].isin(hh_id_set)
    return df.loc[mask].reset_index(drop=True)


def publish_area_h5(
    dataset,
    weights: np.ndarray,
    area_code: str,
    hh_indices: np.ndarray,
    output_path: Path,
) -> dict:
    """Write a single per-area H5 file.

    The H5 contains the same structure as the national dataset but
    filtered to households in this area, with calibrated weights.

    Args:
        dataset: Full cloned dataset.
        weights: Sparse weight vector (length = total cloned households).
        area_code: GSS area code.
        hh_indices: Row indices of households in this area.
        output_path: Where to write the H5 file.

    Returns:
        Dict with area stats: code, n_households, n_active, total_weight.
    """
    if len(hh_indices) == 0:
        logger.warning("Area %s has no assigned households, skipping", area_code)
        return {
            "code": area_code,
            "n_households": 0,
            "n_active": 0,
            "total_weight": 0.0,
        }

    area_weights = weights[hh_indices]

    # Filter to active households (non-zero weight after L0 pruning)
    active_mask = area_weights > 0
    active_indices = hh_indices[active_mask]
    active_weights = area_weights[active_mask]

    if len(active_indices) == 0:
        logger.warning(
            "Area %s: all %d households pruned to zero weight",
            area_code,
            len(hh_indices),
        )
        return {
            "code": area_code,
            "n_households": len(hh_indices),
            "n_active": 0,
            "total_weight": 0.0,
        }

    # Extract household subset
    hh_subset = dataset.household.iloc[active_indices].copy()
    hh_subset = hh_subset.reset_index(drop=True)
    hh_subset["household_weight"] = active_weights

    # Extract person and benunit subsets via FK join
    hh_ids = set(hh_subset["household_id"].values)

    person_mask = dataset.person["person_household_id"].isin(hh_ids)
    person_subset = dataset.person.loc[person_mask].reset_index(drop=True)

    # Benunit FK: benunit_id // 100 == household_id
    benunit_hh_ids = dataset.benunit["benunit_id"].values // 100
    benunit_mask = pd.Series(benunit_hh_ids).isin(hh_ids)
    benunit_subset = dataset.benunit.loc[benunit_mask.values].reset_index(drop=True)

    # Write H5
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as f:
        # Store each entity table as a group of datasets (one per column)
        for table_name, df in [
            ("household", hh_subset),
            ("person", person_subset),
            ("benunit", benunit_subset),
        ]:
            grp = f.create_group(table_name)
            for col in df.columns:
                values = df[col].values
                if values.dtype == object:
                    values = values.astype(str)
                    grp.create_dataset(col, data=values.astype("S"))
                else:
                    grp.create_dataset(col, data=values)

        # Metadata
        f.attrs["area_code"] = area_code
        f.attrs["n_households"] = len(hh_subset)
        f.attrs["n_persons"] = len(person_subset)
        f.attrs["n_benunits"] = len(benunit_subset)
        f.attrs["total_weight"] = float(active_weights.sum())

    return {
        "code": area_code,
        "n_households": len(hh_indices),
        "n_active": len(active_indices),
        "total_weight": float(active_weights.sum()),
    }


def publish_local_h5s(
    dataset,
    weight_file: str,
    area_type: str = "constituency",
    dataset_key: str = "2025",
    output_dir: Optional[Path] = None,
    min_weight: float = 0.0,
) -> pd.DataFrame:
    """Generate per-area H5 files from L0-calibrated weights.

    Reads the sparse weight vector from the L0 output, maps each
    household to its area via clone-and-assign geography columns,
    and writes one H5 per area containing only active households.

    Args:
        dataset: Cloned dataset (post clone-and-assign).
        weight_file: HDF5 file with sparse weight vector (from calibrate_l0).
        area_type: "constituency" or "la".
        dataset_key: Key within the weight HDF5 file.
        output_dir: Override output directory.
        min_weight: Minimum weight threshold (below treated as zero).

    Returns:
        DataFrame with per-area statistics: code, n_households,
        n_active, total_weight.
    """
    if output_dir is None:
        output_dir = LOCAL_H5_DIR / area_type

    # Load sparse weights
    weight_path = STORAGE_FOLDER / weight_file
    with h5py.File(weight_path, "r") as f:
        weights = f[dataset_key][:]

    if min_weight > 0:
        weights = np.where(weights > min_weight, weights, 0.0)

    # Load area codes
    if area_type == "constituency":
        area_codes_df = pd.read_csv(STORAGE_FOLDER / "constituencies_2024.csv")
    elif area_type == "la":
        area_codes_df = pd.read_csv(STORAGE_FOLDER / "local_authorities_2021.csv")
    else:
        raise ValueError(f"Unknown area_type: {area_type}")

    area_codes = area_codes_df["code"].tolist()

    # Build area → household index mapping
    area_indices = _get_area_household_indices(dataset, area_type, area_codes)

    # Generate H5 files
    stats = []
    n_areas = len(area_codes)

    for i, code in enumerate(area_codes):
        hh_indices = area_indices[code]
        h5_path = output_dir / f"{code}.h5"

        stat = publish_area_h5(
            dataset=dataset,
            weights=weights,
            area_code=code,
            hh_indices=hh_indices,
            output_path=h5_path,
        )
        stats.append(stat)

        if (i + 1) % 50 == 0 or (i + 1) == n_areas:
            logger.info(
                "Published %d/%d %s H5 files",
                i + 1,
                n_areas,
                area_type,
            )

    stats_df = pd.DataFrame(stats)

    # Write summary
    summary_path = output_dir / "_summary.csv"
    stats_df.to_csv(summary_path, index=False)

    total_active = stats_df["n_active"].sum()
    total_hh = stats_df["n_households"].sum()
    empty_areas = (stats_df["n_active"] == 0).sum()

    logger.info(
        "Published %d %s H5 files: %d/%d active households, %d empty areas",
        n_areas,
        area_type,
        total_active,
        total_hh,
        empty_areas,
    )

    return stats_df


def validate_local_h5s(
    area_type: str = "constituency",
    output_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Validate published per-area H5 files.

    Checks:
    - All expected area files exist
    - Each H5 has the correct structure (household/person/benunit groups)
    - Weight totals are positive
    - No duplicate household IDs across areas

    Args:
        area_type: "constituency" or "la".
        output_dir: Override output directory.

    Returns:
        DataFrame with validation results per area.
    """
    if output_dir is None:
        output_dir = LOCAL_H5_DIR / area_type

    if area_type == "constituency":
        area_codes_df = pd.read_csv(STORAGE_FOLDER / "constituencies_2024.csv")
    elif area_type == "la":
        area_codes_df = pd.read_csv(STORAGE_FOLDER / "local_authorities_2021.csv")
    else:
        raise ValueError(f"Unknown area_type: {area_type}")

    area_codes = area_codes_df["code"].tolist()
    results = []
    all_hh_ids = set()
    duplicates = 0

    for code in area_codes:
        h5_path = output_dir / f"{code}.h5"
        result = {"code": code, "exists": h5_path.exists()}

        if not h5_path.exists():
            result.update(
                {
                    "valid_structure": False,
                    "n_households": 0,
                    "total_weight": 0.0,
                    "has_duplicates": False,
                }
            )
            results.append(result)
            continue

        with h5py.File(h5_path, "r") as f:
            has_structure = all(g in f for g in ("household", "person", "benunit"))
            result["valid_structure"] = has_structure

            if has_structure:
                n_hh = f.attrs.get("n_households", 0)
                total_w = f.attrs.get("total_weight", 0.0)
                result["n_households"] = n_hh
                result["total_weight"] = total_w

                # Check for duplicate household IDs
                if "household_id" in f["household"]:
                    hh_ids = set(f["household"]["household_id"][:])
                    overlap = hh_ids & all_hh_ids
                    duplicates += len(overlap)
                    result["has_duplicates"] = len(overlap) > 0
                    all_hh_ids.update(hh_ids)
                else:
                    result["has_duplicates"] = False
            else:
                result.update(
                    {
                        "n_households": 0,
                        "total_weight": 0.0,
                        "has_duplicates": False,
                    }
                )

        results.append(result)

    results_df = pd.DataFrame(results)

    missing = (~results_df["exists"]).sum()
    invalid = (~results_df["valid_structure"]).sum()

    logger.info(
        "Validation: %d areas, %d missing, %d invalid structure, %d duplicate HH IDs",
        len(area_codes),
        missing,
        invalid,
        duplicates,
    )

    return results_df
