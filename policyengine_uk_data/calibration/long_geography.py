"""Long-format local geography weights for cloned UK datasets.

The legacy local-area exports store weights as dense or sparse-shaped H5
arrays keyed by area row and household column. Routine builds write assigned
OA-derived rows; small diagnostics can also expand a legacy dense matrix into
one row per active area-household cell.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from policyengine_uk_data.storage import STORAGE_FOLDER
from policyengine_uk_data.utils.calibrate import default_weight_dataset_key

AREA_TYPES = ("oa", "constituency", "la")
AREA_CODE_FILES = {
    "oa": "oa_crosswalk.csv.gz",
    "constituency": "constituencies_2024.csv",
    "la": "local_authorities_2021.csv",
}
GEO_COLUMNS = {
    "oa": "oa_code",
    "constituency": "constituency_code_oa",
    "la": "la_code_oa",
}
LONG_GEOGRAPHY_WEIGHTS_FILE = "local_geography_weights.csv.gz"
LONG_GEOGRAPHY_COLUMNS = [
    "area_type",
    "area_code",
    "area_index",
    "household_index",
    "household_id",
    "source_year",
    "source_household_id",
    "source_household_key",
    "clone_index",
    "weight",
    "weight_source",
]
AREA_SUPPORT_COLUMNS = [
    "area_type",
    "area_code",
    "area_index",
    "n_rows",
    "n_source_households",
    "total_weight",
    "effective_sample_size",
    "weight_source",
]
AREA_SUPPORT_SUMMARY_COLUMNS = [
    "area_type",
    "n_areas",
    "n_nonempty_areas",
    "share_nonempty_areas",
    "total_rows",
    "median_rows",
    "p10_rows",
    "median_source_households",
    "p10_source_households",
    "median_effective_sample_size",
    "p10_effective_sample_size",
    "low_support_areas",
    "total_weight",
]


def geo_column(area_type: str) -> str:
    """Return the cloned-household geography column for a local area type."""
    try:
        return GEO_COLUMNS[area_type]
    except KeyError as exc:
        raise ValueError(f"Unknown area_type: {area_type}") from exc


def _as_area_type_tuple(area_types: str | Sequence[str]) -> tuple[str, ...]:
    if isinstance(area_types, str):
        area_types = (area_types,)
    resolved = tuple(area_types)
    for area_type in resolved:
        geo_column(area_type)
    return resolved


def _normalize_area_code(value) -> str:
    if isinstance(value, bytes):
        value = value.decode()
    if pd.isna(value):
        return ""
    return str(value).strip()


def _normalize_area_codes(values) -> pd.Series:
    return pd.Series([_normalize_area_code(value) for value in values])


def load_area_codes(
    area_type: str,
    storage_folder: Path | str | None = None,
) -> pd.DataFrame:
    """Load canonical area codes for a supported local geography."""
    area_code_file = AREA_CODE_FILES.get(area_type)
    if area_code_file is None:
        raise ValueError(f"Unknown area_type: {area_type}")

    if storage_folder is None:
        storage_folder = STORAGE_FOLDER

    path = Path(storage_folder) / area_code_file
    area_codes = pd.read_csv(path)
    if area_type == "oa" and "oa_code" in area_codes.columns:
        area_codes = area_codes.rename(columns={"oa_code": "code"})
    if "code" not in area_codes.columns:
        raise ValueError(f"{path} must contain a 'code' or 'oa_code' column")
    area_codes = area_codes.copy()
    area_codes["code"] = _normalize_area_codes(area_codes["code"])
    area_codes = area_codes.drop_duplicates(subset=["code"])
    return area_codes


def _household_ids(household: pd.DataFrame) -> np.ndarray:
    if "household_id" in household.columns:
        return household["household_id"].to_numpy()
    return np.arange(len(household), dtype=np.int64)


def _source_household_ids(household: pd.DataFrame) -> np.ndarray:
    if "source_household_id" in household.columns:
        return household["source_household_id"].to_numpy()
    return _household_ids(household)


def _source_years(household: pd.DataFrame) -> np.ndarray:
    if "source_year" in household.columns:
        return household["source_year"].to_numpy()
    return np.array([""] * len(household), dtype=object)


def _source_household_keys(
    source_years: np.ndarray,
    source_household_ids: np.ndarray,
) -> np.ndarray:
    return np.array(
        [
            f"{year}:{household_id}" if str(year) else str(household_id)
            for year, household_id in zip(source_years, source_household_ids)
        ],
        dtype=object,
    )


def _clone_indices(household: pd.DataFrame) -> np.ndarray:
    if "clone_index" in household.columns:
        return household["clone_index"].to_numpy(dtype=np.int64)
    return np.zeros(len(household), dtype=np.int64)


def _default_weights(household: pd.DataFrame) -> np.ndarray:
    if "household_weight" in household.columns:
        return household["household_weight"].to_numpy(dtype=np.float64)
    return np.ones(len(household), dtype=np.float64)


def _validate_weight_array(
    weights: np.ndarray,
    *,
    n_households: int,
    n_areas: int,
    area_type: str,
) -> np.ndarray:
    weights = np.asarray(weights, dtype=np.float64)
    if weights.ndim == 1:
        if len(weights) != n_households:
            raise ValueError(
                f"1D {area_type} weights have {len(weights)} records, "
                f"expected {n_households}."
            )
    elif weights.ndim == 2:
        if weights.shape != (n_areas, n_households):
            raise ValueError(
                f"2D {area_type} weights have shape {weights.shape}, expected "
                f"({n_areas}, {n_households})."
            )
    else:
        raise ValueError(
            f"{area_type} weights must be 1D or 2D; got shape {weights.shape}."
        )
    return weights


def build_long_geography_frame(
    dataset,
    area_types: str | Sequence[str] = AREA_TYPES,
    area_codes: Mapping[str, pd.DataFrame | pd.Series | Sequence[str]]
    | pd.DataFrame
    | pd.Series
    | Sequence[str]
    | None = None,
    weights: np.ndarray | Sequence[float] | None = None,
    weight_source: str | None = None,
    min_weight: float = 0.0,
    drop_zero_weights: bool = False,
) -> pd.DataFrame:
    """Build a long local-geography assignment table.

    Args:
        dataset: Cloned dataset with household geography columns.
        area_types: ``"oa"``, ``"constituency"``, ``"la"``, or any subset.
        area_codes: Canonical area codes defining row order. For multiple area
            types, pass a mapping from area type to codes. If omitted, codes
            are loaded from storage.
        weights: Optional calibrated weights. A 1D vector is interpreted as
            one household weight per record and is joined to each household's
            assigned area. A 2D array is interpreted as the legacy
            ``(n_areas, n_households)`` area-by-household layout and emits
            one row per matrix cell; this is only valid when building one area
            type at a time.
        weight_source: Label describing where ``weight`` came from.
        min_weight: Threshold used when dropping zero/small weights.
        drop_zero_weights: If true, keep only rows with ``weight > min_weight``.

    Returns:
        DataFrame with columns ``area_type``, ``area_code``, ``area_index``,
        ``household_index``, source identity columns, ``weight`` and
        ``weight_source``.
    """
    resolved_area_types = _as_area_type_tuple(area_types)
    household = dataset.household
    n_households = len(household)
    household_indices = np.arange(n_households, dtype=np.int64)
    household_ids = _household_ids(household)
    source_years = _source_years(household)
    source_household_ids = _source_household_ids(household)
    source_household_keys = _source_household_keys(
        source_years,
        source_household_ids,
    )
    clone_indices = _clone_indices(household)

    weight_array = None
    if weights is not None:
        weight_array = np.asarray(weights, dtype=np.float64)
        if weight_array.ndim == 2 and len(resolved_area_types) != 1:
            raise ValueError("2D weights can only be converted for one area_type.")

    frames = []
    for area_type in resolved_area_types:
        column = geo_column(area_type)
        if column not in household.columns:
            raise ValueError(f"dataset.household must contain {column!r}.")

        if area_codes is None:
            area_codes_df = load_area_codes(area_type)
        elif isinstance(area_codes, Mapping):
            area_codes_df = area_codes[area_type]
        else:
            if len(resolved_area_types) != 1:
                raise ValueError(
                    "area_codes must be a mapping when building multiple area_types."
                )
            area_codes_df = area_codes

        if isinstance(area_codes_df, pd.DataFrame):
            canonical_codes = _normalize_area_codes(area_codes_df["code"])
        else:
            canonical_codes = _normalize_area_codes(area_codes_df)

        code_to_index = {code: i for i, code in enumerate(canonical_codes)}
        normalized_codes = _normalize_area_codes(household[column])
        area_indices = normalized_codes.map(code_to_index).to_numpy()
        assigned = ~pd.isna(area_indices)

        if weight_array is None:
            row_weights = _default_weights(household)
            resolved_weight_source = weight_source or "household_weight"
        else:
            weights_for_area = _validate_weight_array(
                weight_array,
                n_households=n_households,
                n_areas=len(canonical_codes),
                area_type=area_type,
            )
            resolved_weight_source = weight_source or "calibrated_weight"
            if weights_for_area.ndim == 2:
                if drop_zero_weights:
                    row_area_indices, row_household_indices = np.where(
                        weights_for_area > min_weight
                    )
                else:
                    row_area_indices, row_household_indices = np.indices(
                        weights_for_area.shape
                    )
                    row_area_indices = row_area_indices.ravel()
                    row_household_indices = row_household_indices.ravel()

                canonical_array = canonical_codes.to_numpy(dtype=object)
                frame = pd.DataFrame(
                    {
                        "area_type": area_type,
                        "area_code": canonical_array[row_area_indices],
                        "area_index": row_area_indices.astype(np.int64),
                        "household_index": row_household_indices.astype(np.int64),
                        "household_id": household_ids[row_household_indices],
                        "source_year": source_years[row_household_indices],
                        "source_household_id": source_household_ids[
                            row_household_indices
                        ],
                        "source_household_key": source_household_keys[
                            row_household_indices
                        ],
                        "clone_index": clone_indices[row_household_indices],
                        "weight": weights_for_area[
                            row_area_indices,
                            row_household_indices,
                        ],
                        "weight_source": resolved_weight_source,
                    }
                )
                frames.append(frame[LONG_GEOGRAPHY_COLUMNS])
                continue
            row_weights = weights_for_area

        frame = pd.DataFrame(
            {
                "area_type": area_type,
                "area_code": normalized_codes,
                "area_index": area_indices,
                "household_index": household_indices,
                "household_id": household_ids,
                "source_year": source_years,
                "source_household_id": source_household_ids,
                "source_household_key": source_household_keys,
                "clone_index": clone_indices,
                "weight": row_weights,
                "weight_source": resolved_weight_source,
            }
        )
        frame = frame.loc[assigned].copy()
        frame["area_index"] = frame["area_index"].astype(np.int64)
        if drop_zero_weights:
            frame = frame.loc[frame["weight"] > min_weight].copy()
        frames.append(frame[LONG_GEOGRAPHY_COLUMNS])

    if not frames:
        return pd.DataFrame(columns=LONG_GEOGRAPHY_COLUMNS)
    return pd.concat(frames, ignore_index=True)


def area_support_from_long_geography(
    frame: pd.DataFrame,
    area_codes: Mapping[str, pd.DataFrame | pd.Series | Sequence[str]] | None = None,
) -> pd.DataFrame:
    """Summarize row and weight support by geography area."""
    if frame.empty:
        if area_codes is None:
            return pd.DataFrame(columns=AREA_SUPPORT_COLUMNS)
        expected_frames = []
        for area_type, codes in area_codes.items():
            if isinstance(codes, pd.DataFrame):
                code_values = (
                    codes["code"] if "code" in codes.columns else codes["oa_code"]
                )
            else:
                code_values = codes
            normalized = _normalize_area_codes(code_values)
            expected_frames.append(
                pd.DataFrame(
                    {
                        "area_type": area_type,
                        "area_code": normalized,
                        "area_index": np.arange(len(normalized), dtype=np.int64),
                        "n_rows": 0,
                        "n_source_households": 0,
                        "total_weight": 0.0,
                        "effective_sample_size": 0.0,
                        "weight_source": "",
                    }
                )
            )
        return pd.concat(expected_frames, ignore_index=True)[AREA_SUPPORT_COLUMNS]

    working = frame.copy()
    working["weight_squared"] = working["weight"].astype(float) ** 2
    grouped = (
        working.groupby(["area_type", "area_code"], sort=False)
        .agg(
            area_index=("area_index", "first"),
            n_rows=("household_id", "size"),
            n_source_households=("source_household_key", "nunique"),
            total_weight=("weight", "sum"),
            weight_squared=("weight_squared", "sum"),
            weight_source=(
                "weight_source",
                lambda values: ",".join(sorted({str(value) for value in values})),
            ),
        )
        .reset_index()
    )
    grouped["effective_sample_size"] = np.where(
        grouped["weight_squared"] > 0,
        grouped["total_weight"] ** 2 / grouped["weight_squared"],
        0.0,
    )
    grouped = grouped.drop(columns=["weight_squared"])

    if area_codes is None:
        return grouped[AREA_SUPPORT_COLUMNS]

    expected_frames = []
    for area_type, codes in area_codes.items():
        if isinstance(codes, pd.DataFrame):
            code_values = codes["code"] if "code" in codes.columns else codes["oa_code"]
        else:
            code_values = codes
        normalized = _normalize_area_codes(code_values)
        expected_frames.append(
            pd.DataFrame(
                {
                    "area_type": area_type,
                    "area_code": normalized,
                    "area_index": np.arange(len(normalized), dtype=np.int64),
                }
            )
        )
    expected = pd.concat(expected_frames, ignore_index=True)
    support = expected.merge(
        grouped,
        on=["area_type", "area_code"],
        how="left",
        suffixes=("", "_observed"),
    )
    support["area_index"] = support["area_index_observed"].fillna(support["area_index"])
    support = support.drop(columns=["area_index_observed"])
    for column in (
        "n_rows",
        "n_source_households",
        "total_weight",
        "effective_sample_size",
    ):
        support[column] = support[column].fillna(0)
    support["weight_source"] = support["weight_source"].fillna("")
    support["area_index"] = support["area_index"].astype(np.int64)
    support["n_rows"] = support["n_rows"].astype(np.int64)
    support["n_source_households"] = support["n_source_households"].astype(np.int64)
    return support[AREA_SUPPORT_COLUMNS]


def summarize_area_support(
    area_support: pd.DataFrame,
    min_source_households: int = 10,
    min_effective_sample_size: float = 10.0,
) -> pd.DataFrame:
    """Collapse per-area support into clone-count credibility diagnostics."""
    rows = []
    for area_type, group in area_support.groupby("area_type", sort=False):
        nonempty = group[group["n_rows"] > 0]
        low_support = group[
            (group["n_source_households"] < min_source_households)
            | (group["effective_sample_size"] < min_effective_sample_size)
        ]
        rows.append(
            {
                "area_type": area_type,
                "n_areas": int(len(group)),
                "n_nonempty_areas": int(len(nonempty)),
                "share_nonempty_areas": (
                    float(len(nonempty) / len(group)) if len(group) else 0.0
                ),
                "total_rows": int(group["n_rows"].sum()),
                "median_rows": float(group["n_rows"].median()) if len(group) else 0.0,
                "p10_rows": float(group["n_rows"].quantile(0.1)) if len(group) else 0.0,
                "median_source_households": float(group["n_source_households"].median())
                if len(group)
                else 0.0,
                "p10_source_households": float(
                    group["n_source_households"].quantile(0.1)
                )
                if len(group)
                else 0.0,
                "median_effective_sample_size": float(
                    group["effective_sample_size"].median()
                )
                if len(group)
                else 0.0,
                "p10_effective_sample_size": float(
                    group["effective_sample_size"].quantile(0.1)
                )
                if len(group)
                else 0.0,
                "low_support_areas": int(len(low_support)),
                "total_weight": float(group["total_weight"].sum()),
            }
        )
    return pd.DataFrame(rows, columns=AREA_SUPPORT_SUMMARY_COLUMNS)


def geography_support_report(
    dataset,
    area_types: str | Sequence[str] = AREA_TYPES,
    area_codes: Mapping[str, pd.DataFrame | pd.Series | Sequence[str]]
    | pd.DataFrame
    | pd.Series
    | Sequence[str]
    | None = None,
    weights: np.ndarray | Sequence[float] | None = None,
    min_source_households: int = 10,
    min_effective_sample_size: float = 10.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return per-area support and summary diagnostics for a dataset."""
    resolved_area_types = _as_area_type_tuple(area_types)
    if area_codes is None:
        expected_codes = {
            area_type: load_area_codes(area_type) for area_type in resolved_area_types
        }
        frame_area_codes = expected_codes
    elif isinstance(area_codes, Mapping):
        expected_codes = area_codes
        frame_area_codes = area_codes
    else:
        expected_codes = None
        frame_area_codes = area_codes

    frame = build_long_geography_frame(
        dataset=dataset,
        area_types=resolved_area_types,
        area_codes=frame_area_codes,
        weights=weights,
        drop_zero_weights=True,
    )
    area_support = area_support_from_long_geography(
        frame,
        area_codes=expected_codes,
    )
    summary = summarize_area_support(
        area_support,
        min_source_households=min_source_households,
        min_effective_sample_size=min_effective_sample_size,
    )
    return area_support, summary


def clone_count_support_sweep(
    dataset,
    clone_counts: Sequence[int] = (1, 2, 5, 10),
    seed: int = 42,
    crosswalk_path: str | None = None,
    area_types: Sequence[str] = ("constituency", "la"),
    area_codes: Mapping[str, pd.DataFrame | pd.Series | Sequence[str]] | None = None,
    min_source_households: int = 10,
    min_effective_sample_size: float = 10.0,
) -> pd.DataFrame:
    """Evaluate local geography support as clone counts scale up."""
    from policyengine_uk_data.calibration.clone_and_assign import clone_and_assign

    summaries = []
    for n_clones in clone_counts:
        cloned = clone_and_assign(
            dataset,
            n_clones=n_clones,
            seed=seed,
            crosswalk_path=crosswalk_path,
        )
        _, summary = geography_support_report(
            cloned,
            area_types=area_types,
            area_codes=area_codes,
            min_source_households=min_source_households,
            min_effective_sample_size=min_effective_sample_size,
        )
        summary.insert(0, "n_clones", n_clones)
        summaries.append(summary)

    if not summaries:
        return pd.DataFrame(columns=["n_clones", *AREA_SUPPORT_SUMMARY_COLUMNS])
    return pd.concat(summaries, ignore_index=True)


def build_area_household_indices(
    dataset,
    area_type: str,
    area_codes: Sequence[str],
) -> dict[str, np.ndarray]:
    """Map each area code to assigned household row indices."""
    assignments = build_long_geography_frame(
        dataset=dataset,
        area_types=area_type,
        area_codes=area_codes,
    )
    groups = assignments.groupby("area_code", sort=False)["household_index"]
    grouped_indices = {code: group.to_numpy(dtype=np.int64) for code, group in groups}
    return {
        _normalize_area_code(code): grouped_indices.get(
            _normalize_area_code(code),
            np.array([], dtype=np.int64),
        )
        for code in area_codes
    }


def _resolve_storage_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return STORAGE_FOLDER / path


def _load_weight_array(
    weight_file: str | Path,
    dataset_key: str | None = None,
    *,
    area_type: str = "area",
    max_dense_cells: int | None = 10_000_000,
) -> np.ndarray:
    if dataset_key is None:
        dataset_key = default_weight_dataset_key()

    weight_path = _resolve_storage_path(weight_file)
    with h5py.File(weight_path, "r") as f:
        if dataset_key not in f:
            available = ", ".join(sorted(f.keys()))
            raise KeyError(
                f"Dataset key {dataset_key!r} not found in {weight_path}; "
                f"available keys: {available}"
            )
        dataset = f[dataset_key]
        if (
            max_dense_cells is not None
            and len(dataset.shape) == 2
            and int(np.prod(dataset.shape)) > max_dense_cells
        ):
            raise ValueError(
                f"Refusing to expand {int(np.prod(dataset.shape)):,} dense "
                f"{area_type} area-household cells into a long CSV. Use assigned "
                "geography rows, a 1D L0 weight vector, or pass a higher "
                "max_dense_cells for an explicit diagnostic conversion."
            )
        return dataset[:]


def write_long_geography_weights(
    dataset,
    weight_files: Mapping[str, str | Path] | None = None,
    dataset_key: str | None = None,
    output_path: str | Path = LONG_GEOGRAPHY_WEIGHTS_FILE,
    area_types: Sequence[str] = AREA_TYPES,
    min_weight: float = 0.0,
    max_dense_cells: int | None = 10_000_000,
) -> pd.DataFrame:
    """Write calibrated local geography weights as a long CSV artifact."""
    weight_files = weight_files or {}
    frames = []
    for area_type in area_types:
        weight_file = weight_files.get(area_type)
        if weight_file is None:
            weights = None
            weight_source = "household_weight"
        else:
            weights = _load_weight_array(
                weight_file,
                dataset_key,
                area_type=area_type,
                max_dense_cells=max_dense_cells,
            )
            weight_source = Path(weight_file).name
        area_codes = load_area_codes(area_type)
        frames.append(
            build_long_geography_frame(
                dataset=dataset,
                area_types=area_type,
                area_codes=area_codes,
                weights=weights,
                weight_source=weight_source,
                min_weight=min_weight,
                drop_zero_weights=True,
            )
        )

    long_weights = pd.concat(frames, ignore_index=True)
    output_path = _resolve_storage_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    long_weights.to_csv(output_path, index=False, compression="infer")
    return long_weights


def load_long_geography_weights(
    path: str | Path = LONG_GEOGRAPHY_WEIGHTS_FILE,
) -> pd.DataFrame:
    """Load the long local geography weights artifact."""
    return pd.read_csv(_resolve_storage_path(path), dtype={"area_code": str})
