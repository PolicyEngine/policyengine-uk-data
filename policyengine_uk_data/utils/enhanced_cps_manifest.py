"""Artifact manifest generation for the public UK enhanced CPS transfer data."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from policyengine_uk.data import UKSingleYearDataset

from policyengine_uk_data.datasets.enhanced_cps import (
    ENHANCED_CPS_FILE,
    ENHANCED_CPS_MANIFEST_FILE,
    ENHANCED_CPS_SOURCE_FILE,
    USD_TO_GBP,
    USD_TO_GBP_SOURCE_URL,
    create_enhanced_cps,
)
from policyengine_uk_data.utils.loss import get_loss_results

ENHANCED_CPS_MANIFEST_SCHEMA_VERSION = 1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repo_root() -> Path | None:
    try:
        root = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            text=True,
        ).strip()
    except Exception:
        return None
    return Path(root)


def _git_value(*args: str) -> str | None:
    try:
        return subprocess.check_output(["git", *args], text=True).strip()
    except Exception:
        return None


def _relative_to_repo(path: Path) -> str | None:
    root = _repo_root()
    if root is None:
        return None
    try:
        return str(path.resolve().relative_to(root))
    except ValueError:
        return None


def _git_blob_sha(path: Path) -> str | None:
    relative_path = _relative_to_repo(path)
    if relative_path is None:
        return None
    return _git_value("rev-parse", f"HEAD:{relative_path}")


def _last_commit(path: Path) -> str | None:
    relative_path = _relative_to_repo(path)
    if relative_path is None:
        return None
    return _git_value("log", "-1", "--format=%H", "--", relative_path)


def _loss_summary(dataset: UKSingleYearDataset, fiscal_year: int) -> dict[str, Any]:
    loss = get_loss_results(dataset, str(fiscal_year))
    abs_relative_error = loss.abs_rel_error.to_numpy()
    include = np.isfinite(abs_relative_error) & (loss.target.to_numpy() != 0)
    included_errors = loss.loc[include, "abs_rel_error"]
    return {
        "target_count": int(len(loss)),
        "nonzero_finite_target_count": int(include.sum()),
        "zero_target_count": int((loss.target == 0).sum()),
        "nonfinite_relative_error_count": int((~np.isfinite(abs_relative_error)).sum()),
        "mean_abs_relative_error": round(float(included_errors.mean()), 6),
        "median_abs_relative_error": round(float(included_errors.median()), 6),
        "p90_abs_relative_error": round(
            float(included_errors.quantile(0.9)),
            6,
        ),
        "share_within_10pct": round(float((included_errors <= 0.10).mean()), 6),
        "share_within_25pct": round(float((included_errors <= 0.25).mean()), 6),
        "excludes_zero_target_relative_errors": True,
    }


def _weight_diagnostics(weights: np.ndarray) -> dict[str, Any]:
    sorted_weights = np.sort(weights)[::-1]
    total = weights.sum()
    threshold = 1e-6
    return {
        "total_household_weight": round(float(total), 6),
        "effective_sample_size": round(
            float(total**2 / np.square(weights).sum()),
            6,
        ),
        "min_household_weight": float(weights.min()),
        "max_household_weight": round(float(weights.max()), 6),
        "top_1_share": round(float(sorted_weights[:1].sum() / total), 9),
        "top_10_share": round(float(sorted_weights[:10].sum() / total), 9),
        "top_100_share": round(float(sorted_weights[:100].sum() / total), 9),
        "near_zero_weight_threshold": threshold,
        "near_zero_weight_count": int((weights <= threshold).sum()),
    }


def build_enhanced_cps_manifest(
    *,
    source_file_path: str | Path = ENHANCED_CPS_SOURCE_FILE,
    artifact_file_path: str | Path = ENHANCED_CPS_FILE,
    fiscal_year: int = 2025,
    include_loss: bool = True,
    include_raw_loss: bool = True,
) -> dict[str, Any]:
    """Build a JSON-serializable manifest for the committed public artifact."""
    source_file_path = Path(source_file_path)
    artifact_file_path = Path(artifact_file_path)

    source = pd.read_csv(source_file_path, usecols=["scenario_id"])
    dataset = UKSingleYearDataset(file_path=str(artifact_file_path))
    values = dataset.load()
    household_weights = np.asarray(values["household_weight"], dtype=float)

    manifest: dict[str, Any] = {
        "schema_version": ENHANCED_CPS_MANIFEST_SCHEMA_VERSION,
        "artifact": "enhanced_cps_2025",
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "description": (
            "Public UK calibrated transfer dataset derived from a public export "
            "of benchmark-compatible PolicyEngine US Enhanced CPS households."
        ),
        "intended_uses": [
            "public demos",
            "reproducible examples",
            "public benchmark analysis",
        ],
        "not_intended_uses": [
            "substitution for FRS or enhanced FRS",
            "precise UK distributional analysis",
            "administrative truth validation",
        ],
        "files": {
            "artifact": {
                "path": _relative_to_repo(artifact_file_path)
                or str(artifact_file_path),
                "sha256": _sha256(artifact_file_path),
                "git_blob_sha": _git_blob_sha(artifact_file_path),
                "last_modified_commit": _last_commit(artifact_file_path),
                "size_bytes": artifact_file_path.stat().st_size,
            },
            "source_manifest": {
                "path": _relative_to_repo(source_file_path) or str(source_file_path),
                "sha256": _sha256(source_file_path),
                "git_blob_sha": _git_blob_sha(source_file_path),
                "last_modified_commit": _last_commit(source_file_path),
                "size_bytes": source_file_path.stat().st_size,
            },
        },
        "build": {
            "builder": "policyengine_uk_data.datasets.save_enhanced_cps",
            "build_command": (
                "uv run --python 3.13 python -m "
                "policyengine_uk_data.storage.write_enhanced_cps_manifest"
            ),
            "fiscal_year": fiscal_year,
            "source_dataset": "PolicyEngine US Enhanced CPS public export",
            "source_scope": "benchmark-compatible households",
            "calibrated": True,
            "calibration_target_year": fiscal_year,
            "exchange_rate": {
                "usd_to_gbp": USD_TO_GBP,
                "source_url": USD_TO_GBP_SOURCE_URL,
                "live_api_called": False,
            },
        },
        "row_counts": {
            "source_households": int(len(source)),
            "h5_households": int(len(values["household_id"])),
            "h5_people": int(len(values["person_id"])),
            "h5_benunits": int(len(values["benunit_id"])),
        },
        "weight_diagnostics": _weight_diagnostics(household_weights),
    }

    if include_loss:
        manifest["loss_diagnostics"] = {
            "target_year": fiscal_year,
            "calibrated": _loss_summary(dataset, fiscal_year),
        }
        if include_raw_loss:
            raw_dataset = create_enhanced_cps(
                source_file_path=source_file_path,
                fiscal_year=fiscal_year,
                calibrate=False,
            )
            manifest["loss_diagnostics"]["raw_transfer_weights"] = _loss_summary(
                raw_dataset,
                fiscal_year,
            )

    return manifest


def write_enhanced_cps_manifest(
    output_file_path: str | Path = ENHANCED_CPS_MANIFEST_FILE,
    **kwargs,
) -> dict[str, Any]:
    """Write the public enhanced CPS artifact manifest to disk."""
    manifest = build_enhanced_cps_manifest(**kwargs)
    output_file_path = Path(output_file_path)
    output_file_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(ENHANCED_CPS_MANIFEST_FILE))
    parser.add_argument("--skip-loss", action="store_true")
    parser.add_argument("--skip-raw-loss", action="store_true")
    args = parser.parse_args()

    write_enhanced_cps_manifest(
        output_file_path=args.output,
        include_loss=not args.skip_loss,
        include_raw_loss=not args.skip_raw_loss,
    )


if __name__ == "__main__":
    main()
