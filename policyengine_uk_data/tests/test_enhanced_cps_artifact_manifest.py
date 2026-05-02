import hashlib
import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from policyengine_uk.data import UKSingleYearDataset

from policyengine_uk_data.datasets import (
    ENHANCED_CPS_FILE,
    ENHANCED_CPS_MANIFEST_FILE,
    ENHANCED_CPS_SOURCE_FILE,
)
from policyengine_uk_data.datasets.policybench_transfer import (
    POLICYBENCH_TRANSFER_SOURCE_FILE,
)
from policyengine_uk_data.storage import STORAGE_FOLDER
from policyengine_uk_data.storage.upload_public_transfer_dataset import (
    upload_public_transfer_dataset,
)
from policyengine_uk_data.utils.hf_destinations import PUBLIC_REPO


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_blob_sha(path: Path) -> str:
    root = Path(
        subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            text=True,
        ).strip()
    )
    relative_path = path.resolve().relative_to(root)
    return subprocess.check_output(
        ["git", "rev-parse", f"HEAD:{relative_path}"],
        text=True,
    ).strip()


def _csv_rows(path: Path) -> int:
    return len(pd.read_csv(path, usecols=["scenario_id"]))


def _dataset_counts(path: Path) -> dict[str, int]:
    values = UKSingleYearDataset(file_path=str(path)).load()
    return {
        "h5_households": len(values["household_id"]),
        "h5_people": len(values["person_id"]),
        "h5_benunits": len(values["benunit_id"]),
    }


def _household_weights(path: Path) -> np.ndarray:
    values = UKSingleYearDataset(file_path=str(path)).load()
    return np.asarray(values["household_weight"], dtype=float)


def test_enhanced_cps_manifest_matches_committed_artifacts():
    manifest = json.loads(ENHANCED_CPS_MANIFEST_FILE.read_text())

    assert manifest["row_counts"]["source_households"] == _csv_rows(
        ENHANCED_CPS_SOURCE_FILE
    )
    assert (
        manifest["row_counts"] | _dataset_counts(ENHANCED_CPS_FILE)
        == manifest["row_counts"]
    )
    assert manifest["files"]["artifact"]["sha256"] == _sha256(ENHANCED_CPS_FILE)
    assert manifest["files"]["source_manifest"]["sha256"] == _sha256(
        ENHANCED_CPS_SOURCE_FILE
    )
    assert manifest["files"]["artifact"]["git_blob_sha"] == _git_blob_sha(
        ENHANCED_CPS_FILE
    )
    assert manifest["files"]["source_manifest"]["git_blob_sha"] == _git_blob_sha(
        ENHANCED_CPS_SOURCE_FILE
    )


def test_enhanced_cps_manifest_matches_docs_and_weight_diagnostics():
    manifest = json.loads(ENHANCED_CPS_MANIFEST_FILE.read_text())
    readme = Path("README.md").read_text()
    source_rows = manifest["row_counts"]["source_households"]

    assert f"{source_rows:,}" in readme

    weights = _household_weights(ENHANCED_CPS_FILE)
    sorted_weights = np.sort(weights)[::-1]
    total_weight = weights.sum()
    diagnostics = manifest["weight_diagnostics"]

    assert np.isclose(diagnostics["total_household_weight"], total_weight)
    assert np.isclose(
        diagnostics["effective_sample_size"],
        total_weight**2 / np.square(weights).sum(),
    )
    assert np.isclose(diagnostics["max_household_weight"], weights.max())
    assert np.isclose(
        diagnostics["top_10_share"],
        sorted_weights[:10].sum() / total_weight,
    )
    assert diagnostics["near_zero_weight_count"] == int(
        (weights <= diagnostics["near_zero_weight_threshold"]).sum()
    )


def test_legacy_policybench_transfer_artifacts_are_explicitly_legacy():
    legacy_source = STORAGE_FOLDER / "policybench_transfer_source_2025.csv"
    legacy_artifact = STORAGE_FOLDER / "policybench_transfer_2025.h5"

    assert _csv_rows(legacy_source) == 1_000
    assert _dataset_counts(legacy_artifact)["h5_households"] == 1_000
    assert POLICYBENCH_TRANSFER_SOURCE_FILE == ENHANCED_CPS_SOURCE_FILE


def test_public_transfer_upload_targets_public_hf_repo():
    with patch(
        "policyengine_uk_data.storage.upload_public_transfer_dataset.upload_files_to_hf",
        autospec=True,
    ) as upload_files_to_hf:
        upload_public_transfer_dataset(version="1.55.3")

    upload_files_to_hf.assert_called_once()
    kwargs = upload_files_to_hf.call_args.kwargs
    assert kwargs["version"] == "1.55.3"
    assert kwargs["hf_repo_name"] == PUBLIC_REPO
    assert kwargs["files"] == [
        ENHANCED_CPS_FILE,
        ENHANCED_CPS_SOURCE_FILE,
        ENHANCED_CPS_MANIFEST_FILE,
    ]
