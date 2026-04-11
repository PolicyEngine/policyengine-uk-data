import hashlib
from io import BytesIO
from importlib import metadata
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from huggingface_hub import CommitOperationAdd

from policyengine_uk_data.utils.data_upload import (
    load_release_manifest_from_hf,
    upload_files_to_hf,
)
from policyengine_uk_data.utils.release_manifest import (
    RELEASE_MANIFEST_SCHEMA_VERSION,
    build_release_manifest,
)


def _write_file(path: Path, content: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


EXPECTED_COMPATIBLE_MODEL_PACKAGES = [
    {"name": "policyengine-uk", "version": "2.74.0"}
]


def test_build_release_manifest_tracks_uk_release_artifacts(tmp_path):
    enhanced_bytes = b"enhanced-frs"
    baseline_bytes = b"baseline-frs"
    weights_bytes = b"la-weights"

    enhanced_path = _write_file(tmp_path / "enhanced_frs_2023_24.h5", enhanced_bytes)
    baseline_path = _write_file(tmp_path / "frs_2023_24.h5", baseline_bytes)
    weights_path = _write_file(
        tmp_path / "local_authority_weights.h5",
        weights_bytes,
    )

    manifest = build_release_manifest(
        files_with_repo_paths=[
            (enhanced_path, "enhanced_frs_2023_24.h5"),
            (baseline_path, "frs_2023_24.h5"),
            (weights_path, "local_authority_weights.h5"),
        ],
        version="1.40.4",
        repo_id="policyengine/policyengine-uk-data-private",
        model_package_version="2.74.0",
        created_at="2026-04-10T12:00:00Z",
    )

    assert manifest["data_package"] == {
        "name": "policyengine-uk-data",
        "version": "1.40.4",
    }
    assert manifest["schema_version"] == RELEASE_MANIFEST_SCHEMA_VERSION
    assert manifest["compatible_model_packages"] == EXPECTED_COMPATIBLE_MODEL_PACKAGES
    assert manifest["default_datasets"] == {
        "national": "enhanced_frs_2023_24",
        "baseline": "frs_2023_24",
    }

    assert manifest["artifacts"]["enhanced_frs_2023_24"]["sha256"] == _sha256(
        enhanced_bytes
    )
    assert manifest["artifacts"]["frs_2023_24"]["sha256"] == _sha256(
        baseline_bytes
    )
    assert manifest["artifacts"]["local_authority_weights"]["kind"] == "weights"


def test_upload_files_to_hf_adds_uk_release_manifest_operations(tmp_path):
    dataset_path = _write_file(
        tmp_path / "enhanced_frs_2023_24.h5",
        b"enhanced-frs",
    )

    mock_api = MagicMock()
    mock_api.create_commit.return_value = MagicMock(oid="commit-sha")

    with (
        patch("policyengine_uk_data.utils.data_upload.HfApi", return_value=mock_api),
        patch(
            "policyengine_uk_data.utils.data_upload.load_release_manifest_from_hf",
            return_value=None,
        ),
        patch(
            "policyengine_uk_data.utils.data_upload.metadata.version",
            return_value="2.74.0",
        ),
        patch.dict(
            "policyengine_uk_data.utils.data_upload.os.environ",
            {"HUGGING_FACE_TOKEN": "token"},
            clear=False,
        ),
    ):
        upload_files_to_hf(
            files=[dataset_path],
            version="1.40.4",
        )

    operations = mock_api.create_commit.call_args.kwargs["operations"]
    operation_paths = [operation.path_in_repo for operation in operations]

    assert "enhanced_frs_2023_24.h5" in operation_paths
    assert "release_manifest.json" in operation_paths
    assert "releases/1.40.4/release_manifest.json" in operation_paths

    release_ops = [
        operation
        for operation in operations
        if operation.path_in_repo.endswith("release_manifest.json")
    ]
    assert len(release_ops) == 2
    for operation in release_ops:
        assert isinstance(operation, CommitOperationAdd)
        assert isinstance(operation.path_or_fileobj, BytesIO)


def test_build_release_manifest_preserves_existing_created_at(tmp_path):
    dataset_path = _write_file(
        tmp_path / "enhanced_frs_2023_24.h5",
        b"enhanced-frs",
    )

    manifest = build_release_manifest(
        files_with_repo_paths=[(dataset_path, "enhanced_frs_2023_24.h5")],
        version="1.40.4",
        repo_id="policyengine/policyengine-uk-data-private",
        model_package_version="2.74.0",
        existing_manifest={
            "schema_version": RELEASE_MANIFEST_SCHEMA_VERSION,
            "data_package": {
                "name": "policyengine-uk-data",
                "version": "1.40.4",
            },
            "compatible_model_packages": EXPECTED_COMPATIBLE_MODEL_PACKAGES,
            "default_datasets": {},
            "created_at": "2026-04-10T12:00:00Z",
            "artifacts": {},
        },
        created_at="2026-04-11T08:30:00Z",
    )

    assert manifest["created_at"] == "2026-04-10T12:00:00Z"


def test_load_release_manifest_from_hf_passes_revision(tmp_path):
    dataset_path = _write_file(
        tmp_path / "release_manifest.json",
        (
            '{"data_package": {"name": "policyengine-uk-data", "version": "1.40.4"}}'
        ).encode()
    )

    with patch(
        "policyengine_uk_data.utils.data_upload.hf_hub_download",
        return_value=str(dataset_path),
    ) as mock_download:
        manifest = load_release_manifest_from_hf(
            version="1.40.4",
            revision="1.40.4",
        )

    assert manifest["data_package"]["version"] == "1.40.4"
    assert mock_download.call_args.kwargs["revision"] == "1.40.4"


def test_upload_files_to_hf_requires_model_package_version(tmp_path):
    dataset_path = _write_file(
        tmp_path / "enhanced_frs_2023_24.h5",
        b"enhanced-frs",
    )

    with (
        patch(
            "policyengine_uk_data.utils.data_upload.load_release_manifest_from_hf",
            return_value=None,
        ),
        patch(
            "policyengine_uk_data.utils.data_upload.metadata.version",
            side_effect=metadata.PackageNotFoundError,
        ),
        patch.dict(
            "policyengine_uk_data.utils.data_upload.os.environ",
            {"HUGGING_FACE_TOKEN": "token"},
            clear=False,
        ),
    ):
        with pytest.raises(
            RuntimeError,
            match="Could not determine installed version for policyengine-uk",
        ):
            upload_files_to_hf(files=[dataset_path], version="1.40.4")


def test_upload_files_to_hf_rejects_finalized_release(tmp_path):
    dataset_path = _write_file(
        tmp_path / "enhanced_frs_2023_24.h5",
        b"enhanced-frs",
    )

    with (
        patch(
            "policyengine_uk_data.utils.data_upload.load_release_manifest_from_hf",
            side_effect=[{"data_package": {"version": "1.40.4"}}],
        ),
        patch.dict(
            "policyengine_uk_data.utils.data_upload.os.environ",
            {"HUGGING_FACE_TOKEN": "token"},
            clear=False,
        ),
    ):
        with pytest.raises(RuntimeError, match="already finalized"):
            upload_files_to_hf(files=[dataset_path], version="1.40.4")
