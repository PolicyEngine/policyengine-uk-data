import hashlib
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, patch

from huggingface_hub import CommitOperationAdd

from policyengine_uk_data.utils.data_upload import upload_files_to_hf
from policyengine_uk_data.utils.release_manifest import (
    RELEASE_MANIFEST_SCHEMA_VERSION,
    build_release_manifest,
)
from policyengine_uk_data.utils.trace_tro import TRACE_TRO_FILENAME


def _write_file(path: Path, content: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


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
        model_package_git_sha="deadbeef",
        model_package_data_build_fingerprint="sha256:fingerprint",
        created_at="2026-04-10T12:00:00Z",
    )

    assert manifest["data_package"] == {
        "name": "policyengine-uk-data",
        "version": "1.40.4",
    }
    assert manifest["schema_version"] == RELEASE_MANIFEST_SCHEMA_VERSION
    assert manifest["compatible_model_packages"] == [
        {
            "name": "policyengine-uk",
            "specifier": "==2.74.0",
        }
    ]
    assert manifest["build"] == {
        "build_id": "policyengine-uk-data-1.40.4",
        "built_at": "2026-04-10T12:00:00Z",
        "built_with_model_package": {
            "name": "policyengine-uk",
            "version": "2.74.0",
            "git_sha": "deadbeef",
            "data_build_fingerprint": "sha256:fingerprint",
        },
    }
    assert manifest["default_datasets"] == {
        "national": "enhanced_frs_2023_24",
        "baseline": "frs_2023_24",
    }

    assert manifest["artifacts"]["enhanced_frs_2023_24"]["sha256"] == _sha256(
        enhanced_bytes
    )
    assert manifest["artifacts"]["frs_2023_24"]["sha256"] == _sha256(baseline_bytes)
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
            "policyengine_uk_data.utils.data_upload._get_model_package_build_metadata",
            return_value={
                "version": "2.74.0",
                "git_sha": "deadbeef",
                "data_build_fingerprint": "sha256:fingerprint",
            },
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
    assert TRACE_TRO_FILENAME in operation_paths
    assert f"releases/1.40.4/{TRACE_TRO_FILENAME}" in operation_paths

    release_ops = [
        operation
        for operation in operations
        if operation.path_in_repo.endswith("release_manifest.json")
    ]
    assert len(release_ops) == 2
    for operation in release_ops:
        assert isinstance(operation, CommitOperationAdd)
        assert isinstance(operation.path_or_fileobj, BytesIO)

    trace_ops = [
        operation for operation in operations if operation.path_in_repo.endswith(".jsonld")
    ]
    assert len(trace_ops) == 2
    for operation in trace_ops:
        assert isinstance(operation, CommitOperationAdd)
        assert isinstance(operation.path_or_fileobj, BytesIO)
