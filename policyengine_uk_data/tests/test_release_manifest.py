import hashlib
from io import BytesIO
from importlib import metadata
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from huggingface_hub import CommitOperationAdd
from huggingface_hub.errors import EntryNotFoundError, RevisionNotFoundError

from policyengine_uk_data.utils.data_upload import (
    _get_model_package_version,
    load_release_manifest_from_hf,
    upload_files_to_hf,
)
from policyengine_uk_data.utils.release_manifest import (
    RELEASE_MANIFEST_SCHEMA_VERSION,
    build_release_manifest,
)

# Synthetic fixture: this verifies manifest propagation, not the package dep range.
CORE_FIXTURE_VERSION = "9.8.7"
EXPECTED_CORE_PACKAGE = {
    "name": "policyengine-core",
    "version": CORE_FIXTURE_VERSION,
}
EXPECTED_COMPATIBLE_CORE_PACKAGES = [
    {"name": "policyengine-core", "specifier": f"=={CORE_FIXTURE_VERSION}"}
]


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
        core_package_metadata=EXPECTED_CORE_PACKAGE,
        data_package_git_sha="cafebabe",
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
    assert manifest["compatible_core_packages"] == EXPECTED_COMPATIBLE_CORE_PACKAGES
    assert manifest["build"] == {
        "build_id": "policyengine-uk-data-1.40.4",
        "built_at": "2026-04-10T12:00:00Z",
        "metadata": {
            "data_package_git_sha": "cafebabe",
        },
        "built_with_model_package": {
            "name": "policyengine-uk",
            "version": "2.74.0",
            "git_sha": "deadbeef",
            "data_build_fingerprint": "sha256:fingerprint",
            "core": EXPECTED_CORE_PACKAGE,
        },
        "built_with_core_package": EXPECTED_CORE_PACKAGE,
    }
    assert "created_at" not in manifest
    assert manifest["metadata"] == {
        "artifact_release": {
            "repo_id": "policyengine/policyengine-uk-data-private",
            "repo_type": "model",
            "version": "1.40.4",
            "visibility": "private",
        }
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
    assert manifest["artifacts"]["enhanced_frs_2023_24"]["uri"] == (
        "hf://model/policyengine/policyengine-uk-data-private@1.40.4/"
        "enhanced_frs_2023_24.h5"
    )
    assert manifest["artifacts"]["enhanced_frs_2023_24"]["metadata"] == {
        "repo_type": "model",
        "visibility": "private",
    }


def test_build_release_manifest_validates_against_bundle_contract(tmp_path):
    policyengine_bundles = pytest.importorskip("policyengine_bundles")
    dataset_path = _write_file(
        tmp_path / "enhanced_frs_2023_24.h5",
        b"enhanced-frs",
    )

    manifest = build_release_manifest(
        files_with_repo_paths=[(dataset_path, "enhanced_frs_2023_24.h5")],
        version="1.40.4",
        repo_id="policyengine/policyengine-uk-data-private",
        model_package_version="2.74.0",
        model_package_git_sha="deadbeef",
        model_package_data_build_fingerprint="sha256:fingerprint",
        core_package_metadata=EXPECTED_CORE_PACKAGE,
        data_package_git_sha="cafebabe",
        created_at="2026-04-10T12:00:00Z",
    )

    policyengine_bundles.DataReleaseManifest.model_validate(manifest)


def test_build_release_manifest_refreshes_compatible_model_packages_for_draft_retry(
    tmp_path,
):
    dataset_path = _write_file(
        tmp_path / "enhanced_frs_2023_24.h5",
        b"enhanced-frs",
    )

    manifest = build_release_manifest(
        files_with_repo_paths=[(dataset_path, "enhanced_frs_2023_24.h5")],
        version="1.40.4",
        repo_id="policyengine/policyengine-uk-data-private",
        model_package_version="9.99.9",
        existing_manifest={
            "schema_version": RELEASE_MANIFEST_SCHEMA_VERSION,
            "data_package": {
                "name": "policyengine-uk-data",
                "version": "1.40.4",
            },
            "compatible_model_packages": [
                {
                    "name": "policyengine-uk",
                    "specifier": "==1.0.0",
                }
            ],
            "compatible_core_packages": [],
            "default_datasets": {},
            "created_at": "2026-04-10T12:00:00Z",
            "artifacts": {},
        },
    )

    assert manifest["compatible_model_packages"] == [
        {"name": "policyengine-uk", "specifier": "==9.99.9"}
    ]


def test_load_release_manifest_from_hf_raises_non_missing_download_errors():
    with patch(
        "policyengine_uk_data.utils.data_upload.hf_hub_download",
        side_effect=RuntimeError("boom"),
    ):
        with pytest.raises(RuntimeError, match="boom"):
            load_release_manifest_from_hf(version="1.40.4")


def test_load_release_manifest_from_hf_continues_on_missing_entry(tmp_path):
    manifest_path = tmp_path / "release_manifest.json"
    manifest_path.write_text('{"data_package": {"version": "1.40.4"}}')

    with patch(
        "policyengine_uk_data.utils.data_upload.hf_hub_download",
        side_effect=[
            EntryNotFoundError("missing"),
            str(manifest_path),
        ],
    ):
        manifest = load_release_manifest_from_hf(version="1.40.4")

    assert manifest["data_package"]["version"] == "1.40.4"


def test_load_release_manifest_from_hf_uses_explicit_revision_when_requested(tmp_path):
    manifest_path = tmp_path / "release_manifest.json"
    manifest_path.write_text('{"data_package": {"version": "1.40.4"}}')

    with patch(
        "policyengine_uk_data.utils.data_upload.hf_hub_download",
        return_value=str(manifest_path),
    ) as mock_download:
        manifest = load_release_manifest_from_hf(
            version="1.40.4",
            revision="1.40.4",
        )

    assert manifest["data_package"]["version"] == "1.40.4"
    assert mock_download.call_args.kwargs["revision"] == "1.40.4"


def test_load_release_manifest_from_hf_returns_none_when_revision_is_missing():
    with patch(
        "policyengine_uk_data.utils.data_upload.hf_hub_download",
        side_effect=RevisionNotFoundError(
            "missing revision",
            response=MagicMock(),
        ),
    ):
        assert (
            load_release_manifest_from_hf(
                version="1.40.4",
                revision="1.40.4",
            )
            is None
        )


def test_get_model_package_version_prefers_imported_checkout(tmp_path):
    package_root = tmp_path / "policyengine_uk"
    package_root.mkdir()
    (package_root / "__init__.py").write_text("")
    pyproject_path = tmp_path / "pyproject.toml"
    pyproject_path.write_text(
        '[project]\nname = "policyengine-uk"\nversion = "2.78.0"\n'
    )
    fake_spec = MagicMock(origin=str(package_root / "__init__.py"))

    with (
        patch(
            "policyengine_uk_data.utils.data_upload.find_spec", return_value=fake_spec
        ),
        patch(
            "policyengine_uk_data.utils.data_upload.metadata.version",
            side_effect=metadata.PackageNotFoundError,
        ),
    ):
        assert _get_model_package_version() == "2.78.0"


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
                "core": EXPECTED_CORE_PACKAGE,
            },
        ),
        patch(
            "policyengine_uk_data.utils.data_upload._get_data_package_git_sha",
            return_value="cafebabe",
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

    payload = release_ops[0].path_or_fileobj.getvalue()
    manifest = json.loads(payload.decode("utf-8"))
    assert manifest["compatible_core_packages"] == EXPECTED_COMPATIBLE_CORE_PACKAGES
    assert manifest["build"]["built_with_core_package"] == EXPECTED_CORE_PACKAGE
    assert manifest["build"]["metadata"] == {
        "data_package_git_sha": "cafebabe",
    }
    assert (
        manifest["build"]["built_with_model_package"]["core"] == EXPECTED_CORE_PACKAGE
    )


def test_upload_files_to_hf_rejects_finalized_release(tmp_path):
    dataset_path = _write_file(
        tmp_path / "enhanced_frs_2023_24.h5",
        b"enhanced-frs",
    )
    finalized_manifest = {
        "schema_version": RELEASE_MANIFEST_SCHEMA_VERSION,
        "data_package": {
            "name": "policyengine-uk-data",
            "version": "1.40.4",
        },
        "compatible_model_packages": [
            {"name": "policyengine-uk", "specifier": "==2.74.0"}
        ],
        "default_datasets": {"national": "enhanced_frs_2023_24"},
        "artifacts": {
            "enhanced_frs_2023_24": {
                "kind": "microdata",
                "path": "enhanced_frs_2023_24.h5",
                "repo_id": "policyengine/policyengine-uk-data-private",
                "revision": "1.40.4",
                "sha256": _sha256(b"enhanced-frs"),
                "size_bytes": len(b"enhanced-frs"),
            }
        },
    }
    mock_api = MagicMock()

    with (
        patch("policyengine_uk_data.utils.data_upload.HfApi", return_value=mock_api),
        patch(
            "policyengine_uk_data.utils.data_upload.load_release_manifest_from_hf",
            side_effect=lambda *args, **kwargs: (
                finalized_manifest if kwargs.get("revision") == "1.40.4" else None
            ),
        ),
    ):
        with pytest.raises(RuntimeError, match="already finalized"):
            upload_files_to_hf(
                files=[dataset_path],
                version="1.40.4",
            )

    mock_api.create_commit.assert_not_called()


def test_upload_files_to_hf_rejects_tag_that_points_to_different_revision(tmp_path):
    dataset_path = _write_file(
        tmp_path / "enhanced_frs_2023_24.h5",
        b"enhanced-frs",
    )
    mock_api = MagicMock()
    mock_api.create_commit.return_value = MagicMock(oid="new-commit")
    mock_api.create_tag.side_effect = RuntimeError("409 Tag reference exists already")
    mock_api.repo_info.return_value = MagicMock(sha="old-commit")

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
                "core": EXPECTED_CORE_PACKAGE,
            },
        ),
        patch(
            "policyengine_uk_data.utils.data_upload._get_data_package_git_sha",
            return_value="cafebabe",
        ),
    ):
        with pytest.raises(RuntimeError, match="refusing to treat new-commit"):
            upload_files_to_hf(
                files=[dataset_path],
                version="1.40.4",
            )
