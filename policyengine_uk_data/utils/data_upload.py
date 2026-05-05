from io import BytesIO
from typing import Any, Dict, List, Mapping, Optional, Tuple
from huggingface_hub import HfApi, CommitOperationAdd, hf_hub_download
from huggingface_hub.errors import EntryNotFoundError, RevisionNotFoundError
from google.cloud import storage
from pathlib import Path
from importlib import metadata
from importlib.util import find_spec
import google.auth
import json
import logging
import os
import subprocess
import tomllib

from policyengine_uk_data.utils.release_manifest import (
    build_release_manifest,
    serialize_release_manifest,
)

RELEASE_MANIFEST_PATH = "release_manifest.json"
REPO_ROOT = Path(__file__).resolve().parents[2]


def _get_model_package_version(
    package_name: str = "policyengine-uk",
) -> Optional[str]:
    module_name = package_name.replace("-", "_")
    spec = find_spec(module_name)
    module_origin = getattr(spec, "origin", None) if spec is not None else None
    if module_origin is not None:
        package_root = Path(module_origin).resolve().parent
        for parent in [package_root, *package_root.parents]:
            pyproject_path = parent / "pyproject.toml"
            if not pyproject_path.exists():
                continue
            with open(pyproject_path, "rb") as f:
                pyproject = tomllib.load(f)
            project = pyproject.get("project", {})
            if project.get("name") == package_name and project.get("version"):
                return project["version"]
    try:
        return metadata.version(package_name)
    except metadata.PackageNotFoundError:
        logging.warning(
            "Could not determine installed version for %s while building release manifest.",
            package_name,
        )
        return None


def _get_model_package_build_metadata(
    package_name: str = "policyengine-uk",
) -> Dict[str, Any]:
    metadata_payload: Dict[str, Any] = {
        "version": _get_model_package_version(package_name),
        "git_sha": None,
        "data_build_fingerprint": None,
        "core": None,
    }
    module_name = package_name.replace("-", "_")
    try:
        build_metadata_module = __import__(
            f"{module_name}.build_metadata",
            fromlist=["get_runtime_metadata", "get_data_build_metadata"],
        )
        for metadata_getter_name in (
            "get_runtime_metadata",
            "get_data_build_metadata",
        ):
            metadata_getter = getattr(build_metadata_module, metadata_getter_name, None)
            if not callable(metadata_getter):
                continue
            package_metadata = metadata_getter()
            metadata_payload["version"] = (
                package_metadata.get("version") or metadata_payload["version"]
            )
            metadata_payload["git_sha"] = package_metadata.get("git_sha")
            metadata_payload["data_build_fingerprint"] = package_metadata.get(
                "data_build_fingerprint"
            )
            metadata_payload["core"] = package_metadata.get("core")
            break
    except Exception:
        logging.warning(
            "Could not load build metadata from %s while building release manifest.",
            package_name,
            exc_info=True,
        )
    return metadata_payload


def _get_core_package_runtime_metadata(
    package_name: str = "policyengine-core",
) -> Dict[str, Any] | None:
    try:
        from policyengine_core import get_runtime_metadata

        return dict(get_runtime_metadata())
    except (AttributeError, ImportError):
        pass
    except Exception:
        logging.warning(
            "Could not load runtime metadata from %s while building release manifest.",
            package_name,
            exc_info=True,
        )

    try:
        return {
            "name": package_name,
            "version": metadata.version(package_name),
        }
    except metadata.PackageNotFoundError:
        logging.warning(
            "Could not determine installed version for %s while building release manifest.",
            package_name,
        )
        return None


def _get_data_package_git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        logging.warning(
            "Could not determine policyengine-uk-data git SHA while building release manifest.",
            exc_info=True,
        )
        return None


def load_release_manifest_from_hf(
    version: str,
    hf_repo_name: str = "policyengine/policyengine-uk-data-private",
    hf_repo_type: str = "model",
    revision: Optional[str] = None,
) -> Optional[Dict]:
    token = os.environ.get("HUGGING_FACE_TOKEN")
    candidate_paths = [
        f"releases/{version}/{RELEASE_MANIFEST_PATH}",
        RELEASE_MANIFEST_PATH,
    ]

    for path_in_repo in candidate_paths:
        try:
            manifest_path = hf_hub_download(
                repo_id=hf_repo_name,
                filename=path_in_repo,
                repo_type=hf_repo_type,
                token=token,
                revision=revision,
            )
        except RevisionNotFoundError:
            if revision is not None:
                return None
            raise
        except EntryNotFoundError:
            continue

        with open(manifest_path) as f:
            manifest = json.load(f)

        data_package = manifest.get("data_package", {})
        if data_package.get("version") == version:
            return manifest

    return None


def assert_release_not_finalized(
    version: str,
    hf_repo_name: str = "policyengine/policyengine-uk-data-private",
    hf_repo_type: str = "model",
) -> None:
    finalized_manifest = load_release_manifest_from_hf(
        version=version,
        hf_repo_name=hf_repo_name,
        hf_repo_type=hf_repo_type,
        revision=version,
    )
    if finalized_manifest is not None:
        raise RuntimeError(
            f"Release {version} is already finalized on {hf_repo_name}. "
            "Refusing to mutate release manifest state after the tag exists."
        )


def create_release_tag(
    version: str,
    revision: str,
    hf_repo_name: str = "policyengine/policyengine-uk-data-private",
    hf_repo_type: str = "model",
    token: Optional[str] = None,
    api: Optional[HfApi] = None,
) -> None:
    api = api or HfApi()
    token = token or os.environ.get("HUGGING_FACE_TOKEN")
    try:
        api.create_tag(
            token=token,
            repo_id=hf_repo_name,
            tag=version,
            revision=revision,
            repo_type=hf_repo_type,
            exist_ok=False,
        )
        logging.info(
            "Tagged revision %s with %s in Hugging Face repository %s.",
            revision,
            version,
            hf_repo_name,
        )
    except Exception as e:
        if "Tag reference exists already" in str(e) or "409" in str(e):
            tagged_revision = getattr(
                api.repo_info(
                    repo_id=hf_repo_name,
                    repo_type=hf_repo_type,
                    revision=version,
                    token=token,
                ),
                "sha",
                None,
            )
            if tagged_revision == revision:
                logging.info(
                    "Tag %s already exists in %s and already points to %s.",
                    version,
                    hf_repo_name,
                    revision,
                )
                return
            raise RuntimeError(
                f"Tag {version} already exists in {hf_repo_name} at "
                f"{tagged_revision}; refusing to treat {revision} as finalized."
            ) from e
        raise


def create_release_manifest_commit_operations(
    files_with_repo_paths: List[Tuple[Path, str]],
    version: str,
    hf_repo_name: str = "policyengine/policyengine-uk-data-private",
    hf_repo_type: str = "model",
    model_package_name: str = "policyengine-uk",
    model_package_version: Optional[str] = None,
    model_package_git_sha: Optional[str] = None,
    model_package_data_build_fingerprint: Optional[str] = None,
    core_package_metadata: Optional[Mapping[str, Any]] = None,
    data_package_git_sha: Optional[str] = None,
    existing_manifest: Optional[Dict] = None,
) -> Tuple[Dict, List[CommitOperationAdd]]:
    manifest = build_release_manifest(
        files_with_repo_paths=files_with_repo_paths,
        version=version,
        repo_id=hf_repo_name,
        repo_type=hf_repo_type,
        model_package_name=model_package_name,
        model_package_version=model_package_version,
        model_package_git_sha=model_package_git_sha,
        model_package_data_build_fingerprint=model_package_data_build_fingerprint,
        core_package_metadata=core_package_metadata,
        data_package_git_sha=data_package_git_sha,
        existing_manifest=existing_manifest,
    )
    manifest_payload = serialize_release_manifest(manifest)
    operations = [
        CommitOperationAdd(
            path_in_repo=RELEASE_MANIFEST_PATH,
            path_or_fileobj=BytesIO(manifest_payload),
        ),
        CommitOperationAdd(
            path_in_repo=f"releases/{version}/{RELEASE_MANIFEST_PATH}",
            path_or_fileobj=BytesIO(manifest_payload),
        ),
    ]
    return manifest, operations


def upload_data_files(
    files: List[str],
    gcs_bucket_name: str = "policyengine-uk-data-private",
    hf_repo_name: str = "policyengine/policyengine-uk-data",
    hf_repo_type: str = "model",
    version: str = None,
):
    if version is None:
        version = metadata.version("policyengine-uk-data")

    upload_files_to_hf(
        files=files,
        version=version,
        hf_repo_name=hf_repo_name,
        hf_repo_type=hf_repo_type,
    )

    upload_files_to_gcs(
        files=files,
        version=version,
        gcs_bucket_name=gcs_bucket_name,
    )


def upload_files_to_hf(
    files: List[str],
    version: str,
    hf_repo_name: str = "policyengine/policyengine-uk-data-private",
    hf_repo_type: str = "model",
):
    """
    Upload files to Hugging Face repository and tag the commit with the version.
    """
    api = HfApi()
    token = os.environ.get(
        "HUGGING_FACE_TOKEN",
    )
    assert_release_not_finalized(
        version=version,
        hf_repo_name=hf_repo_name,
        hf_repo_type=hf_repo_type,
    )
    hf_operations = []
    files_with_repo_paths = []

    for file_path in files:
        file_path = Path(file_path)
        if not file_path.exists():
            raise ValueError(f"File {file_path} does not exist.")
        repo_path = file_path.name
        files_with_repo_paths.append((file_path, repo_path))
        hf_operations.append(
            CommitOperationAdd(
                path_in_repo=repo_path,
                path_or_fileobj=str(file_path),
            )
        )

    existing_manifest = load_release_manifest_from_hf(
        version=version,
        hf_repo_name=hf_repo_name,
        hf_repo_type=hf_repo_type,
    )
    model_build_metadata = _get_model_package_build_metadata()
    core_package_metadata = (
        model_build_metadata.get("core") or _get_core_package_runtime_metadata()
    )
    _, manifest_operations = create_release_manifest_commit_operations(
        files_with_repo_paths=files_with_repo_paths,
        version=version,
        hf_repo_name=hf_repo_name,
        hf_repo_type=hf_repo_type,
        model_package_version=model_build_metadata["version"],
        model_package_git_sha=model_build_metadata["git_sha"],
        model_package_data_build_fingerprint=model_build_metadata[
            "data_build_fingerprint"
        ],
        core_package_metadata=core_package_metadata,
        data_package_git_sha=_get_data_package_git_sha(),
        existing_manifest=existing_manifest,
    )
    hf_operations.extend(manifest_operations)

    commit_info = api.create_commit(
        token=token,
        repo_id=hf_repo_name,
        operations=hf_operations,
        repo_type=hf_repo_type,
        commit_message=f"Upload data files for version {version}",
    )
    logging.info(f"Uploaded files to Hugging Face repository {hf_repo_name}.")

    create_release_tag(
        version=version,
        revision=commit_info.oid,
        hf_repo_name=hf_repo_name,
        hf_repo_type=hf_repo_type,
        token=token,
        api=api,
    )


def upload_files_to_gcs(
    files: List[str],
    version: str,
    gcs_bucket_name: str = "policyengine-uk-data-private",
):
    """
    Upload files to Google Cloud Storage and set metadata with the version.
    """
    credentials, project_id = google.auth.default()
    storage_client = storage.Client(credentials=credentials, project=project_id)
    bucket = storage_client.bucket(gcs_bucket_name)

    for file_path in files:
        file_path = Path(file_path)
        blob = bucket.blob(file_path.name)
        blob.upload_from_filename(file_path)
        logging.info(f"Uploaded {file_path.name} to GCS bucket {gcs_bucket_name}.")

        # Set metadata
        blob.metadata = {"version": version}
        blob.patch()
        logging.info(
            f"Set metadata for {file_path.name} in GCS bucket {gcs_bucket_name}."
        )
