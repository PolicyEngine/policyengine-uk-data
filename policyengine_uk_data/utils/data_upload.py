from io import BytesIO
from typing import Dict, List, Optional, Tuple
from huggingface_hub import HfApi, CommitOperationAdd, hf_hub_download
from huggingface_hub.errors import RevisionNotFoundError
from google.cloud import storage
from pathlib import Path
from importlib import metadata
import google.auth
import json
import logging
import os

from policyengine_uk_data.utils.release_manifest import (
    build_release_manifest,
    serialize_release_manifest,
)

RELEASE_MANIFEST_PATH = "release_manifest.json"


def _get_model_package_version(
    package_name: str = "policyengine-uk",
) -> str:
    try:
        return metadata.version(package_name)
    except metadata.PackageNotFoundError:
        raise RuntimeError(
            "Could not determine installed version for "
            f"{package_name} while building a release manifest."
        )


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
            return None
        except Exception:
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
    if (
        load_release_manifest_from_hf(
            version=version,
            hf_repo_name=hf_repo_name,
            hf_repo_type=hf_repo_type,
            revision=version,
        )
        is not None
    ):
        raise RuntimeError(
            f"Release {version} is already finalized on {hf_repo_name}. "
            "Refusing to mutate release manifest state after the tag exists."
        )


def get_repo_head_revision(
    api: HfApi,
    repo_id: str,
    repo_type: str,
    token: Optional[str] = None,
) -> Optional[str]:
    repo_info = api.repo_info(
        repo_id=repo_id,
        repo_type=repo_type,
        token=token,
    )
    return getattr(repo_info, "sha", None)


def create_release_manifest_commit_operations(
    files_with_repo_paths: List[Tuple[Path, str]],
    version: str,
    hf_repo_name: str = "policyengine/policyengine-uk-data-private",
    model_package_name: str = "policyengine-uk",
    model_package_version: Optional[str] = None,
    existing_manifest: Optional[Dict] = None,
) -> Tuple[Dict, List[CommitOperationAdd]]:
    if not model_package_version:
        raise RuntimeError(
            "A compatible policyengine-uk version is required when publishing "
            "a release manifest."
        )
    manifest = build_release_manifest(
        files_with_repo_paths=files_with_repo_paths,
        version=version,
        repo_id=hf_repo_name,
        model_package_name=model_package_name,
        model_package_version=model_package_version,
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
    hf_repo_name: str = "policyengine/policyengine-uk-data-private",
    hf_repo_type: str = "model",
    version: str = None,
):
    if version is None:
        version = metadata.version("policyengine-uk-data")

    commit_oid = upload_files_to_hf(
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

    create_release_tag(
        version=version,
        revision=commit_oid,
        hf_repo_name=hf_repo_name,
        hf_repo_type=hf_repo_type,
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

    model_package_version = _get_model_package_version()
    existing_manifest = load_release_manifest_from_hf(
        version=version,
        hf_repo_name=hf_repo_name,
        hf_repo_type=hf_repo_type,
    )
    parent_commit = get_repo_head_revision(
        api=api,
        repo_id=hf_repo_name,
        repo_type=hf_repo_type,
        token=token,
    )
    _, manifest_operations = create_release_manifest_commit_operations(
        files_with_repo_paths=files_with_repo_paths,
        version=version,
        hf_repo_name=hf_repo_name,
        model_package_version=model_package_version,
        existing_manifest=existing_manifest,
    )
    hf_operations.extend(manifest_operations)

    commit_info = api.create_commit(
        token=token,
        repo_id=hf_repo_name,
        operations=hf_operations,
        repo_type=hf_repo_type,
        commit_message=f"Upload data files for version {version}",
        parent_commit=parent_commit,
    )
    logging.info(f"Uploaded files to Hugging Face repository {hf_repo_name}.")
    return commit_info.oid


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
            f"Tagged commit with {version} in Hugging Face repository {hf_repo_name}."
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
        else:
            raise


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
