from typing import List
from huggingface_hub import HfApi, CommitOperationAdd
from huggingface_hub.errors import RevisionNotFoundError
from google.cloud import storage
from pathlib import Path
from importlib import metadata
import google.auth
import logging


def upload_data_files(
    files: List[str],
    gcs_bucket_name: str = "policyengine-uk-data-private",
    hf_repo_name: str = "policyengine/policyengine-uk-data",
    hf_repo_type: str = "model",
    version: str = None,
):
    if version is None:
        version = metadata.version("policyengine-uk-data")

    api = HfApi()
    hf_operations = []

    for file_path in files:
        file_path = Path(file_path)
        if not file_path.exists():
            raise ValueError(f"File {file_path} does not exist.")
        hf_operations.append(
            CommitOperationAdd(
                path_in_repo=file_path.name,
                path_or_fileobj=str(file_path),
            )
        )
    commit_info = api.create_commit(
        repo_id=hf_repo_name,
        operations=hf_operations,
        repo_type=hf_repo_type,
        commit_message=f"Upload data files for version {version}",
    )
    logging.info(f"Uploaded files to Hugging Face repository {hf_repo_name}.")
    # Tag commit with version

    # Create the new tag
    api.create_tag(
        repo_id=hf_repo_name,
        tag=version,
        revision=commit_info.oid,
        repo_type=hf_repo_type,
    )
    logging.info(
        f"Tagged commit with {version} in Hugging Face repository {hf_repo_name}."
    )

    # Upload to GCS
    credentials, project_id = google.auth.default()
    storage_client = storage.Client(
        credentials=credentials, project=project_id
    )
    bucket = storage_client.bucket(gcs_bucket_name)
    for file_path in files:
        file_path = Path(file_path)
        blob = bucket.blob(file_path.name)
        blob.upload_from_filename(file_path)
        logging.info(
            f"Uploaded {file_path.name} to GCS bucket {gcs_bucket_name}."
        )

        # Set metadata
        blob.metadata = {"version": version}
        blob.patch()
        logging.info(
            f"Set metadata for {file_path.name} in GCS bucket {gcs_bucket_name}."
        )
