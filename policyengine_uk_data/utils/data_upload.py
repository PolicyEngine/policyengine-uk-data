from typing import List
from huggingface_hub import HfApi, CommitOperationAdd
from huggingface_hub.errors import RevisionNotFoundError
from google.cloud import storage
from pathlib import Path
from importlib import metadata
import google.auth


def upload_data_files(
    files: List[str],
    gcs_bucket_name: str = "policyengine-uk-data-private",
    hf_repo_name: str = "policyengine/policyengine-uk-data",
    hf_repo_type: str = "model",
):
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
    print(f"Uploaded files to Hugging Face repository {hf_repo_name}.")
    # Tag commit with version
    tag_name = version

    # Delete the tag if it already exists to ensure the new commit is tagged.
    # missing_ok=True ensures that if the tag doesn't exist, no error is raised.

    try:
        api.delete_tag(
            repo_id=hf_repo_name,
            tag=tag_name,
            repo_type=hf_repo_type,
        )
        print(f"Tag {version} already exists: deleting the old tag.")
    except RevisionNotFoundError:
        pass

    # Create the new tag
    api.create_tag(
        repo_id=hf_repo_name,
        tag=tag_name,
        revision=commit_info.oid,
        repo_type=hf_repo_type,
    )
    print(
        f"Tagged commit with {tag_name} in Hugging Face repository {hf_repo_name}."
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
        blob.metadata = {"version": version}
        blob.upload_from_filename(file_path)
        print(f"Uploaded {file_path.name} to GCS bucket {gcs_bucket_name}.")

        # Set metadata
        blob.metadata = {"version": version}
        blob.patch()
        print(
            f"Set metadata for {file_path.name} in GCS bucket {gcs_bucket_name}."
        )
