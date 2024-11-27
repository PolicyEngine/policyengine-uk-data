from huggingface_hub import hf_hub_download, login, HfApi
import os
import pkg_resources


def download(
    repo: str, repo_filename: str, local_folder: str, version: str = None
):
    token = os.environ.get(
        "HUGGING_FACE_TOKEN",
    )
    login(token=token)

    hf_hub_download(
        repo_id=repo,
        repo_type="model",
        filename=repo_filename,
        local_dir=local_folder,
        revision=version,
    )


def upload(local_file_path: str, repo: str, repo_file_path: str):
    token = os.environ.get(
        "HUGGING_FACE_TOKEN",
    )
    login(token=token)
    api = HfApi()
    api.upload_file(
        path_or_fileobj=local_file_path,
        path_in_repo=repo_file_path,
        repo_id=repo,
        repo_type="model",
    )
