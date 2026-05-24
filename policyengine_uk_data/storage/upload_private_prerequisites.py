from policyengine_uk_data.storage.download_private_prerequisites import (
    PRIVATE_PREREQUISITES,
)
from policyengine_uk_data.utils.hf_destinations import PRIVATE_REPO
from policyengine_uk_data.utils.huggingface import upload
from pathlib import Path
import zipfile


def zip_folder(folder):
    folder = Path(folder)
    with zipfile.ZipFile(folder.with_suffix(".zip"), "w") as zip_ref:
        for file in folder.glob("*"):
            zip_ref.write(file, file.name)


FOLDER = Path(__file__).parent

FILES = [Path(FOLDER / filename) for filename, _ in PRIVATE_PREREQUISITES]

for file in FILES:
    if not file.exists():
        zip_folder(FOLDER / file.name[:-4])
    if not file.exists():
        raise FileNotFoundError(f"File {file} not found")
    upload(
        repo=PRIVATE_REPO,
        repo_file_path=file.name,
        local_file_path=file,
    )
