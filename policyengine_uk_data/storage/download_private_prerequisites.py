from policyengine_uk_data.utils.huggingface import download, upload
from pathlib import Path
import zipfile


def extract_zipped_folder(folder):
    folder = Path(folder)
    with zipfile.ZipFile(folder, "r") as zip_ref:
        zip_ref.extractall(folder.parent)


FOLDER = Path(__file__).parent

FILES = [
    "frs_2022_23.zip",
    "lcfs_2021_22.zip",
    "was_2006_20.zip",
    "etb_1977_21.zip",
    "spi_2020_21.zip",
]

FILES = [Path(file) for file in FILES]

for file in FILES:
    if file.exists():
        continue
    download(
        repo="policyengine/policyengine-uk-data",
        repo_filename=file.name,
        local_folder=file.parent,
    )
    extract_zipped_folder(FOLDER / file)
    (FOLDER / file).unlink()
