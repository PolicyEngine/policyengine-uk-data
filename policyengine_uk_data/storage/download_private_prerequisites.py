from policyengine_uk_data.utils.huggingface import download, upload
from pathlib import Path
import zipfile


def extract_zipped_folder(folder):
    folder = Path(folder)
    with zipfile.ZipFile(folder, "r") as zip_ref:
        zip_ref.extractall(folder.parent / folder.stem)


FOLDER = Path(__file__).parent

FILES = [
    "frs_2020_21.zip",
    "frs_2022_23.zip",
    "lcfs_2021_22.zip",
    "was_2006_20.zip",
    "etb_1977_21.zip",
    "spi_2020_21.zip",
]

FILES = [FOLDER / file for file in FILES]

for file in FILES:
    download(
        repo="policyengine/policyengine-uk-data",
        repo_filename=file.name,
        local_folder=file.parent,
    )
    print(f"Extracting {file}")
    extract_zipped_folder(file)
    file.unlink()
