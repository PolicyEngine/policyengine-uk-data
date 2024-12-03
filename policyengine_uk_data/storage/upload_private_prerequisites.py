from policyengine_uk_data.utils.huggingface import upload
from pathlib import Path
import zipfile


def zip_folder(folder):
    folder = Path(folder)
    with zipfile.ZipFile(folder.with_suffix(".zip"), "w") as zip_ref:
        for file in folder.glob("*"):
            zip_ref.write(file, file.name)


FOLDER = Path(__file__).parent

FILES = [
    "frs_2020_21.zip",
    "frs_2022_23.zip",
    "lcfs_2021_22.zip",
    "was_2006_20.zip",
    "etb_1977_21.zip",
    "spi_2020_21.zip",
]

FILES = [Path(FOLDER / file) for file in FILES]

for file in FILES:
    if not file.exists():
        zip_folder(FOLDER / file.name[:-4])
    if not file.exists():
        raise FileNotFoundError(f"File {file} not found")
    upload(
        repo="policyengine/policyengine-uk-data",
        repo_file_path=file.name,
        local_file_path=file,
    )
