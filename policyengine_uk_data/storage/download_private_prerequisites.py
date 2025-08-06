from policyengine_uk_data.utils.huggingface import download, upload
from pathlib import Path
import zipfile
import warnings


def extract_zipped_folder(folder):
    folder = Path(folder)
    with zipfile.ZipFile(folder, "r") as zip_ref:
        zip_ref.extractall(folder.parent / folder.stem)


def download_prerequisites():
    """Download prerequisite data files from HuggingFace.

    This function downloads and extracts the required data files that are
    typically obtained by running `make download`.
    """
    folder = Path(__file__).parent

    files = [
        "frs_2020_21.zip",
        "frs_2022_23.zip",
        "frs_2023_24.zip",
        "lcfs_2021_22.zip",
        "was_2006_20.zip",
        "etb_1977_21.zip",
        "spi_2020_21.zip",
    ]

    files = [folder / file for file in files]

    for file in files:
        download(
            repo="policyengine/policyengine-uk-data",
            repo_filename=file.name,
            local_folder=file.parent,
        )
        extract_zipped_folder(file)
        file.unlink()


def check_prerequisites():
    """Check if prerequisite data files/folders are present.

    Returns:
        bool: True if all prerequisites are present, False otherwise.
    """
    folder = Path(__file__).parent

    expected_folders = [
        "frs_2020_21",
        "frs_2022_23",
        "frs_2023_24",
        "lcfs_2021_22",
        "was_2006_20",
        "etb_1977_21",
        "spi_2020_21",
    ]

    missing = []
    for folder_name in expected_folders:
        if not (folder / folder_name).exists():
            missing.append(folder_name)

    if missing:
        warnings.warn(
            f"Missing prerequisite data folders: {', '.join(missing)}. "
            f"Run `from policyengine_uk_data import download_prerequisites; download_prerequisites()` "
            f"to download them.",
            UserWarning,
            stacklevel=3,
        )
        return False

    return True


# Keep backwards compatibility for direct script execution
if __name__ == "__main__":
    download_prerequisites()
