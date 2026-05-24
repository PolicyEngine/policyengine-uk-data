from policyengine_uk_data.datasets.frs_release import CURRENT_FRS_RELEASE
from policyengine_uk_data.datasets.private_releases import (
    CURRENT_ETB_RELEASE,
    CURRENT_LCFS_RELEASE,
    CURRENT_WAS_RELEASE,
)
from policyengine_uk_data.datasets.spi import SPI_RELEASE_NAME
from policyengine_uk_data.utils.hf_destinations import PRIVATE_REPO
from policyengine_uk_data.utils.huggingface import download
from pathlib import Path
from pathlib import PurePosixPath
import shutil
import zipfile
import warnings


PRIVATE_PREREQUISITES = [
    (CURRENT_FRS_RELEASE.raw_zip_name, CURRENT_FRS_RELEASE.ukds_tab_subdir),
    (CURRENT_LCFS_RELEASE.raw_zip_name, CURRENT_LCFS_RELEASE.ukds_tab_subdir),
    (CURRENT_WAS_RELEASE.raw_zip_name, CURRENT_WAS_RELEASE.ukds_tab_subdir),
    (CURRENT_ETB_RELEASE.raw_zip_name, CURRENT_ETB_RELEASE.ukds_tab_subdir),
    (f"{SPI_RELEASE_NAME}.zip", None),
]


def _validate_zip_path(path: PurePosixPath) -> None:
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Unsafe path in zip file: {path}")


def _copy_zip_member(zip_ref, member, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zip_ref.open(member) as source, open(destination, "wb") as target:
        shutil.copyfileobj(source, target)


def _extract_all(zip_ref, destination):
    destination = Path(destination)
    for member in zip_ref.infolist():
        if member.is_dir():
            continue
        member_path = PurePosixPath(member.filename)
        _validate_zip_path(member_path)
        _copy_zip_member(zip_ref, member, destination.joinpath(*member_path.parts))


def _extract_tab_subdir(zip_ref, tab_subdir, destination):
    prefix = PurePosixPath(tab_subdir)
    extracted = set()
    for member in zip_ref.infolist():
        if member.is_dir():
            continue
        member_path = PurePosixPath(member.filename)
        _validate_zip_path(member_path)
        try:
            relative_path = member_path.relative_to(prefix)
        except ValueError:
            continue
        if len(relative_path.parts) != 1:
            continue
        filename = relative_path.name
        if filename in extracted:
            raise ValueError(f"Duplicate FRS TAB filename in zip file: {filename}")
        _copy_zip_member(zip_ref, member, Path(destination) / filename)
        extracted.add(filename)
    return len(extracted)


def _extract_flat_files(zip_ref, destination):
    extracted_count = 0
    for member in zip_ref.infolist():
        if member.is_dir():
            continue
        member_path = PurePosixPath(member.filename)
        _validate_zip_path(member_path)
        if len(member_path.parts) != 1:
            continue
        _copy_zip_member(zip_ref, member, Path(destination) / member_path.name)
        extracted_count += 1
    return extracted_count


def extract_zipped_folder(folder, tab_subdir=None):
    folder = Path(folder)
    destination = folder.parent / folder.stem
    with zipfile.ZipFile(folder, "r") as zip_ref:
        if tab_subdir is None:
            _extract_all(zip_ref, destination)
            return

        extracted_count = _extract_tab_subdir(zip_ref, tab_subdir, destination)
        if extracted_count == 0:
            extracted_count = _extract_flat_files(zip_ref, destination)
        if extracted_count == 0:
            raise ValueError(
                f"No files found under {tab_subdir!r} or at the zip root in {folder}."
            )


def download_prerequisites():
    """Download prerequisite data files from HuggingFace.

    This function downloads and extracts the required data files that are
    typically obtained by running `make download`.
    """
    folder = Path(__file__).parent

    for filename, tab_subdir in PRIVATE_PREREQUISITES:
        file = folder / filename
        download(
            repo=PRIVATE_REPO,
            repo_filename=file.name,
            local_folder=file.parent,
        )
        extract_zipped_folder(file, tab_subdir=tab_subdir)
        file.unlink()


def check_prerequisites():
    """Check if prerequisite data files/folders are present.

    Returns:
        bool: True if all prerequisites are present, False otherwise.
    """
    folder = Path(__file__).parent

    expected_folders = [Path(filename).stem for filename, _ in PRIVATE_PREREQUISITES]

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
