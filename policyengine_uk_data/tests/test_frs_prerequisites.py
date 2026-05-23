import zipfile

import pytest

from policyengine_uk_data.datasets.frs_release import CURRENT_FRS_RELEASE
from policyengine_uk_data.storage.download_private_prerequisites import (
    PRIVATE_PREREQUISITES,
    extract_zipped_folder,
)


def test_private_prerequisites_use_current_frs_release():
    prerequisite_names = [filename for filename, _ in PRIVATE_PREREQUISITES]

    assert CURRENT_FRS_RELEASE.raw_zip_name in prerequisite_names
    assert "frs_2023_24.zip" not in prerequisite_names


def test_current_frs_release_uses_survey_year_as_base_year():
    assert CURRENT_FRS_RELEASE.base_year == CURRENT_FRS_RELEASE.survey_year


def test_extract_zipped_folder_flattens_current_ukds_tab_layout(tmp_path):
    zip_path = tmp_path / CURRENT_FRS_RELEASE.raw_zip_name
    with zipfile.ZipFile(zip_path, "w") as zip_ref:
        zip_ref.writestr("UKDA-9563-tab/tab/adult.tab", "adult")
        zip_ref.writestr("UKDA-9563-tab/tab/househol.tab", "household")
        zip_ref.writestr("UKDA-9563-tab/mrdoc/pdf/9563_userguide.pdf", "docs")

    extract_zipped_folder(
        zip_path,
        tab_subdir=CURRENT_FRS_RELEASE.ukds_tab_subdir,
    )

    destination = tmp_path / CURRENT_FRS_RELEASE.name
    assert (destination / "adult.tab").read_text() == "adult"
    assert (destination / "househol.tab").read_text() == "household"
    assert not (destination / "UKDA-9563-tab").exists()


def test_extract_zipped_folder_falls_back_to_flat_zip_layout(tmp_path):
    zip_path = tmp_path / "frs_flat.zip"
    with zipfile.ZipFile(zip_path, "w") as zip_ref:
        zip_ref.writestr("adult.tab", "adult")
        zip_ref.writestr("househol.tab", "household")

    extract_zipped_folder(
        zip_path,
        tab_subdir=CURRENT_FRS_RELEASE.ukds_tab_subdir,
    )

    destination = tmp_path / "frs_flat"
    assert (destination / "adult.tab").read_text() == "adult"
    assert (destination / "househol.tab").read_text() == "household"


def test_extract_zipped_folder_rejects_unsafe_member_paths(tmp_path):
    zip_path = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(zip_path, "w") as zip_ref:
        zip_ref.writestr("../adult.tab", "adult")

    with pytest.raises(ValueError, match="Unsafe path"):
        extract_zipped_folder(zip_path)
