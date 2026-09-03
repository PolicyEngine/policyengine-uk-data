import pytest

from policyengine_uk_data.datasets.frs import (
    create_frs,
    survey_year_from_frs_folder_name,
    validate_frs_survey_year,
)
from policyengine_uk_data.datasets.frs_release import CURRENT_FRS_RELEASE


def test_release_folder_name_encodes_the_survey_year():
    assert survey_year_from_frs_folder_name("frs_2024_25") == 2024
    assert survey_year_from_frs_folder_name("/data/frs_2023_24") == 2023
    assert survey_year_from_frs_folder_name("synthetic_fixture") is None


def test_current_release_folder_matches_its_survey_year():
    assert (
        survey_year_from_frs_folder_name(CURRENT_FRS_RELEASE.name)
        == CURRENT_FRS_RELEASE.survey_year
    )


def test_validate_frs_survey_year_accepts_the_release_survey_year(tmp_path):
    validate_frs_survey_year(tmp_path / "frs_2024_25", 2024)
    validate_frs_survey_year(tmp_path / "synthetic_fixture", 2025)


def test_validate_frs_survey_year_rejects_a_mismatched_year(tmp_path):
    with pytest.raises(ValueError, match="survey year 2024") as excinfo:
        validate_frs_survey_year(tmp_path / "frs_2024_25", 2025)
    assert "year=2025" in str(excinfo.value)
    assert "does not uprate" in str(excinfo.value)


def test_create_frs_rejects_a_year_that_does_not_match_the_release(tmp_path):
    raw_folder = tmp_path / "frs_2024_25"
    raw_folder.mkdir()

    with pytest.raises(ValueError, match="survey year 2024"):
        create_frs(raw_folder, 2025)


def test_create_frs_rejects_a_policy_year_before_the_survey_year(tmp_path):
    raw_folder = tmp_path / "frs_2024_25"
    raw_folder.mkdir()

    with pytest.raises(ValueError, match="precedes survey year"):
        create_frs(raw_folder, 2024, policy_year=2023)
