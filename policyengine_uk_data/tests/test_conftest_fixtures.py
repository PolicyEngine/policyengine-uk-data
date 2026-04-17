"""Tests for the per-year fixture factory in conftest.py (step 6 of #345)."""

from pathlib import Path

import pytest
from policyengine_uk.data import UKSingleYearDataset

from policyengine_uk_data.storage import STORAGE_FOLDER


def _write_tiny_h5(path: Path, year: int) -> None:
    import pandas as pd

    person = pd.DataFrame(
        {
            "person_id": [1001],
            "person_benunit_id": [101],
            "person_household_id": [1],
            "age": [30],
        }
    )
    benunit = pd.DataFrame({"benunit_id": [101]})
    household = pd.DataFrame({"household_id": [1], "household_weight": [1.0]})
    ds = UKSingleYearDataset(
        person=person, benunit=benunit, household=household, fiscal_year=year
    )
    ds.save(path)


def test_factory_loads_requested_year(enhanced_frs_for_year, tmp_path, monkeypatch):
    """A file at ``enhanced_frs_<year>.h5`` is picked up by the factory."""
    monkeypatch.setattr("policyengine_uk_data.tests.conftest.STORAGE_FOLDER", tmp_path)
    _write_tiny_h5(tmp_path / "enhanced_frs_2027.h5", 2027)
    ds = enhanced_frs_for_year(2027)
    assert ds.time_period == "2027"


def test_factory_skips_when_year_missing(enhanced_frs_for_year, tmp_path, monkeypatch):
    """Missing per-year file triggers a pytest skip, not an error."""
    monkeypatch.setattr("policyengine_uk_data.tests.conftest.STORAGE_FOLDER", tmp_path)
    with pytest.raises(pytest.skip.Exception):
        enhanced_frs_for_year(2099)


def test_factory_2023_falls_back_to_legacy_filename(
    enhanced_frs_for_year, tmp_path, monkeypatch
):
    """The base year 2023 accepts the existing ``enhanced_frs_2023_24.h5``."""
    monkeypatch.setattr("policyengine_uk_data.tests.conftest.STORAGE_FOLDER", tmp_path)
    # Only the legacy-named file exists.
    _write_tiny_h5(tmp_path / "enhanced_frs_2023_24.h5", 2023)
    ds = enhanced_frs_for_year(2023)
    assert ds.time_period == "2023"


def test_factory_prefers_new_filename_when_both_exist(
    enhanced_frs_for_year, tmp_path, monkeypatch
):
    """When both files exist, the panel-style filename wins."""
    monkeypatch.setattr("policyengine_uk_data.tests.conftest.STORAGE_FOLDER", tmp_path)
    # Use a marker year mismatch to prove which file was loaded.
    _write_tiny_h5(tmp_path / "enhanced_frs_2023.h5", 2023)
    _write_tiny_h5(tmp_path / "enhanced_frs_2023_24.h5", 9999)
    ds = enhanced_frs_for_year(2023)
    # If the factory picked the legacy file, time_period would be "9999".
    assert ds.time_period == "2023"


def test_factory_accepts_integer_and_string_years(
    enhanced_frs_for_year, tmp_path, monkeypatch
):
    """String years (e.g. from dataset.time_period) are coerced to int."""
    monkeypatch.setattr("policyengine_uk_data.tests.conftest.STORAGE_FOLDER", tmp_path)
    _write_tiny_h5(tmp_path / "enhanced_frs_2026.h5", 2026)
    ds = enhanced_frs_for_year("2026")
    assert ds.time_period == "2026"


def test_existing_enhanced_frs_fixture_still_points_at_legacy_name():
    """Backwards-compat check: don't break what was already working."""
    import inspect

    from policyengine_uk_data.tests import conftest

    source = inspect.getsource(conftest.enhanced_frs)
    assert "enhanced_frs_2023_24.h5" in source


def test_storage_folder_is_not_shadowed():
    """Sanity: the module-level STORAGE_FOLDER export is still the real one."""
    # If someone accidentally replaced the import with a local stub, this
    # catches it before a real run writes into the wrong directory.
    assert str(STORAGE_FOLDER).endswith("storage")
