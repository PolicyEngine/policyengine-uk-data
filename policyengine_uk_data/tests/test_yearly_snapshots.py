"""Tests for the per-year snapshot helper."""

from pathlib import Path

import pandas as pd
import pytest
from policyengine_uk.data import UKSingleYearDataset

from policyengine_uk_data.datasets import yearly_snapshots
from policyengine_uk_data.datasets.yearly_snapshots import (
    create_yearly_snapshots,
)


def _tiny_base(year: int = 2023) -> UKSingleYearDataset:
    """Build the smallest dataset the helper can legitimately process.

    Two households, two benefit units, three persons, one scalable variable
    (``employment_income``) plus the three ID columns on the appropriate
    tables.
    """
    household = pd.DataFrame(
        {
            "household_id": [1, 2],
            "household_weight": [1.0, 1.0],
        }
    )
    benunit = pd.DataFrame(
        {
            "benunit_id": [101, 201],
        }
    )
    person = pd.DataFrame(
        {
            "person_id": [1001, 1002, 2001],
            "person_benunit_id": [101, 101, 201],
            "person_household_id": [1, 1, 2],
            "employment_income": [10_000.0, 0.0, 25_000.0],
        }
    )
    return UKSingleYearDataset(
        person=person,
        benunit=benunit,
        household=household,
        fiscal_year=year,
    )


def _identity_uprate(dataset: UKSingleYearDataset, target_year: int):
    """Stand-in for ``uprate_dataset`` that just retags the year.

    Using the real uprating function would require the ``uprating_factors.csv``
    file to cover every variable in the test dataset; for these tests we only
    care about the orchestration — not the scalar values — so we skip the
    factor lookup entirely.
    """
    copy = dataset.copy()
    copy.time_period = target_year
    return copy


@pytest.fixture
def patched_uprate(monkeypatch):
    monkeypatch.setattr(yearly_snapshots, "uprate_dataset", _identity_uprate)


def test_creates_one_file_per_year(tmp_path: Path, patched_uprate):
    base = _tiny_base(year=2023)
    written = create_yearly_snapshots(
        base, years=[2023, 2024, 2025], output_dir=tmp_path
    )

    assert [p.name for p in written] == [
        "enhanced_frs_2023.h5",
        "enhanced_frs_2024.h5",
        "enhanced_frs_2025.h5",
    ]
    for path in written:
        assert path.exists()


def test_each_saved_file_carries_its_target_year(tmp_path: Path, patched_uprate):
    base = _tiny_base(year=2023)
    written = create_yearly_snapshots(base, years=[2023, 2030], output_dir=tmp_path)

    loaded_years = [UKSingleYearDataset(file_path=str(p)).time_period for p in written]
    assert loaded_years == ["2023", "2030"]


def test_panel_ids_preserved_across_snapshots(tmp_path: Path, patched_uprate):
    """The whole point of step 2: same IDs appear in every year's file."""
    base = _tiny_base(year=2023)
    create_yearly_snapshots(base, years=[2023, 2024, 2025], output_dir=tmp_path)

    base_ids = sorted(base.person["person_id"].tolist())
    for year in (2023, 2024, 2025):
        loaded = UKSingleYearDataset(
            file_path=str(tmp_path / f"enhanced_frs_{year}.h5")
        )
        assert sorted(loaded.person["person_id"].tolist()) == base_ids


def test_base_year_snapshot_is_a_straight_copy(tmp_path: Path, patched_uprate):
    """Asking for the base year should not go through uprating."""
    base = _tiny_base(year=2023)
    create_yearly_snapshots(base, years=[2023], output_dir=tmp_path)

    loaded = UKSingleYearDataset(file_path=str(tmp_path / "enhanced_frs_2023.h5"))
    # Employment income values must be untouched.
    assert loaded.person["employment_income"].tolist() == [
        10_000.0,
        0.0,
        25_000.0,
    ]


def test_missing_output_directory_raises(patched_uprate):
    base = _tiny_base(year=2023)
    with pytest.raises(FileNotFoundError):
        create_yearly_snapshots(
            base, years=[2023], output_dir="/tmp/does-not-exist-xyz-123"
        )


def test_panel_id_drift_is_detected(tmp_path: Path, monkeypatch):
    """If a future uprating step drops a row, step 2 must fail loudly."""

    def dropping_uprate(dataset, target_year):
        copy = dataset.copy()
        copy.time_period = target_year
        copy.person = copy.person.iloc[:-1].copy()
        return copy

    monkeypatch.setattr(yearly_snapshots, "uprate_dataset", dropping_uprate)

    base = _tiny_base(year=2023)
    with pytest.raises(AssertionError, match="person"):
        create_yearly_snapshots(base, years=[2024], output_dir=tmp_path)


def test_custom_filename_template(tmp_path: Path, patched_uprate):
    base = _tiny_base(year=2023)
    written = create_yearly_snapshots(
        base,
        years=[2024],
        output_dir=tmp_path,
        filename_template="panel_{year}.h5",
    )
    assert written[0].name == "panel_2024.h5"
    assert written[0].exists()
