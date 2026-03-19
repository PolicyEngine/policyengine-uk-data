"""Tests for Phase 6: per-area H5 publishing."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import h5py
import numpy as np
import pandas as pd
import pytest

from policyengine_uk_data.calibration.publish_local_h5s import (
    _get_area_household_indices,
    publish_area_h5,
    publish_local_h5s,
    validate_local_h5s,
)


@pytest.fixture
def mock_dataset():
    """Create a minimal mock dataset with OA geography columns."""
    n_households = 20
    n_persons = 40

    hh = pd.DataFrame(
        {
            "household_id": np.arange(1, n_households + 1),
            "household_weight": np.ones(n_households) * 500.0,
            "region": ["LONDON"] * 10 + ["SCOTLAND"] * 10,
            "constituency_code_oa": (
                ["E14001001"] * 5
                + ["E14001002"] * 5
                + ["S14000001"] * 5
                + ["S14000002"] * 5
            ),
            "la_code_oa": (
                ["E09000001"] * 5
                + ["E09000002"] * 5
                + ["S12000001"] * 5
                + ["S12000002"] * 5
            ),
        }
    )

    person = pd.DataFrame(
        {
            "person_id": np.arange(1, n_persons + 1),
            "person_household_id": np.repeat(np.arange(1, n_households + 1), 2),
            "person_benunit_id": np.arange(1, n_persons + 1),
            "age": np.random.default_rng(42).integers(0, 80, n_persons),
        }
    )

    benunit = pd.DataFrame(
        {
            "benunit_id": np.concatenate(
                [
                    np.arange(1, n_households + 1) * 100 + 1,
                    np.arange(1, 6) * 100 + 2,
                ]
            ),
        }
    )

    dataset = MagicMock()
    dataset.household = hh
    dataset.person = person
    dataset.benunit = benunit
    return dataset


@pytest.fixture
def mock_weights():
    """Sparse weight vector: some households pruned to zero."""
    rng = np.random.default_rng(42)
    weights = rng.uniform(100, 600, size=20)
    # Prune ~30% to zero
    weights[rng.choice(20, size=6, replace=False)] = 0.0
    return weights


class TestGetAreaHouseholdIndices:
    def test_constituency_mapping(self, mock_dataset):
        area_codes = ["E14001001", "E14001002", "S14000001", "S14000002"]
        result = _get_area_household_indices(mock_dataset, "constituency", area_codes)
        assert set(result.keys()) == set(area_codes)
        assert len(result["E14001001"]) == 5
        assert len(result["S14000002"]) == 5

    def test_la_mapping(self, mock_dataset):
        area_codes = ["E09000001", "E09000002", "S12000001", "S12000002"]
        result = _get_area_household_indices(mock_dataset, "la", area_codes)
        assert len(result["E09000001"]) == 5

    def test_unknown_code_returns_empty(self, mock_dataset):
        result = _get_area_household_indices(
            mock_dataset, "constituency", ["X99999999"]
        )
        assert len(result["X99999999"]) == 0

    def test_all_households_covered(self, mock_dataset):
        area_codes = ["E14001001", "E14001002", "S14000001", "S14000002"]
        result = _get_area_household_indices(mock_dataset, "constituency", area_codes)
        all_indices = np.concatenate(list(result.values()))
        assert len(np.unique(all_indices)) == 20


class TestPublishAreaH5:
    def test_writes_valid_h5(self, mock_dataset, mock_weights):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "E14001001.h5"
            hh_indices = np.array([0, 1, 2, 3, 4])

            stat = publish_area_h5(
                dataset=mock_dataset,
                weights=mock_weights,
                area_code="E14001001",
                hh_indices=hh_indices,
                output_path=output_path,
            )

            assert output_path.exists()
            assert stat["code"] == "E14001001"
            assert stat["n_households"] == 5

            with h5py.File(output_path, "r") as f:
                assert "household" in f
                assert "person" in f
                assert "benunit" in f
                assert f.attrs["area_code"] == "E14001001"

    def test_empty_indices_skips(self, mock_dataset, mock_weights):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "X99.h5"
            stat = publish_area_h5(
                dataset=mock_dataset,
                weights=mock_weights,
                area_code="X99",
                hh_indices=np.array([], dtype=np.int64),
                output_path=output_path,
            )
            assert stat["n_active"] == 0
            assert not output_path.exists()

    def test_pruned_households_excluded(self, mock_dataset):
        """Households with zero weight should not appear in H5."""
        weights = np.zeros(20)
        weights[0] = 500.0  # Only first household active

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "E14001001.h5"
            hh_indices = np.array([0, 1, 2, 3, 4])

            stat = publish_area_h5(
                dataset=mock_dataset,
                weights=weights,
                area_code="E14001001",
                hh_indices=hh_indices,
                output_path=output_path,
            )

            assert stat["n_active"] == 1
            with h5py.File(output_path, "r") as f:
                assert f.attrs["n_households"] == 1

    def test_weights_correctly_assigned(self, mock_dataset):
        weights = np.full(20, 250.0)
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "E14001001.h5"
            hh_indices = np.array([0, 1, 2, 3, 4])

            stat = publish_area_h5(
                dataset=mock_dataset,
                weights=weights,
                area_code="E14001001",
                hh_indices=hh_indices,
                output_path=output_path,
            )

            assert stat["total_weight"] == pytest.approx(1250.0)

    def test_persons_match_households(self, mock_dataset, mock_weights):
        """Person table should only contain members of active households."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "E14001001.h5"
            hh_indices = np.array([0, 1, 2, 3, 4])

            publish_area_h5(
                dataset=mock_dataset,
                weights=mock_weights,
                area_code="E14001001",
                hh_indices=hh_indices,
                output_path=output_path,
            )

            with h5py.File(output_path, "r") as f:
                n_hh = f.attrs["n_households"]
                n_persons = f.attrs["n_persons"]
                # Each household has 2 persons in our mock
                assert n_persons <= n_hh * 2


class TestPublishLocalH5s:
    def test_generates_all_area_files(self, mock_dataset, mock_weights):
        """Integration test: full publish cycle with mock weight file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Write mock weight file
            weight_path = tmpdir / "test_weights.h5"
            with h5py.File(weight_path, "w") as f:
                f.create_dataset("2025", data=mock_weights)

            # Write mock area codes CSV
            area_codes = pd.DataFrame(
                {
                    "code": [
                        "E14001001",
                        "E14001002",
                        "S14000001",
                        "S14000002",
                    ],
                    "name": ["A", "B", "C", "D"],
                }
            )
            area_csv = tmpdir / "constituencies_2024.csv"
            area_codes.to_csv(area_csv, index=False)

            output_dir = tmpdir / "output"

            # Patch STORAGE_FOLDER for test
            import policyengine_uk_data.calibration.publish_local_h5s as mod

            original_folder = mod.STORAGE_FOLDER
            mod.STORAGE_FOLDER = tmpdir

            try:
                stats = publish_local_h5s(
                    dataset=mock_dataset,
                    weight_file="test_weights.h5",
                    area_type="constituency",
                    output_dir=output_dir,
                )
            finally:
                mod.STORAGE_FOLDER = original_folder

            # Check all area files created (some may be empty if all pruned)
            active_areas = stats[stats["n_active"] > 0]
            for _, row in active_areas.iterrows():
                assert (output_dir / f"{row['code']}.h5").exists()

            # Summary CSV written
            assert (output_dir / "_summary.csv").exists()

    def test_summary_statistics(self, mock_dataset, mock_weights):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            weight_path = tmpdir / "test_weights.h5"
            with h5py.File(weight_path, "w") as f:
                f.create_dataset("2025", data=mock_weights)

            area_codes = pd.DataFrame(
                {
                    "code": [
                        "E14001001",
                        "E14001002",
                        "S14000001",
                        "S14000002",
                    ],
                    "name": ["A", "B", "C", "D"],
                }
            )
            area_csv = tmpdir / "constituencies_2024.csv"
            area_codes.to_csv(area_csv, index=False)

            import policyengine_uk_data.calibration.publish_local_h5s as mod

            original_folder = mod.STORAGE_FOLDER
            mod.STORAGE_FOLDER = tmpdir

            try:
                stats = publish_local_h5s(
                    dataset=mock_dataset,
                    weight_file="test_weights.h5",
                    area_type="constituency",
                    output_dir=tmpdir / "out",
                )
            finally:
                mod.STORAGE_FOLDER = original_folder

            assert len(stats) == 4
            assert "n_active" in stats.columns
            assert "total_weight" in stats.columns
            # Total active should equal non-zero weights
            assert stats["n_active"].sum() == (mock_weights > 0).sum()


class TestValidateLocalH5s:
    def test_validates_published_files(self, mock_dataset, mock_weights):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            output_dir = tmpdir / "constituency"

            weight_path = tmpdir / "test_weights.h5"
            with h5py.File(weight_path, "w") as f:
                f.create_dataset("2025", data=mock_weights)

            area_codes = pd.DataFrame(
                {
                    "code": [
                        "E14001001",
                        "E14001002",
                        "S14000001",
                        "S14000002",
                    ],
                    "name": ["A", "B", "C", "D"],
                }
            )
            area_csv = tmpdir / "constituencies_2024.csv"
            area_codes.to_csv(area_csv, index=False)

            import policyengine_uk_data.calibration.publish_local_h5s as mod

            original_folder = mod.STORAGE_FOLDER
            mod.STORAGE_FOLDER = tmpdir

            try:
                publish_local_h5s(
                    dataset=mock_dataset,
                    weight_file="test_weights.h5",
                    area_type="constituency",
                    output_dir=output_dir,
                )

                results = validate_local_h5s(
                    area_type="constituency",
                    output_dir=output_dir,
                )
            finally:
                mod.STORAGE_FOLDER = original_folder

            # Active areas should have valid structure
            existing = results[results["exists"]]
            assert existing["valid_structure"].all()

    def test_detects_missing_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            area_codes = pd.DataFrame({"code": ["E14001001"], "name": ["Test"]})
            area_csv = tmpdir / "constituencies_2024.csv"
            area_codes.to_csv(area_csv, index=False)

            import policyengine_uk_data.calibration.publish_local_h5s as mod

            original_folder = mod.STORAGE_FOLDER
            mod.STORAGE_FOLDER = tmpdir

            try:
                results = validate_local_h5s(
                    area_type="constituency",
                    output_dir=tmpdir / "empty",
                )
            finally:
                mod.STORAGE_FOLDER = original_folder

            assert not results["exists"].any()
