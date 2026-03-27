"""Tests for the Phase 4 sparse matrix builder.

Tests the assignment matrix, metric computation, and sparse matrix
construction without requiring a full dataset or Microsimulation.
"""

import numpy as np
import pandas as pd
import pytest
from scipy import sparse as sp
from unittest.mock import MagicMock, patch

from policyengine_uk_data.calibration.matrix_builder import (
    build_assignment_matrix,
    _geo_column,
)


class MockDataset:
    """Minimal dataset stub with household geography columns."""

    def __init__(self, n_households=100, area_codes=None, area_type="constituency"):
        rng = np.random.default_rng(42)
        geo_col = _geo_column(area_type)

        if area_codes is None:
            area_codes = [f"E14{i:06d}" for i in range(10)]

        # Assign each household to a random area (some unassigned)
        assignments = []
        for _ in range(n_households):
            if rng.random() < 0.9:
                assignments.append(rng.choice(area_codes))
            else:
                assignments.append("")  # Unassigned (e.g. NI)

        self.household = pd.DataFrame(
            {
                "household_id": np.arange(n_households),
                "household_weight": np.ones(n_households) * 10.0,
                geo_col: assignments,
            }
        )
        self.time_period = 2025
        self._area_codes = area_codes


class TestBuildAssignmentMatrix:
    def test_shape(self):
        codes = [f"E14{i:06d}" for i in range(10)]
        ds = MockDataset(n_households=50, area_codes=codes)
        A = build_assignment_matrix(ds, "constituency", pd.Series(codes))
        assert A.shape == (10, 50)

    def test_sparse_type(self):
        codes = [f"E14{i:06d}" for i in range(5)]
        ds = MockDataset(n_households=20, area_codes=codes)
        A = build_assignment_matrix(ds, "constituency", pd.Series(codes))
        assert sp.issparse(A)

    def test_each_household_in_at_most_one_area(self):
        codes = [f"E14{i:06d}" for i in range(10)]
        ds = MockDataset(n_households=100, area_codes=codes)
        A = build_assignment_matrix(ds, "constituency", pd.Series(codes))
        # Each column should sum to 0 or 1
        col_sums = np.array(A.sum(axis=0)).flatten()
        assert np.all((col_sums == 0) | (col_sums == 1))

    def test_unassigned_households_have_zero_columns(self):
        codes = [f"E14{i:06d}" for i in range(5)]
        ds = MockDataset(n_households=50, area_codes=codes)
        A = build_assignment_matrix(ds, "constituency", pd.Series(codes))
        col_sums = np.array(A.sum(axis=0)).flatten()
        # Some households should be unassigned (empty string)
        n_unassigned = (col_sums == 0).sum()
        assert n_unassigned > 0

    def test_binary_values(self):
        codes = [f"E14{i:06d}" for i in range(5)]
        ds = MockDataset(n_households=30, area_codes=codes)
        A = build_assignment_matrix(ds, "constituency", pd.Series(codes))
        assert np.all(A.data == 1.0)

    def test_unknown_codes_ignored(self):
        codes = [f"E14{i:06d}" for i in range(5)]
        ds = MockDataset(n_households=30, area_codes=codes)
        # Query with different area codes
        other_codes = pd.Series([f"E06{i:06d}" for i in range(3)])
        A = build_assignment_matrix(ds, "constituency", other_codes)
        assert A.shape == (3, 30)
        assert A.nnz == 0  # No matches

    def test_la_area_type(self):
        codes = [f"E06{i:06d}" for i in range(5)]
        ds = MockDataset(n_households=20, area_codes=codes, area_type="la")
        A = build_assignment_matrix(ds, "la", pd.Series(codes))
        assert A.shape == (5, 20)
        assert A.nnz > 0

    def test_all_assigned_sums_to_n_areas(self):
        """If every household is assigned, row sums partition all households."""
        codes = ["AREA_A", "AREA_B"]
        geo_col = _geo_column("constituency")
        ds = MagicMock()
        ds.household = pd.DataFrame(
            {
                geo_col: ["AREA_A"] * 5 + ["AREA_B"] * 3,
            }
        )
        A = build_assignment_matrix(ds, "constituency", pd.Series(codes))
        assert A.shape == (2, 8)
        row_sums = np.array(A.sum(axis=1)).flatten()
        assert row_sums[0] == 5
        assert row_sums[1] == 3
        assert A.sum() == 8


class TestGeoColumn:
    def test_constituency(self):
        assert _geo_column("constituency") == "constituency_code_oa"

    def test_la(self):
        assert _geo_column("la") == "la_code_oa"
