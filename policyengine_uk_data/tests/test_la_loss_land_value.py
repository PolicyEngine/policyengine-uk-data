"""Tests for the LA-level household-land-value column wired into the
local-authority calibration loss matrix.

Two layers:

1. Light-weight checks against the per-LA target dict from la_land.py —
   these run without a Microsimulation and exercise the ordering /
   summation properties the loss-matrix code relies on.
2. Full ``create_local_authority_target_matrix`` build, gated on the
   enhanced FRS fixture so CI environments without the dataset skip
   gracefully.
"""

import numpy as np
import pandas as pd
import pytest

from policyengine_uk_data.storage import STORAGE_FOLDER
from policyengine_uk_data.targets.sources._land import HOUSEHOLD_LAND_VALUES
from policyengine_uk_data.targets.sources.la_land import _compute_la_targets


LA_CODES = pd.read_csv(STORAGE_FOLDER / "local_authorities_2021.csv")
LA_TARGETS = _compute_la_targets()


# ── Layer 1: per-LA targets line up with the LA code ordering ────────


def test_targets_cover_every_la_code():
    """Every code in local_authorities_2021.csv has an LA land target."""
    missing = set(LA_CODES["code"]) - set(LA_TARGETS)
    assert not missing, f"LA codes missing land targets: {sorted(missing)[:5]}"


def test_target_vector_in_la_codes_order_is_finite_positive():
    """Reindexing by la_codes order yields a clean float vector."""
    year = 2025
    vec = (
        LA_CODES["code"]
        .map({code: values[year] for code, values in LA_TARGETS.items()})
        .values
    )
    assert len(vec) == 360
    assert np.isfinite(vec).all()
    assert (vec > 0).all()


def test_target_vector_sums_to_national_household_land():
    """Sum of the 360 LA targets equals the ONS national figure for that year."""
    for year in (2024, 2025, 2026):
        vec = (
            LA_CODES["code"]
            .map({code: values[year] for code, values in LA_TARGETS.items()})
            .values
        )
        rel_error = abs(vec.sum() / HOUSEHOLD_LAND_VALUES[year] - 1)
        assert rel_error < 1e-6, (
            f"{year}: sum £{vec.sum() / 1e12:.3f}tn != "
            f"national £{HOUSEHOLD_LAND_VALUES[year] / 1e12:.3f}tn"
        )


# ── Layer 2: full LA loss matrix build ───────────────────────────────


def test_la_loss_matrix_includes_household_land_value(enhanced_frs):
    """The LA target matrix must expose ons/household_land_value in both
    matrix (per-household) and y (per-LA) so the calibrator can train on it.
    """
    from policyengine_uk_data.datasets.local_areas.local_authorities.loss import (
        create_local_authority_target_matrix,
    )

    matrix, y, _ = create_local_authority_target_matrix(
        enhanced_frs, time_period=enhanced_frs.time_period
    )

    assert "ons/household_land_value" in matrix.columns
    assert "ons/household_land_value" in y.columns


def test_la_loss_y_vector_length_360(enhanced_frs):
    """y has one entry per LA and matches local_authorities_2021.csv ordering
    by length."""
    from policyengine_uk_data.datasets.local_areas.local_authorities.loss import (
        create_local_authority_target_matrix,
    )

    _, y, _ = create_local_authority_target_matrix(
        enhanced_frs, time_period=enhanced_frs.time_period
    )

    assert len(y) == 360
    assert len(y["ons/household_land_value"]) == 360


def test_la_loss_y_sums_to_national_for_calibration_year(enhanced_frs):
    """Sum of LA-level y values equals the ONS national household-land total
    for the calibration year (within float tolerance)."""
    from policyengine_uk_data.datasets.local_areas.local_authorities.loss import (
        create_local_authority_target_matrix,
    )

    year = int(enhanced_frs.time_period)
    fallback = max(HOUSEHOLD_LAND_VALUES)
    expected = HOUSEHOLD_LAND_VALUES.get(year, HOUSEHOLD_LAND_VALUES[fallback])

    _, y, _ = create_local_authority_target_matrix(
        enhanced_frs, time_period=enhanced_frs.time_period
    )

    rel_error = abs(y["ons/household_land_value"].sum() / expected - 1)
    assert rel_error < 1e-6


def test_la_loss_y_ordering_matches_la_codes(enhanced_frs):
    """y["ons/household_land_value"] must be ordered by local_authorities_2021.csv,
    so the country mask and the targets refer to the same LAs at each index."""
    from policyengine_uk_data.datasets.local_areas.local_authorities.loss import (
        create_local_authority_target_matrix,
    )

    year = int(enhanced_frs.time_period)
    fallback = max(HOUSEHOLD_LAND_VALUES)
    land_year = year if year in HOUSEHOLD_LAND_VALUES else fallback
    expected = (
        LA_CODES["code"]
        .map({code: values[land_year] for code, values in LA_TARGETS.items()})
        .values
    )

    _, y, _ = create_local_authority_target_matrix(
        enhanced_frs, time_period=enhanced_frs.time_period
    )

    np.testing.assert_array_equal(y["ons/household_land_value"].values, expected)


def test_la_loss_y_all_positive(enhanced_frs):
    """No LA should have a non-positive household-land target."""
    from policyengine_uk_data.datasets.local_areas.local_authorities.loss import (
        create_local_authority_target_matrix,
    )

    _, y, _ = create_local_authority_target_matrix(
        enhanced_frs, time_period=enhanced_frs.time_period
    )

    assert (y["ons/household_land_value"] > 0).all()


def test_la_loss_matrix_column_matches_household_land_value(enhanced_frs):
    """matrix['ons/household_land_value'] should equal the per-household
    household_land_value pulled from policyengine-uk for the calibration year."""
    from policyengine_uk import Microsimulation
    from policyengine_uk_data.datasets.local_areas.local_authorities.loss import (
        create_local_authority_target_matrix,
    )

    matrix, _, _ = create_local_authority_target_matrix(
        enhanced_frs, time_period=enhanced_frs.time_period
    )

    sim = Microsimulation(dataset=enhanced_frs)
    sim.default_calculation_period = enhanced_frs.time_period
    expected = sim.calculate("household_land_value").values

    np.testing.assert_array_equal(matrix["ons/household_land_value"].values, expected)
