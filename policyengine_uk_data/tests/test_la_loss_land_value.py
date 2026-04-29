"""Tests for the LA-level main-residence-value column wired into the
local-authority calibration loss matrix.

Two layers:

1. Light-weight checks against the per-LA target dict from la_land.py —
   these run without a Microsimulation and exercise the ordering /
   shape properties the loss-matrix code relies on.
2. Full ``create_local_authority_target_matrix`` build, gated on the
   enhanced FRS fixture so CI environments without the dataset skip
   gracefully.
"""

import numpy as np
import pandas as pd

from policyengine_uk_data.storage import STORAGE_FOLDER
from policyengine_uk_data.targets.sources.la_land import (
    _compute_la_targets,
    load_la_avg_prices,
)
from policyengine_uk_data.targets.sources.local_la_extras import (
    load_household_counts,
    load_tenure_data,
)


LA_CODES = pd.read_csv(STORAGE_FOLDER / "local_authorities_2021.csv")
LA_TARGETS = _compute_la_targets()


# ── Layer 1: per-LA targets line up with the LA code ordering ────────


def test_explicit_targets_cover_english_las():
    """Direct-formula targets are produced for LAs with EHS tenure data
    (England). Other UK countries fall through to the national-share
    fallback in loss.py — same as the existing tenure target."""
    prefixes = {code[0] for code in LA_TARGETS}
    assert prefixes == {"E"}


def test_target_vector_in_la_codes_order_is_finite_positive_where_present():
    """Reindexing by la_codes order yields a clean float vector for
    LAs with a target; LAs missing inputs become NaN (later filled by
    the national-share fallback inside loss.py)."""
    vec = LA_CODES["code"].map(LA_TARGETS).values
    finite = vec[~np.isnan(vec.astype(float))]
    assert len(vec) == 360
    assert (finite > 0).all()


def test_targets_match_observed_product_inline():
    """Per-LA target equals avg_price × ownership_share × n_households —
    the same shape as private rent's ``median_rent × renter_pct × n_hh``.
    """
    prices = load_la_avg_prices().set_index("code")["avg_house_price"]
    tenure = load_tenure_data().set_index("la_code")
    households = load_household_counts().set_index("la_code")["households"]

    for code, target in LA_TARGETS.items():
        if code not in tenure.index or code not in households.index:
            continue
        ownership = (
            tenure.loc[code, "owned_outright_pct"]
            + tenure.loc[code, "owned_mortgage_pct"]
        ) / 100
        expected = prices.loc[code] * ownership * households.loc[code]
        assert abs(target - expected) < 1e-3


# ── Layer 2: full LA loss matrix build ───────────────────────────────


def test_la_loss_matrix_includes_main_residence_value(enhanced_frs):
    """The LA target matrix must expose housing/main_residence_value in
    both matrix (per-household) and y (per-LA) so the calibrator can
    train on it."""
    from policyengine_uk_data.datasets.local_areas.local_authorities.loss import (
        create_local_authority_target_matrix,
    )

    matrix, y, _ = create_local_authority_target_matrix(
        enhanced_frs, time_period=enhanced_frs.time_period
    )

    assert "housing/main_residence_value" in matrix.columns
    assert "housing/main_residence_value" in y.columns


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
    assert len(y["housing/main_residence_value"]) == 360


def test_la_loss_y_matches_observed_product_for_covered_las(enhanced_frs):
    """For LAs with all inputs present, y equals avg_price × ownership × n_households.

    LAs missing inputs use the national-share fallback (covered in
    test_la_loss_y_all_positive)."""
    from policyengine_uk_data.datasets.local_areas.local_authorities.loss import (
        create_local_authority_target_matrix,
    )

    _, y, _ = create_local_authority_target_matrix(
        enhanced_frs, time_period=enhanced_frs.time_period
    )

    expected_by_code = LA_TARGETS
    for i, code in enumerate(LA_CODES["code"].values):
        if code not in expected_by_code:
            continue  # fallback path
        actual = y["housing/main_residence_value"].iloc[i]
        expected = expected_by_code[code]
        assert abs(actual - expected) < 1e-3, (
            f"{code}: y {actual:,.2f} != expected {expected:,.2f}"
        )


def test_la_loss_y_all_positive(enhanced_frs):
    """No LA should have a non-positive main-residence-value target."""
    from policyengine_uk_data.datasets.local_areas.local_authorities.loss import (
        create_local_authority_target_matrix,
    )

    _, y, _ = create_local_authority_target_matrix(
        enhanced_frs, time_period=enhanced_frs.time_period
    )

    assert (y["housing/main_residence_value"] > 0).all()


def test_la_loss_matrix_column_matches_main_residence_value(enhanced_frs):
    """matrix['housing/main_residence_value'] should equal the per-household
    main_residence_value pulled from policyengine-uk for the calibration year."""
    from policyengine_uk import Microsimulation
    from policyengine_uk_data.datasets.local_areas.local_authorities.loss import (
        create_local_authority_target_matrix,
    )

    matrix, _, _ = create_local_authority_target_matrix(
        enhanced_frs, time_period=enhanced_frs.time_period
    )

    sim = Microsimulation(dataset=enhanced_frs)
    sim.default_calculation_period = enhanced_frs.time_period
    expected = sim.calculate("main_residence_value").values

    np.testing.assert_array_equal(
        matrix["housing/main_residence_value"].values, expected
    )
