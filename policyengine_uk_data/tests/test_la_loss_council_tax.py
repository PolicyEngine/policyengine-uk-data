"""Tests for the LA-level council-tax band-count columns wired into the
local-authority calibration loss matrix.

Layered like test_la_loss_land_value.py:

1. Light checks against la_council_tax.csv — exercise the data shape
   the loss-matrix code relies on without needing a Microsimulation.
2. Full ``create_local_authority_target_matrix`` build, gated on the
   enhanced FRS fixture so CI environments without the dataset skip
   gracefully.

Note: only bands A-H are wired. Band I is Wales-only and mostly null,
and the Band D ``£`` amount is a per-rate quantity that does not fit
the linear matrix-times-weights aggregation — both are deliberately
out of scope for this PR.
"""

import numpy as np
import pandas as pd
import pytest

from policyengine_uk_data.storage import STORAGE_FOLDER


CT_DATA = pd.read_csv(STORAGE_FOLDER / "la_council_tax.csv")
LA_CODES = pd.read_csv(STORAGE_FOLDER / "local_authorities_2021.csv")
WIRED_BANDS = list("ABCDEFGH")


# ── Layer 1: CSV shape and joinability ───────────────────────────────


def test_csv_has_a_row_for_every_la_code():
    """The CSV must cover every LA in local_authorities_2021.csv so the
    left-merge inside loss.py never produces NaN-only rows in the
    has_count branch."""
    missing = set(LA_CODES["code"]) - set(CT_DATA["code"])
    assert not missing, (
        f"LA codes missing from la_council_tax.csv: {sorted(missing)[:5]}"
    )


def test_band_count_columns_exist_for_every_wired_band():
    """Every wired band needs a count_band_{X} column, otherwise the loss
    matrix loop will KeyError on missing CSV columns."""
    for band in WIRED_BANDS:
        assert f"count_band_{band}" in CT_DATA.columns


def test_england_and_wales_have_band_a_to_h_populated():
    """E/W rows should have non-null counts for A-H. If the CSV regresses
    to NaN there, the loss matrix will silently fall back to the
    national-share estimate and the calibrator loses its real signal."""
    ew = CT_DATA[CT_DATA["country"].isin(["ENGLAND", "WALES"])]
    for band in WIRED_BANDS:
        non_null = ew[f"count_band_{band}"].notna().sum()
        # City of London suppresses Band A; allow up to 5 missing per band.
        assert non_null >= len(ew) - 5, (
            f"Band {band}: only {non_null}/{len(ew)} E/W LAs have a count"
        )


def test_scotland_band_counts_are_null_as_documented():
    """Scotland VOA band counts are absent — they should consistently be
    NaN so the loss matrix routes them through the fallback."""
    scotland = CT_DATA[CT_DATA["country"] == "SCOTLAND"]
    for band in WIRED_BANDS:
        assert scotland[f"count_band_{band}"].isna().all(), (
            f"Band {band}: Scotland rows unexpectedly populated"
        )


def test_ni_council_tax_disabled():
    """NI uses domestic rates, not council tax. has_council_tax must be
    False for every NI row, otherwise downstream code may try to read
    targets that do not exist."""
    ni = CT_DATA[CT_DATA["country"] == "NORTHERN_IRELAND"]
    assert not ni.empty
    assert (ni["has_council_tax"] == False).all()  # noqa: E712


# ── Layer 2: full LA loss matrix build ───────────────────────────────


def test_la_loss_matrix_includes_all_wired_band_columns(enhanced_frs):
    """matrix and y must expose voa/council_tax/{A..H} for the calibrator
    to train on the new targets."""
    from policyengine_uk_data.datasets.local_areas.local_authorities.loss import (
        create_local_authority_target_matrix,
    )

    matrix, y, _ = create_local_authority_target_matrix(
        enhanced_frs, time_period=enhanced_frs.time_period
    )
    for band in WIRED_BANDS:
        col = f"voa/council_tax/{band}"
        assert col in matrix.columns, f"missing matrix column {col}"
        assert col in y.columns, f"missing y column {col}"


def test_la_loss_band_y_vectors_length_360(enhanced_frs):
    from policyengine_uk_data.datasets.local_areas.local_authorities.loss import (
        create_local_authority_target_matrix,
    )

    _, y, _ = create_local_authority_target_matrix(
        enhanced_frs, time_period=enhanced_frs.time_period
    )
    for band in WIRED_BANDS:
        assert len(y[f"voa/council_tax/{band}"]) == 360


def test_la_loss_band_y_finite_and_positive(enhanced_frs):
    """No NaNs / negatives in y. Scotland and NI cells must be filled by
    the fallback (national_band_count × la_household_share)."""
    from policyengine_uk_data.datasets.local_areas.local_authorities.loss import (
        create_local_authority_target_matrix,
    )

    _, y, _ = create_local_authority_target_matrix(
        enhanced_frs, time_period=enhanced_frs.time_period
    )
    for band in WIRED_BANDS:
        col = f"voa/council_tax/{band}"
        assert np.isfinite(y[col]).all(), f"{col}: NaN/inf in y"
        assert (y[col] > 0).all(), f"{col}: non-positive y values"


def test_la_loss_band_matrix_columns_are_indicators(enhanced_frs):
    """matrix entries are 0 or 1 (the household either is or isn't in band X)."""
    from policyengine_uk_data.datasets.local_areas.local_authorities.loss import (
        create_local_authority_target_matrix,
    )

    matrix, _, _ = create_local_authority_target_matrix(
        enhanced_frs, time_period=enhanced_frs.time_period
    )
    for band in WIRED_BANDS:
        col = f"voa/council_tax/{band}"
        unique = set(np.unique(matrix[col].values))
        assert unique <= {0.0, 1.0}, f"{col}: non-indicator values {unique}"


def test_la_loss_band_matrix_rows_sum_to_at_most_one(enhanced_frs):
    """Each household sits in at most one band. Summing the wired band
    columns across each household should be 0 or 1."""
    from policyengine_uk_data.datasets.local_areas.local_authorities.loss import (
        create_local_authority_target_matrix,
    )

    matrix, _, _ = create_local_authority_target_matrix(
        enhanced_frs, time_period=enhanced_frs.time_period
    )
    band_sum = sum(matrix[f"voa/council_tax/{b}"] for b in WIRED_BANDS)
    assert ((band_sum == 0) | (band_sum == 1)).all()


def test_la_loss_band_y_matches_csv_for_english_la(enhanced_frs):
    """For an English LA with VOA data, y[band] must be the CSV value
    verbatim — not the national-share fallback."""
    from policyengine_uk_data.datasets.local_areas.local_authorities.loss import (
        create_local_authority_target_matrix,
    )

    _, y, _ = create_local_authority_target_matrix(
        enhanced_frs, time_period=enhanced_frs.time_period
    )

    # Hartlepool — first LA in local_authorities_2021.csv, VOA-covered.
    target_code = "E06000001"
    la_index = LA_CODES.index[LA_CODES["code"] == target_code][0]
    ct_row = CT_DATA[CT_DATA["code"] == target_code].iloc[0]
    for band in WIRED_BANDS:
        expected = float(ct_row[f"count_band_{band}"])
        actual = float(y[f"voa/council_tax/{band}"].iloc[la_index])
        assert actual == pytest.approx(expected, rel=0, abs=0.5), (
            f"{target_code} band {band}: y={actual}, CSV={expected}"
        )


def test_la_loss_band_y_uses_fallback_for_scotland(enhanced_frs):
    """Scottish LAs have no VOA band counts; y must use the
    national_band_count × la_household_share fallback rather than NaN
    or 0."""
    from policyengine_uk_data.datasets.local_areas.local_authorities.loss import (
        create_local_authority_target_matrix,
    )

    _, y, _ = create_local_authority_target_matrix(
        enhanced_frs, time_period=enhanced_frs.time_period
    )

    scotland_la = CT_DATA[CT_DATA["country"] == "SCOTLAND"]["code"].iloc[0]
    la_index = LA_CODES.index[LA_CODES["code"] == scotland_la][0]
    for band in WIRED_BANDS:
        val = float(y[f"voa/council_tax/{band}"].iloc[la_index])
        assert val > 0, (
            f"Scotland LA {scotland_la} band {band}: fallback produced {val}"
        )


def test_la_loss_band_y_uses_fallback_for_ni(enhanced_frs):
    """NI LAs have no council tax data at all; y must still be positive
    via the fallback (the country mask zeroes the prediction anyway)."""
    from policyengine_uk_data.datasets.local_areas.local_authorities.loss import (
        create_local_authority_target_matrix,
    )

    _, y, _ = create_local_authority_target_matrix(
        enhanced_frs, time_period=enhanced_frs.time_period
    )

    ni_la = CT_DATA[CT_DATA["country"] == "NORTHERN_IRELAND"]["code"].iloc[0]
    la_index = LA_CODES.index[LA_CODES["code"] == ni_la][0]
    for band in WIRED_BANDS:
        val = float(y[f"voa/council_tax/{band}"].iloc[la_index])
        assert val > 0, f"NI LA {ni_la} band {band}: fallback produced {val}"
