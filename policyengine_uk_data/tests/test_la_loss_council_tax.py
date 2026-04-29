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


def test_la_loss_band_y_finite(enhanced_frs):
    """No NaNs / negatives in y. NI LAs are zero (no council tax there);
    other LAs are positive via direct value or national-share fallback."""
    from policyengine_uk_data.datasets.local_areas.local_authorities.loss import (
        create_local_authority_target_matrix,
    )

    _, y, _ = create_local_authority_target_matrix(
        enhanced_frs, time_period=enhanced_frs.time_period
    )

    ni_indices = LA_CODES.index[
        LA_CODES["code"].isin(
            CT_DATA.loc[CT_DATA["country"] == "NORTHERN_IRELAND", "code"]
        )
    ]
    for band in WIRED_BANDS:
        col = f"voa/council_tax/{band}"
        assert np.isfinite(y[col]).all(), f"{col}: NaN/inf in y"
        assert (y[col] >= 0).all(), f"{col}: negative y values"
        # All non-NI LAs have positive targets.
        non_ni = y[col].drop(ni_indices)
        assert (non_ni > 0).all(), f"{col}: non-positive y for non-NI LA"


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


def test_la_loss_band_y_zero_for_ni(enhanced_frs):
    """NI LAs have no council tax (domestic rates instead). Band targets
    must be exactly zero — a positive fallback would be an impossible
    constraint (NI households have council_tax_band == None, so the
    matrix column is zero everywhere) and the optimiser would waste
    loss on a constraint it cannot satisfy."""
    from policyengine_uk_data.datasets.local_areas.local_authorities.loss import (
        create_local_authority_target_matrix,
    )

    _, y, _ = create_local_authority_target_matrix(
        enhanced_frs, time_period=enhanced_frs.time_period
    )

    ni_codes = CT_DATA[CT_DATA["country"] == "NORTHERN_IRELAND"]["code"]
    for ni_la in ni_codes:
        la_index = LA_CODES.index[LA_CODES["code"] == ni_la][0]
        for band in WIRED_BANDS:
            val = float(y[f"voa/council_tax/{band}"].iloc[la_index])
            assert val == 0.0, (
                f"NI LA {ni_la} band {band}: expected 0 (no CT in NI), got {val}"
            )


# ── Council tax £ paid (net of CTR) ─────────────────────────────────


def test_csv_has_net_council_tax_column():
    """The CSV must expose total_council_tax_net so loss.py can wire it."""
    assert "total_council_tax_net" in CT_DATA.columns


def test_net_council_tax_covers_england_and_wales():
    """Direct-formula values are produced for England (MHCLG taxbase × Band D)
    and Wales (Welsh Government Council Tax Income). Scotland and NI fall
    through to the loss.py national-share fallback, same pattern as
    band counts and the existing tenure target."""
    cov = (
        CT_DATA.assign(has_net=CT_DATA["total_council_tax_net"].notna())
        .groupby("country")["has_net"]
        .agg(["sum", "count"])
    )
    assert cov.loc["ENGLAND", "sum"] == cov.loc["ENGLAND", "count"]
    assert cov.loc["WALES", "sum"] == cov.loc["WALES", "count"]
    assert cov.loc["SCOTLAND", "sum"] == 0
    assert cov.loc["NORTHERN_IRELAND", "sum"] == 0


def test_net_council_tax_value_range():
    """Per-LA net council tax should be in £2m–£1.5bn. Lower bound is
    set by Isles of Scilly (~£3m on ~1,100 households); upper bound by
    Birmingham (~£700m). A 1000x outlier — like the IoS fallback leak
    pre-review on #371 — must be caught by bounds, not spotted by eye."""
    covered = CT_DATA["total_council_tax_net"].dropna()
    out_of_range = covered[(covered < 2e6) | (covered > 1.5e9)]
    assert out_of_range.empty, (
        f"Net CT outside [£2m, £1.5bn]: {out_of_range.tolist()[:3]}"
    )


def test_england_net_total_in_range_of_mhclg_summary():
    """Sum of England LA net targets should be within ~5% of MHCLG's
    published England Council Tax Requirement (~£45.86bn for 2026-27;
    we compute from 2025 taxbase × 2026-27 Band D so a small year-mismatch
    gap is expected)."""
    eng_total = CT_DATA.loc[
        CT_DATA["country"] == "ENGLAND", "total_council_tax_net"
    ].sum()
    assert 4.3e10 < eng_total < 5.0e10, (
        f"England net total £{eng_total / 1e9:.2f}bn outside [£43bn, £50bn]"
    )


def test_la_loss_matrix_includes_council_tax_net(enhanced_frs):
    """matrix and y must expose housing/council_tax_net so the
    calibrator trains on the net £-amount target alongside band counts."""
    from policyengine_uk_data.datasets.local_areas.local_authorities.loss import (
        create_local_authority_target_matrix,
    )

    matrix, y, _ = create_local_authority_target_matrix(
        enhanced_frs, time_period=enhanced_frs.time_period
    )
    assert "housing/council_tax_net" in matrix.columns
    assert "housing/council_tax_net" in y.columns


def test_la_loss_council_tax_net_matrix_uses_net_variable(enhanced_frs):
    """Matrix col must be council_tax_less_benefit (net of CTR), so both
    sides of the calibration constraint are net per the 28 Apr standup
    decision on FRS-net-of-CTR alignment."""
    from policyengine_uk import Microsimulation
    from policyengine_uk_data.datasets.local_areas.local_authorities.loss import (
        create_local_authority_target_matrix,
    )

    matrix, _, _ = create_local_authority_target_matrix(
        enhanced_frs, time_period=enhanced_frs.time_period
    )

    sim = Microsimulation(dataset=enhanced_frs)
    sim.default_calculation_period = enhanced_frs.time_period
    expected = sim.calculate("council_tax_less_benefit").values
    np.testing.assert_array_equal(matrix["housing/council_tax_net"].values, expected)


def test_la_loss_council_tax_net_y_finite(enhanced_frs):
    """y has no NaN/inf. NI LAs are zero (no council tax there);
    Scotland uses fallback; England/Wales use direct values."""
    from policyengine_uk_data.datasets.local_areas.local_authorities.loss import (
        create_local_authority_target_matrix,
    )

    _, y, _ = create_local_authority_target_matrix(
        enhanced_frs, time_period=enhanced_frs.time_period
    )
    col = y["housing/council_tax_net"]
    assert np.isfinite(col).all()
    assert (col >= 0).all()

    ni_indices = LA_CODES.index[
        LA_CODES["code"].isin(
            CT_DATA.loc[CT_DATA["country"] == "NORTHERN_IRELAND", "code"]
        )
    ]
    assert (col.iloc[ni_indices] == 0).all(), (
        "NI LAs should have y=0 for housing/council_tax_net "
        "(NI uses domestic rates, not council tax)"
    )
    assert (col.drop(ni_indices) > 0).all(), (
        "non-NI LAs should have positive y (direct value or fallback)"
    )


def test_la_loss_council_tax_net_y_matches_csv_for_english_la(enhanced_frs):
    """For a covered (English) LA, y must equal the CSV value verbatim
    rather than the national-share fallback."""
    from policyengine_uk_data.datasets.local_areas.local_authorities.loss import (
        create_local_authority_target_matrix,
    )

    _, y, _ = create_local_authority_target_matrix(
        enhanced_frs, time_period=enhanced_frs.time_period
    )

    target_code = "E06000001"  # Hartlepool
    la_index = LA_CODES.index[LA_CODES["code"] == target_code][0]
    expected = float(
        CT_DATA.loc[CT_DATA["code"] == target_code, "total_council_tax_net"].iloc[0]
    )
    actual = float(y["housing/council_tax_net"].iloc[la_index])
    assert actual == pytest.approx(expected, rel=1e-6)


def test_la_loss_council_tax_net_y_uses_fallback_for_scotland(enhanced_frs):
    """Scottish LAs have no published net CT in the CSV; the fallback
    must produce a positive value."""
    from policyengine_uk_data.datasets.local_areas.local_authorities.loss import (
        create_local_authority_target_matrix,
    )

    _, y, _ = create_local_authority_target_matrix(
        enhanced_frs, time_period=enhanced_frs.time_period
    )
    scotland_code = CT_DATA[CT_DATA["country"] == "SCOTLAND"]["code"].iloc[0]
    la_index = LA_CODES.index[LA_CODES["code"] == scotland_code][0]
    assert float(y["housing/council_tax_net"].iloc[la_index]) > 0


def test_la_loss_english_council_tax_net_in_reach_of_initial_weights(
    enhanced_frs,
):
    """Sum of English LA net council-tax targets should be in the same
    order of magnitude (0.3x–3x) as the implied initial weighted English
    council_tax_less_benefit total — so the calibrator can reach the
    target via reweighting rather than 100x weight inflation."""
    from policyengine_uk import Microsimulation
    from policyengine_uk_data.datasets.local_areas.local_authorities.loss import (
        create_local_authority_target_matrix,
    )

    _, y, _ = create_local_authority_target_matrix(
        enhanced_frs, time_period=enhanced_frs.time_period
    )

    sim = Microsimulation(dataset=enhanced_frs)
    weights = sim.calculate("household_weight", 2025).values
    ct_net = sim.calculate("council_tax_less_benefit", enhanced_frs.time_period).values
    country = sim.calculate("country", enhanced_frs.time_period).values
    england_initial = (
        weights[country == "ENGLAND"] * ct_net[country == "ENGLAND"]
    ).sum()

    english_indices = [
        i for i, c in enumerate(LA_CODES["code"].values) if c.startswith("E0")
    ]
    english_target_sum = y["housing/council_tax_net"].iloc[english_indices].sum()

    if england_initial > 0:
        ratio = english_target_sum / england_initial
        assert 0.3 < ratio < 3.0, (
            f"England target sum (£{english_target_sum / 1e9:.1f}bn) / "
            f"initial weighted England net CT (£{england_initial / 1e9:.1f}bn) "
            f"= {ratio:.2f}; calibration target may be hard to reach"
        )
