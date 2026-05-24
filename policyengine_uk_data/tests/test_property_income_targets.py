"""Tests for property income calibration targets (#230).

Verifies that property income targets reflect the more comprehensive
HMRC Property Rental Income Statistics rather than SPI alone.
"""

from policyengine_uk_data.targets import get_all_targets


def test_property_income_targets_scaled():
    """Property income targets should be ~1.9x the raw SPI values.

    Raw SPI 2023-24 total is scaled up to better match HMRC rental
    income statistics, which cover more landlords than SPI.
    """
    base_year = 2024
    targets = get_all_targets(year=base_year)
    total = sum(
        t.values[base_year]
        for t in targets
        if "property_income" in t.name
        and "count" not in t.name
        and base_year in t.values
    )
    # Raw SPI gives roughly half of all landlord income; scaling should
    # leave the current base-year target in this broad administrative range.
    assert total > 45e9, (
        f"Property income target total £{total / 1e9:.1f}bn is below £45bn. "
        "Scaling factor may not be applied."
    )
    assert total < 60e9, (
        f"Property income target total £{total / 1e9:.1f}bn exceeds £60bn. "
        "Possible double-counting or excessive scaling."
    )


def test_no_aggregate_row_double_counting():
    """Projected years should not have a 12_570_to_inf aggregate band.

    The incomes_projection.csv contains aggregate rows that would
    double-count income if included alongside per-band rows.
    """
    targets = get_all_targets(year=2025)
    for t in targets:
        if "property_income" in t.name and "12_570_to_inf" in t.name:
            assert False, (
                f"Found aggregate target {t.name} — "
                "this causes double-counting with per-band targets."
            )


def test_projected_property_income_reasonable():
    """Property income targets for 2025 should be in a reasonable range."""
    targets = get_all_targets(year=2025)
    total = sum(
        t.values[2025]
        for t in targets
        if "property_income" in t.name and "count" not in t.name and 2025 in t.values
    )
    # With 1.9x scaling, 2025 projection should be ~£57bn
    assert 50e9 < total < 90e9, (
        f"Property income target total for 2025 is £{total / 1e9:.1f}bn, "
        "outside expected £50-90bn range."
    )


def test_projected_targets_keep_top_open_ended_band():
    """Projection target parsing should keep the true top income band."""
    targets = get_all_targets(year=2025)
    assert any(
        t.name == "hmrc/property_income_income_band_1_000_000_to_inf" for t in targets
    )
