"""Tests for property income calibration targets (#230).

Verifies that property income targets reflect the more comprehensive
HMRC Property Rental Income Statistics rather than SPI alone.
"""

from policyengine_uk_data.targets import get_all_targets


def test_property_income_targets_scaled():
    """Property income targets should be ~1.9x the raw SPI values.

    Raw SPI 2022-23 total is ~£27bn. After scaling, targets for the
    base year should be ~£52bn (matching HMRC rental income stats).
    """
    targets = get_all_targets(year=2023)
    total = sum(
        t.values[2023]
        for t in targets
        if "property_income" in t.name and "count" not in t.name and 2023 in t.values
    )
    # Raw SPI gives ~£27bn, scaled by 1.9x should give ~£52bn
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
    # With 1.9x scaling, 2025 projection should be ~£67bn
    assert 50e9 < total < 90e9, (
        f"Property income target total for 2025 is £{total / 1e9:.1f}bn, "
        "outside expected £50-90bn range."
    )
