"""Tests for the targets registry.

Verifies that:
1. All source modules load without error
2. No duplicate target names
3. Core targets exist for key years
4. Target values match the current system's hardcoded values
"""

import pytest
from policyengine_uk_data.targets import get_all_targets, Target


def test_registry_loads():
    """All source modules should load and return targets."""
    targets = get_all_targets()
    assert len(targets) > 0, "Registry returned no targets"


def test_no_duplicate_names():
    """Target names should be unique across all sources."""
    targets = get_all_targets()
    names = [t.name for t in targets]
    duplicates = [n for n in names if names.count(n) > 1]
    assert len(duplicates) == 0, f"Duplicate target names: {set(duplicates)}"


def test_obr_income_tax_exists():
    """OBR income tax target should exist for 2025."""
    targets = get_all_targets(year=2025)
    names = {t.name for t in targets}
    assert "obr/income_tax" in names


def test_obr_income_tax_value():
    """OBR income tax for 2025 should be ~£329bn (Table 3.4 accrued basis)."""
    targets = get_all_targets(year=2025)
    it = next(t for t in targets if t.name == "obr/income_tax")
    # Table 3.4 D6 = 328.96bn for FY 2025-26 → calendar 2025
    assert abs(it.values[2025] - 329e9) < 1e9


def test_ons_uk_population_exists():
    """UK population target should exist."""
    targets = get_all_targets(year=2025)
    names = {t.name for t in targets}
    assert "ons/uk_population" in names


def test_hmrc_spi_targets_exist():
    """HMRC SPI income band targets should exist."""
    targets = get_all_targets(year=2025)
    spi_targets = [t for t in targets if t.source == "hmrc_spi"]
    # 13 bands × 6 income types × 2 (count + amount) = 156 per year
    assert len(spi_targets) >= 100, (
        f"Expected 100+ SPI targets, got {len(spi_targets)}"
    )


def test_dwp_pip_targets():
    """DWP PIP targets should exist."""
    targets = get_all_targets(year=2025)
    names = {t.name for t in targets}
    assert "dwp/pip_dl_standard_claimants" in names
    assert "dwp/pip_dl_enhanced_claimants" in names


def test_voa_council_tax_targets():
    """VOA council tax band targets should exist."""
    targets = get_all_targets(year=2024)
    voa = [t for t in targets if t.source == "voa"]
    # 11 regions × 9 (8 bands + total) = 99
    assert len(voa) >= 90, f"Expected 90+ VOA targets, got {len(voa)}"


def test_core_target_count():
    """Total target count should be substantial."""
    targets = get_all_targets(year=2025)
    assert len(targets) >= 200, (
        f"Expected 200+ targets for 2025, got {len(targets)}"
    )


def test_two_child_limit_targets():
    """Two-child limit targets should exist."""
    targets = get_all_targets(year=2026)
    names = {t.name for t in targets}
    assert "dwp/uc/two_child_limit/households_affected" in names
    assert "dwp/uc/two_child_limit/children_affected" in names


def test_scottish_child_payment():
    """Scottish child payment should exist."""
    targets = get_all_targets(year=2025)
    names = {t.name for t in targets}
    assert "sss/scottish_child_payment" in names


def test_savings_interest():
    """ONS savings interest target should exist."""
    targets = get_all_targets(year=2025)
    names = {t.name for t in targets}
    assert "ons/savings_interest_income" in names
