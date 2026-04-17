"""Tests for the year-resolution policy in build_loss_matrix (#345, step 4)."""

import pytest

from policyengine_uk_data.targets.build_loss_matrix import (
    YEAR_FALLBACK_TOLERANCE,
    resolve_target_value,
)
from policyengine_uk_data.targets.schema import (
    GeographicLevel,
    Target,
    Unit,
)


def _target(values: dict[int, float], *, source: str = "ons") -> Target:
    """Minimal Target instance for year-resolution tests."""
    return Target(
        name=f"{source}/test",
        variable="test_variable",
        source=source,
        unit=Unit.COUNT,
        geographic_level=GeographicLevel.NATIONAL,
        values=values,
    )


def test_exact_year_match_returns_that_value():
    t = _target({2024: 10.0, 2025: 20.0, 2026: 30.0})
    assert resolve_target_value(t, 2025) == 20.0


def test_exact_match_preferred_over_fallback():
    t = _target({2024: 10.0, 2025: 20.0})
    # Even though 2024 is "nearest" to 2024, the exact match for 2025
    # must win outright.
    assert resolve_target_value(t, 2024) == 10.0


def test_falls_back_to_nearest_past_year_within_tolerance():
    t = _target({2023: 7.0})
    # 2024 and 2025 fall back to 2023 within tolerance.
    assert resolve_target_value(t, 2024) == 7.0
    assert resolve_target_value(t, 2025) == 7.0
    assert resolve_target_value(t, 2026) == 7.0


def test_returns_none_when_only_future_years_available():
    """Extrapolating backwards would misreport historical reality."""
    t = _target({2025: 50.0, 2026: 60.0})
    assert resolve_target_value(t, 2023) is None
    assert resolve_target_value(t, 2024) is None


def test_returns_none_when_fallback_exceeds_tolerance():
    t = _target({2020: 5.0})
    # 2024 is four years away — beyond the three-year default.
    assert resolve_target_value(t, 2024) is None


def test_custom_tolerance_is_honoured():
    t = _target({2020: 5.0})
    # Explicitly widen the tolerance.
    assert resolve_target_value(t, 2024, tolerance=4) == 5.0
    # Or tighten it.
    assert resolve_target_value(t, 2022, tolerance=1) is None


def test_empty_values_returns_none():
    t = _target({})
    assert resolve_target_value(t, 2025) is None


def test_default_tolerance_is_three_years():
    """Lock the public tolerance constant so a change is deliberate."""
    assert YEAR_FALLBACK_TOLERANCE == 3


def test_non_voa_target_does_not_get_population_scaled():
    """Only VOA council-tax counts should track population growth."""
    t = _target({2024: 100.0}, source="dwp")
    # 2025 is within tolerance but DWP data must not be rescaled.
    assert resolve_target_value(t, 2025) == 100.0


def test_voa_target_scales_with_population_when_extrapolating(monkeypatch):
    """VOA counts must move roughly in line with population."""
    t = _target({2024: 100.0}, source="voa")

    fake_pop = {2024: 67_000_000.0, 2025: 68_000_000.0}

    def fake_total_population(year):
        return fake_pop[year]

    monkeypatch.setattr(
        "policyengine_uk_data.targets.sources.local_age.get_uk_total_population",
        fake_total_population,
    )

    resolved = resolve_target_value(t, 2025)
    expected = 100.0 * 68_000_000.0 / 67_000_000.0
    assert resolved == pytest.approx(expected)


def test_voa_target_returns_base_when_year_matches_exactly(monkeypatch):
    """Population scaling only kicks in when we actually extrapolate."""
    t = _target({2025: 123.0}, source="voa")

    # If the scaler is called, blow up — it must not be touched here.
    def explode(year):
        raise AssertionError("Population scaler called on exact match")

    monkeypatch.setattr(
        "policyengine_uk_data.targets.sources.local_age.get_uk_total_population",
        explode,
    )

    assert resolve_target_value(t, 2025) == 123.0


def test_voa_guards_against_zero_population_base(monkeypatch):
    """If the population lookup returns zero, fall back to the raw value."""
    t = _target({2024: 100.0}, source="voa")

    monkeypatch.setattr(
        "policyengine_uk_data.targets.sources.local_age.get_uk_total_population",
        lambda year: 0.0,
    )
    # Division by zero must be avoided; raw value returned as-is.
    assert resolve_target_value(t, 2025) == 100.0
