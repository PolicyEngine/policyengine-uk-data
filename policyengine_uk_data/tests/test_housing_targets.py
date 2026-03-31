"""Tests for housing affordability calibration targets.

See: https://github.com/PolicyEngine/policyengine-uk-data/issues/317
"""

from policyengine_uk_data.targets.sources.housing import (
    _MORTGAGE_TOTAL,
    _PRIVATE_RENT_TOTAL,
    _SOCIAL_RENT_TOTAL,
    get_targets,
)


# ── Target values ────────────────────────────────────────────────────


def test_private_rent_total_in_range():
    """Private rent total should be £70bn-£110bn (5.4m × ~£1,374/mo)."""
    assert 70e9 < _PRIVATE_RENT_TOTAL < 110e9


def test_social_rent_total_in_range():
    """Social rent total should be £20bn-£40bn (5.0m × ~£118/wk)."""
    assert 20e9 < _SOCIAL_RENT_TOTAL < 40e9


def test_mortgage_total_in_range():
    """Mortgage total should be £80bn-£120bn (7.5m × ~£1,100/mo)."""
    assert 80e9 < _MORTGAGE_TOTAL < 120e9


def test_social_rent_less_than_private():
    """Total social rent should be less than total private rent."""
    assert _SOCIAL_RENT_TOTAL < _PRIVATE_RENT_TOTAL


# ── Target structure ─────────────────────────────────────────────────


def test_get_targets_returns_three():
    """get_targets() should return mortgage, private rent, and social rent."""
    targets = get_targets()
    assert len(targets) == 3


def test_target_names():
    """Target names should match expected values."""
    names = {t.name for t in get_targets()}
    assert names == {
        "housing/total_mortgage",
        "housing/rent_private",
        "housing/rent_social",
    }


def test_all_targets_have_2025():
    """All housing targets should have a value for 2025."""
    for t in get_targets():
        assert 2025 in t.values, f"{t.name} missing value for 2025"


def test_social_rent_target_variable():
    """Social rent target should use the rent variable."""
    targets = get_targets()
    social = next(t for t in targets if t.name == "housing/rent_social")
    assert social.variable == "rent"


def test_targets_in_registry():
    """Housing targets should appear in the global registry."""
    from policyengine_uk_data.targets import get_all_targets

    targets = get_all_targets(year=2025)
    names = {t.name for t in targets}
    assert "housing/rent_social" in names
    assert "housing/rent_private" in names
    assert "housing/total_mortgage" in names
