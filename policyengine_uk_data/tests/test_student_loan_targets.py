"""Tests for SLC student loan calibration targets."""


def test_slc_targets_registered():
    """SLC targets appear in the target registry."""
    from policyengine_uk_data.targets.registry import get_all_targets

    targets = {t.name: t for t in get_all_targets()}
    assert "slc/plan_2_borrowers_above_threshold" in targets
    assert "slc/plan_5_borrowers_above_threshold" in targets


def test_slc_plan2_values():
    """Plan 2 target values match SLC Table 6a."""
    from policyengine_uk_data.targets.registry import get_all_targets

    targets = {t.name: t for t in get_all_targets()}
    p2 = targets["slc/plan_2_borrowers_above_threshold"]
    assert p2.values[2025] == 3_670_000
    assert p2.values[2026] == 4_130_000
    assert p2.values[2029] == 4_820_000


def test_slc_plan5_values():
    """Plan 5 target values match SLC Table 6a."""
    from policyengine_uk_data.targets.registry import get_all_targets

    targets = {t.name: t for t in get_all_targets()}
    p5 = targets["slc/plan_5_borrowers_above_threshold"]
    assert 2025 not in p5.values  # no Plan 5 borrowers yet in 2024-25
    assert p5.values[2026] == 25_000
    assert p5.values[2029] == 700_000


def test_slc_testing_mode_uses_snapshot_without_network(monkeypatch):
    """Dataset-build CI should not depend on a live SLC endpoint."""
    from policyengine_uk_data.targets.sources import slc

    slc._fetch_slc_data.cache_clear()
    monkeypatch.setenv("TESTING", "1")

    def fail_network(*args, **kwargs):
        raise AssertionError("network should not be used in TESTING mode")

    monkeypatch.setattr(slc.requests, "get", fail_network)

    assert slc._fetch_slc_data() == slc._TESTING_DATA

    slc._fetch_slc_data.cache_clear()
