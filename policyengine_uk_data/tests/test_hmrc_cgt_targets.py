"""Tests for the HMRC capital gains targets.

HMRC counts CGT *taxpayers* — people whose gains exceed the annual
exempt amount (AEA) — not everyone with a positive imputed gain. These
tests pin the AEA gate on both the count and the amount, and pin the
AEA to the year in force so a future edit cannot silently hard-code a
single threshold.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from policyengine_uk_data.targets.sources.hmrc_cgt import (
    _CGT_BASE_YEAR,
    _CGT_TAXPAYERS,
    _CGT_TOTAL_GAINS,
    compute_capital_gains_total,
    compute_cgt_taxpayers,
    get_targets,
)


def _dummy_ctx(capital_gains, household_of_person, aea_by_year=None):
    """Minimal _SimContext stand-in.

    ``household_of_person`` maps each person to a household index.
    """
    aea_by_year = aea_by_year or {2023: 6_000, 2024: 3_000, 2026: 3_000}
    n_households = max(household_of_person) + 1

    class _Params:
        def __init__(self, value):
            self.gov = SimpleNamespace(
                hmrc=SimpleNamespace(cgt=SimpleNamespace(annual_exempt_amount=value))
            )

    def parameters(instant):
        return _Params(aea_by_year[int(str(instant)[:4])])

    ctx = SimpleNamespace()
    ctx.sim = SimpleNamespace(tax_benefit_system=SimpleNamespace(parameters=parameters))
    ctx.pe_person = lambda variable: (
        np.array(capital_gains, dtype=float)
        if variable == "capital_gains"
        else pytest.fail(f"unexpected pe_person call: {variable}")
    )
    ctx.household_from_person = lambda values: np.bincount(
        np.array(household_of_person),
        weights=np.asarray(values),
        minlength=n_households,
    )
    return ctx


def test_count_excludes_people_below_the_aea():
    """Sub-threshold gains are not CGT taxpayers, however many there are."""
    ctx = _dummy_ctx(
        capital_gains=[100.0, 2_999.0, 3_001.0, 50_000.0],
        household_of_person=[0, 0, 1, 2],
    )
    out = compute_cgt_taxpayers(ctx, SimpleNamespace(name="hmrc/cgt_taxpayers"), 2026)
    np.testing.assert_array_equal(out, [0.0, 1.0, 1.0])


def test_amount_is_gated_on_the_same_threshold_as_the_count():
    """HMRC's £65.9bn is gains of CGT-liable taxpayers, not all gains."""
    ctx = _dummy_ctx(
        capital_gains=[100.0, 2_999.0, 3_001.0, 50_000.0],
        household_of_person=[0, 0, 1, 2],
    )
    out = compute_capital_gains_total(
        ctx, SimpleNamespace(name="hmrc/capital_gains_total"), 2026
    )
    np.testing.assert_array_equal(out, [0.0, 3_001.0, 50_000.0])


def test_aea_tracks_the_year():
    """A £5,000 gain is a taxpayer in 2024 (£3,000 AEA) but not in 2023
    (£6,000 AEA). A hard-coded threshold would fail one of these."""
    args = dict(capital_gains=[5_000.0], household_of_person=[0])
    target = SimpleNamespace(name="hmrc/cgt_taxpayers")
    assert compute_cgt_taxpayers(_dummy_ctx(**args), target, 2024)[0] == 1.0
    assert compute_cgt_taxpayers(_dummy_ctx(**args), target, 2023)[0] == 0.0


def test_targets_load_with_expected_base_year_values():
    targets = {t.name: t for t in get_targets()}
    assert set(targets) == {"hmrc/capital_gains_total", "hmrc/cgt_taxpayers"}

    gains = targets["hmrc/capital_gains_total"]
    assert gains.values[_CGT_BASE_YEAR] == pytest.approx(_CGT_TOTAL_GAINS)
    assert gains.variable == "capital_gains"
    assert gains.custom_compute is compute_capital_gains_total

    count = targets["hmrc/cgt_taxpayers"]
    assert count.is_count
    assert count.custom_compute is compute_cgt_taxpayers
    assert all(v == _CGT_TAXPAYERS for v in count.values.values())


def test_gains_target_is_projected_and_increasing():
    """A single-outturn-year target would not constrain the projection
    years the calibration runs for."""
    gains = {t.name: t for t in get_targets()}["hmrc/capital_gains_total"]
    years = sorted(gains.values)
    assert years[-1] >= 2029
    assert all(gains.values[b] > gains.values[a] for a, b in zip(years, years[1:]))


def test_targets_are_discovered_by_the_registry():
    from policyengine_uk_data.targets import get_all_targets

    names = {t.name for t in get_all_targets(year=2026)}
    assert {"hmrc/capital_gains_total", "hmrc/cgt_taxpayers"} <= names
