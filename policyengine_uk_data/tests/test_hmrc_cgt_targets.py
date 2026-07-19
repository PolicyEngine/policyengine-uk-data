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
    _MAX_YEAR,
    compute_capital_gains_total,
    compute_cgt_taxpayers,
    get_targets,
)

_GAINS_TARGET = SimpleNamespace(name="hmrc/capital_gains_total")
_COUNT_TARGET = SimpleNamespace(name="hmrc/cgt_taxpayers")


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


# ── AEA gate: boundary and year sensitivity ──────────────────────────


def test_gain_exactly_at_the_aea_is_not_a_taxpayer():
    """The gate is a strict ``>``: the AEA itself is exempt, so a gain
    of exactly the AEA leaves nothing chargeable and HMRC would not
    count that person. One pound above is a taxpayer."""
    ctx = _dummy_ctx(
        capital_gains=[3_000.0, 3_001.0],
        household_of_person=[0, 1],
    )
    np.testing.assert_array_equal(
        compute_cgt_taxpayers(ctx, _COUNT_TARGET, 2026), [0.0, 1.0]
    )
    np.testing.assert_array_equal(
        compute_capital_gains_total(ctx, _GAINS_TARGET, 2026), [0.0, 3_001.0]
    )


def test_aea_boundary_moves_with_the_year():
    """The same gains vector must produce different counts in 2023
    (£6,000 AEA) and 2024 onwards (£3,000 AEA), and the boundary must
    sit at each year's own threshold."""
    gains = [3_000.0, 3_001.0, 6_000.0, 6_001.0]
    households = [0, 1, 2, 3]

    count_2024 = compute_cgt_taxpayers(
        _dummy_ctx(gains, households), _COUNT_TARGET, 2024
    )
    count_2023 = compute_cgt_taxpayers(
        _dummy_ctx(gains, households), _COUNT_TARGET, 2023
    )

    # 2024: £3,000 AEA — everything strictly above £3,000 counts.
    np.testing.assert_array_equal(count_2024, [0.0, 1.0, 1.0, 1.0])
    # 2023: £6,000 AEA — only the £6,001 gain counts.
    np.testing.assert_array_equal(count_2023, [0.0, 0.0, 0.0, 1.0])
    assert count_2024.sum() > count_2023.sum()


# ── The amount gate and the count gate describe one population ───────


def test_amount_and_count_select_exactly_the_same_people():
    """The module docstring's central claim: the £65.9bn amount and the
    378k count are two statistics about *one* population, so no
    sub-threshold gain may leak into the amount. Putting each person in
    their own household lets us compare the two gates person by person.
    """
    gains = [
        -10_000.0,
        0.0,
        1.0,
        2_999.0,
        3_000.0,
        3_001.0,
        50_000.0,
        1_000_000.0,
    ]
    households = list(range(len(gains)))
    ctx = _dummy_ctx(gains, households)

    counted = compute_cgt_taxpayers(ctx, _COUNT_TARGET, 2026) > 0
    contributing = compute_capital_gains_total(ctx, _GAINS_TARGET, 2026) != 0

    np.testing.assert_array_equal(counted, contributing)
    # And the contributing people carry their full, unreduced gain --
    # the gate selects people, it does not deduct the AEA.
    np.testing.assert_array_equal(
        compute_capital_gains_total(ctx, _GAINS_TARGET, 2026),
        [0.0, 0.0, 0.0, 0.0, 0.0, 3_001.0, 50_000.0, 1_000_000.0],
    )


# ── Negative gains (losses) ──────────────────────────────────────────


def test_losses_contribute_nothing_to_either_target():
    ctx = _dummy_ctx(
        capital_gains=[-50_000.0, -1.0],
        household_of_person=[0, 1],
    )
    np.testing.assert_array_equal(
        compute_cgt_taxpayers(ctx, _COUNT_TARGET, 2026), [0.0, 0.0]
    )
    np.testing.assert_array_equal(
        compute_capital_gains_total(ctx, _GAINS_TARGET, 2026), [0.0, 0.0]
    )


def test_gated_gains_can_exceed_ungated_gains_because_losses_are_dropped():
    """Documented asymmetry, pinned deliberately.

    HMRC's chargeable-gains total does not net one person's loss against
    another's gain, so the gated household column can be *larger* than a
    naive household sum of ``capital_gains``. If a future edit starts
    netting losses this test fails loudly rather than quietly shifting
    the target's meaning.
    """
    # Both people in household 0: a £50k loss and a £100k gain.
    ctx = _dummy_ctx(
        capital_gains=[-50_000.0, 100_000.0],
        household_of_person=[0, 0],
    )
    gated = compute_capital_gains_total(ctx, _GAINS_TARGET, 2026)
    ungated = ctx.household_from_person(np.array([-50_000.0, 100_000.0]))

    assert gated[0] == pytest.approx(100_000.0)
    assert ungated[0] == pytest.approx(50_000.0)
    assert gated[0] > ungated[0]


# ── Projection behaviour ─────────────────────────────────────────────


def test_gains_target_covers_every_projection_year_and_rises():
    gains = {t.name: t for t in get_targets()}["hmrc/capital_gains_total"]
    expected_years = set(range(_CGT_BASE_YEAR, _MAX_YEAR + 1))
    assert set(gains.values) == expected_years

    ordered = [gains.values[y] for y in sorted(expected_years)]
    assert all(b > a for a, b in zip(ordered, ordered[1:]))


def test_taxpayer_count_is_deliberately_flat_across_projection_years():
    """The flat count is a documented choice, not an oversight.

    The gains total is uprated with PolicyEngine's ``capital_gains``
    factors, but those are a nominal-income index; applying them to a
    *headcount* would forecast more CGT taxpayers purely because gains
    are worth more in cash terms. The repo has no administrative basis
    for forecasting CGT taxpayer numbers, so the 2023-24 outturn count
    is held flat. Changing this should be a conscious decision that
    updates this test.
    """
    count = {t.name: t for t in get_targets()}["hmrc/cgt_taxpayers"]
    assert set(count.values) == set(range(_CGT_BASE_YEAR, _MAX_YEAR + 1))
    assert set(count.values.values()) == {float(_CGT_TAXPAYERS)}


# ── Registry round-trip ──────────────────────────────────────────────


def test_registry_keeps_custom_compute_for_the_calibration_years():
    """``custom_compute`` is ``Field(exclude=True)`` on the Target schema.

    Exclusion applies to serialisation, not attribute access, but a
    future change to how the registry assembles targets (a
    ``model_dump``/``model_validate`` round-trip, say) would silently
    drop the callable -- and the targets would then fall through to
    ``_compute_simple_count``, which counts everyone with positive
    gains rather than CGT taxpayers. That failure is invisible except
    as a bad calibration, so pin it here.
    """
    from policyengine_uk_data.targets import get_all_targets

    for year in range(_CGT_BASE_YEAR, _MAX_YEAR + 1):
        targets = {t.name: t for t in get_all_targets(year=year)}
        assert "hmrc/capital_gains_total" in targets, year
        assert "hmrc/cgt_taxpayers" in targets, year
        assert (
            targets["hmrc/capital_gains_total"].custom_compute
            is compute_capital_gains_total
        ), year
        assert targets["hmrc/cgt_taxpayers"].custom_compute is compute_cgt_taxpayers, (
            year
        )


def test_registry_year_filter_excludes_years_before_the_outturn():
    """The targets describe a 2023-24 outturn mapped to calendar 2024,
    so they must not appear for earlier years."""
    from policyengine_uk_data.targets import get_all_targets

    names = {t.name for t in get_all_targets(year=_CGT_BASE_YEAR - 1)}
    assert "hmrc/capital_gains_total" not in names
    assert "hmrc/cgt_taxpayers" not in names


# ── Real _SimContext integration ─────────────────────────────────────


def test_sim_context_exposes_the_methods_the_targets_call():
    """Cheap guard against a rename on the real context.

    ``_dummy_ctx`` above is a stub, so it would happily keep passing if
    ``pe_person`` or ``household_from_person`` were renamed on
    ``_SimContext``. This test needs no dataset and always runs.
    """
    from policyengine_uk_data.targets.build_loss_matrix import _SimContext

    for method in ("pe_person", "household_from_person"):
        assert callable(getattr(_SimContext, method, None)), method


@pytest.mark.slow
def test_compute_functions_run_against_a_real_simulation(enhanced_frs):
    """End-to-end check on a real Microsimulation.

    Skips (via the ``enhanced_frs`` fixture) when the dataset is not
    present locally; CI builds the datasets before running the suite, so
    this executes there. Together with the introspection test above,
    a rename of ``ctx.pe_person`` or ``ctx.household_from_person``
    fails a test rather than only failing in production.
    """
    from policyengine_uk import Microsimulation
    from policyengine_uk_data.targets.build_loss_matrix import _SimContext

    time_period = str(enhanced_frs.time_period)
    year = int(time_period)
    sim = Microsimulation(dataset=enhanced_frs)
    sim.default_calculation_period = time_period
    ctx = _SimContext(sim, time_period, enhanced_frs, None)

    weights = sim.calculate("household_weight", time_period).values
    n_households = len(weights)

    counts = compute_cgt_taxpayers(ctx, _COUNT_TARGET, year)
    gains = compute_capital_gains_total(ctx, _GAINS_TARGET, year)

    for name, col in (("count", counts), ("gains", gains)):
        assert col.shape == (n_households,), name
        assert np.all(np.isfinite(col)), name
        assert np.all(col >= 0), name

    weighted_taxpayers = float((counts * weights).sum())
    weighted_gains = float((gains * weights).sum())

    # The microdata is known to carry far more CGT taxpayers than
    # HMRC's 378k; this is a plausibility band, not a calibration check.
    assert 100_000 < weighted_taxpayers < 3_000_000, weighted_taxpayers
    assert weighted_gains > 0

    # Gating must bind: fewer taxpayers than people with any positive gain.
    any_positive = float(
        (
            ctx.household_from_person(
                (np.asarray(ctx.pe_person("capital_gains")) > 0).astype(float)
            )
            * weights
        ).sum()
    )
    assert weighted_taxpayers <= any_positive
