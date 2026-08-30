"""Data-independent checks on the childcare calibration targets.

`test_childcare.py` skips entirely when the private enhanced FRS is absent, so
these cover the parts that do not need a dataset: the shape of the targets, the
tolerance lookup, and the known-miss guard.
"""

import importlib

import pytest

from policyengine_uk_data.datasets.childcare.targets import (
    DEFAULT_TOLERANCE,
    KNOWN_MISSES,
    SMOKE_TOLERANCE,
    SMOKE_TOLERANCES,
    TARGETS,
    TOLERANCES,
    tolerance,
)

PROGRAMMES = {"tfc", "extended", "targeted", "universal"}

# Spending is deliberately sparse: extended has no target, because the only
# figure derivable from DfE is a full-usage ceiling the model pays 75% of.
# See the targets module docstring and test_extended_has_no_spending_target.
SPENDING_PROGRAMMES = PROGRAMMES - {"extended"}


def test_the_registry_covers_the_programmes_it_claims():
    assert set(TARGETS) == {"spending", "caseload"}
    assert set(TARGETS["caseload"]) == PROGRAMMES
    assert set(TARGETS["spending"]) == SPENDING_PROGRAMMES


def test_targets_are_positive_and_in_their_stated_units():
    # Spending in £bn, caseload in thousands of children.
    for programme, value in TARGETS["spending"].items():
        assert 0 < value < 20, f"{programme} spending {value} is not £bn"
    for programme, value in TARGETS["caseload"].items():
        assert 0 < value < 10_000, f"{programme} caseload {value} is not thousands"


def test_tfc_targets_keep_the_published_precision():
    # HMRC, Tax-Free Childcare statistics, 2024-25: £632.2m of top-up paid to
    # 1,085,020 children with used accounts.
    assert TARGETS["spending"]["tfc"] == pytest.approx(0.6322)
    assert TARGETS["caseload"]["tfc"] == pytest.approx(1_085.02)


def test_tolerance_falls_back_to_the_default():
    # No override is set for these, so they take the default.
    assert ("caseload", "universal") not in TOLERANCES
    assert tolerance("caseload", "universal") == DEFAULT_TOLERANCE
    assert tolerance("caseload", "extended") == DEFAULT_TOLERANCE


def test_the_committed_tolerance_overrides_are_the_ones_that_apply():
    # Both Tax-Free Childcare figures are official published outturns, so
    # they are held tighter than the default. 0.25 is a provisional QA
    # threshold pending a captured 512-epoch result, not a measured bound.
    assert tolerance("spending", "tfc") == 0.25
    assert tolerance("caseload", "tfc") == 0.25


def test_an_override_is_preferred_to_the_default(monkeypatch):
    monkeypatch.setitem(TOLERANCES, ("caseload", "universal"), 0.05)
    assert tolerance("caseload", "universal") == 0.05


def test_known_misses_name_real_programmes_and_carry_a_reason():
    for (metric, programme), reason in KNOWN_MISSES.items():
        assert metric in TARGETS, metric
        assert programme in PROGRAMMES, programme
        # A bare exemption is worse than none: it has to say why and where.
        assert len(reason) > 40, (metric, programme)
        assert "#" in reason, f"{metric}/{programme} should cite an issue"


def test_extended_caseload_matches_the_dfe_january_2025_census():
    """Extended is 3-4 plus 2-year-olds on the working parent entitlement.

    January 2024 cannot serve this programme: the 2-year-old entitlement began
    in April 2024, so that census counts only 3 and 4-year-olds and misses half
    the scheme the model implements.
    """
    three_and_four, two_year_olds = 379_000, 242_500
    assert TARGETS["caseload"]["extended"] == pytest.approx(
        (three_and_four + two_year_olds) / 1e3
    )


def test_extended_has_no_spending_target():
    """The only derivable figure is a full-usage ceiling the model pays 75% of.

    3 and 4-year-olds get 1,140 funded hours in the model, not 570, so the
    naive 570-hour construction is 65% of the comparable model quantity, and
    even the correct one calibrates an hours distribution to a ceiling.
    """
    assert "extended" not in TARGETS["spending"]


def test_the_smoke_build_has_its_own_weaker_contract():
    """A 32-epoch build cannot be held to a release threshold.

    Tax-Free Childcare spending measured 1.11x on one branch's smoke build
    and 0.60x on another's, so the smoke contract catches collapse and
    runaway and nothing finer. The release thresholds validate the numbers.
    """
    assert SMOKE_TOLERANCE > DEFAULT_TOLERANCE
    # Both observed smoke ratios clear it, with room for the next one.
    for ratio in (0.60, 1.11):
        assert abs(ratio - 1) < SMOKE_TOLERANCE
    # A target that lost or doubled its population still fails.
    for ratio in (0.2, 2.0):
        assert abs(ratio - 1) > SMOKE_TOLERANCE


def test_the_smoke_contract_is_scoped_to_the_target_that_needs_it():
    """Only TFC spending is loosened; the other six keep release thresholds.

    A global smoke override would widen every check to ±60% and let a real
    regression in any of them through the pull request.
    """
    assert set(SMOKE_TOLERANCES) == {("spending", "tfc")}
    for metric in TARGETS:
        for programme in TARGETS[metric]:
            release = tolerance(metric, programme)
            smoke = tolerance(metric, programme, smoke=True)
            if (metric, programme) in SMOKE_TOLERANCES:
                assert smoke == SMOKE_TOLERANCES[(metric, programme)]
            else:
                assert smoke == release, (
                    f"{metric}/{programme} has no smoke override, so the "
                    "smoke build must hold it to its release threshold"
                )


def test_no_tolerance_admits_a_lost_or_doubled_population():
    """An unpinned 1.1 or 99 would pass everything, in either build."""
    for metric in TARGETS:
        for programme in TARGETS[metric]:
            for allowed in (
                tolerance(metric, programme),
                tolerance(metric, programme, smoke=True),
            ):
                assert 0 < allowed <= SMOKE_TOLERANCE, (
                    f"{metric}/{programme}: a tolerance above the smoke "
                    "contract cannot reject a zeroed or doubled target"
                )


def test_the_optimiser_fits_take_up_rates_only():
    """Pins the C5 fix: the hours distribution is not a fitted parameter.

    Without an extended spending target the objective cannot identify the
    hours mean and sd — (15, 5) and (30, 10) give the same clipped mask and
    so the same loss — so they are fixed assumptions shared by both draw
    sites. Re-adding them to the optimiser must fail here.
    """
    import inspect

    from policyengine_uk_data.datasets.childcare import takeup_rate
    from policyengine_uk_data.datasets.childcare.assumptions import (
        EXTENDED_HOURS_MEAN,
        EXTENDED_HOURS_SD,
    )

    source = inspect.getsource(takeup_rate)
    assert "tfc, extended, targeted, universal = params" in source, (
        "the optimiser takes four take-up rates; the hours distribution is an "
        "assumption, not a fitted parameter"
    )
    assert "x0 = [0.5, 0.5, 0.5, 0.5]" in source

    # Both draw sites consume the shared assumptions rather than literals.
    frs_source = inspect.getsource(
        importlib.import_module("policyengine_uk_data.datasets.frs")
    )
    for module_source in (source, frs_source):
        assert "EXTENDED_HOURS_MEAN" in module_source
        assert "EXTENDED_HOURS_SD" in module_source
    assert (EXTENDED_HOURS_MEAN, EXTENDED_HOURS_SD) == (15.019, 4.972)
