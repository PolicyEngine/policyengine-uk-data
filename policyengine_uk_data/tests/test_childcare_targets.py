"""Data-independent checks on the childcare calibration targets.

`test_childcare.py` skips entirely when the private enhanced FRS is absent, so
these cover the parts that do not need a dataset: the shape of the targets, the
tolerance lookup, and the known-miss guard.
"""

import pytest

from policyengine_uk_data.datasets.childcare.targets import (
    DEFAULT_TOLERANCE,
    KNOWN_MISSES,
    TARGETS,
    TOLERANCES,
    tolerance,
)

PROGRAMMES = {"tfc", "extended", "targeted", "universal"}


def test_every_programme_has_both_metrics():
    assert set(TARGETS) == {"spending", "caseload"}
    for metric in TARGETS:
        assert set(TARGETS[metric]) == PROGRAMMES, metric


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
    assert tolerance("spending", "extended") == DEFAULT_TOLERANCE
    assert tolerance("caseload", "universal") == DEFAULT_TOLERANCE


def test_tolerance_overrides_are_used_when_present():
    TOLERANCES[("spending", "tfc")] = 0.05
    try:
        assert tolerance("spending", "tfc") == 0.05
        assert tolerance("caseload", "tfc") == DEFAULT_TOLERANCE
    finally:
        del TOLERANCES[("spending", "tfc")]


def test_known_misses_name_real_programmes_and_carry_a_reason():
    for (metric, programme), reason in KNOWN_MISSES.items():
        assert metric in TARGETS, metric
        assert programme in PROGRAMMES, programme
        # A bare exemption is worse than none: it has to say why and where.
        assert len(reason) > 40, (metric, programme)
        assert "#" in reason, f"{metric}/{programme} should cite an issue"


def test_extended_targets_match_the_dfe_january_2025_census():
    """Extended is 3-4 plus 2-year-olds on the working parent entitlement.

    January 2024 cannot serve this programme: the 2-year-old entitlement began
    in April 2024, so that census counts only 3 and 4-year-olds and misses half
    the scheme the model implements.
    """
    three_and_four, two_year_olds = 379_000, 242_500
    assert TARGETS["caseload"]["extended"] == pytest.approx(
        (three_and_four + two_year_olds) / 1e3
    )

    # Full 570 additional hours at the 2024-25 DfE rate for each age band.
    expected = (three_and_four * 570 * 5.88 + two_year_olds * 570 * 8.28) / 1e9
    assert TARGETS["spending"]["extended"] == pytest.approx(expected, abs=0.001)
