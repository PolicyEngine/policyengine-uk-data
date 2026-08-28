"""Calibration targets for the childcare programmes.

Single source of truth, imported by both the take-up optimisation
(``takeup_rate.py``) and the test that checks the built dataset against these
targets. They were previously duplicated, and had drifted: the test still
asserted the 0.6 / 660 Tax-Free Childcare targets from the September 2024
release after ``takeup_rate.py`` had moved to the June 2025 figures.

TFC targets are from HMRC "Tax-Free Childcare statistics: June 2025"
(published 27 August 2025, covering the 2024-25 outturn):

  - spending: £632.2m (Table 1, annual government top-up)
  - caseload: 1,085 thousand children with used accounts in 2024-25 (Table 2)

The caseload target was previously recorded as 985 thousand, citing the same
release. That release reports 1,085,020 children with used accounts in
2024-25, unrevised in the March 2026 release, and 985,000 does not appear
anywhere in it. Corrected here to match the cited source.

The prior 0.6 / 660 targets were calibrated against the September 2024 release
(2023-24 outturn) and have since been overtaken by the TFC account expansion
and the April and September 2024 expansions of the working-parent
entitlement, which are the changes falling inside the target year. (An
earlier version of this note cited the September 2025 expansion, which
cannot affect a 2024-25 outturn.)

**Period mapping.** HMRC and DfE report financial years; the calibration
checks evaluate the model at annual period 2024. Targets here are the
2024-25 financial year figures. For spending the difference is small — the
HMRC monthly series sums to £638.2m for calendar 2024 against the £632.2m
fiscal-year figure, under 1% — but an annual-unique caseload cannot be
converted by summing months, so the fiscal-year count is used as-is.

**Sources for the other programmes.** The extended (working parent) targets
are inherited unsourced from before this module existed and are flagged for
tracing. The universal and early-learning-for-2-year-olds targets are
corrected against DfE's published figures in a follow-up change; see
"Funded early education and childcare", reporting year 2026
(https://explore-education-statistics.service.gov.uk/find-statistics/funded-early-education-and-childcare/2026).
"""

# Spending in £bn, caseload in thousands of children, both for 2024.
TARGETS = {
    "spending": {
        "tfc": 0.6322,  # HMRC £632.2m, 2024-25
        "extended": 2.5,
        "targeted": 0.6,
        "universal": 1.7,
    },
    "caseload": {
        "tfc": 1_085.02,  # HMRC 1,085,020 children, 2024-25
        "extended": 740,
        "targeted": 130,
        "universal": 490,
    },
}

# Fraction by which the built dataset may differ from a target before the
# check fails. The previous check allowed any ratio in (0, 2), which cannot
# detect even a doubling.
#
# 0.4 is a first step, not a resting place: it is wide enough that a 39% error
# still passes. Tightening it needs deviations measured on the artefact users
# actually receive, and CI cannot supply those — see the warning below.
# `report_ratios` records what the CI build sees on every run, which is a
# starting point but not the same thing.
#
# IMPORTANT — WHAT THIS CHECK CAN AND CANNOT SEE
#
# CI builds with TESTING=1, which cuts calibration from 512 epochs to 32
# (datasets/create_datasets.py). The resulting weights are under-converged by
# construction, and other tests in this repo say so explicitly: see
# test_vehicle_ownership.py ("under the reduced-epoch CI build the
# vehicle-ownership target under-converges ... the full-calibration release
# dataset matches NTS") and test_scotland_babies.py.
#
# So this check validates a smoke build, not the release artefact. The two can
# diverge a long way: Tax-Free Childcare spending is 1.87x target on the
# published enhanced_frs_2024_25 (v1.56.16) against about 1.12x on a CI build.
# A green run here is not evidence that the released dataset meets its targets.
# Closing that gap needs a release-calibration gate, which does not exist yet.
DEFAULT_TOLERANCE = 0.4
TOLERANCES: dict[tuple[str, str], float] = {}


def tolerance(metric: str, programme: str) -> float:
    """Allowed fractional deviation from target for one programme and metric."""
    return TOLERANCES.get((metric, programme), DEFAULT_TOLERANCE)


# Targets the built dataset is known not to meet, with the issue tracking each.
# Listed rather than silently tolerated so that CI reports the gap without
# failing. A companion test fails if an entry here starts passing, so the list
# cannot outlive the problems it records — which is how the initial entry for
# Tax-Free Childcare spending was removed: it was recorded from the published
# artefact at 1.87x, and the built dataset meets the target at 1.12x.
KNOWN_MISSES: dict[tuple[str, str], str] = {}
