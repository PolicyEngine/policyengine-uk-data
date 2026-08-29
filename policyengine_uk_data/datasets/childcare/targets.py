"""Calibration targets for the childcare programmes.

Single source of truth, imported by both the take-up optimisation
(``takeup_rate.py``) and the test that checks the built dataset against these
targets. They were previously duplicated, and had drifted: the test still
asserted the 0.6 / 660 Tax-Free Childcare targets from the September 2024
release after ``takeup_rate.py`` had moved to the June 2025 figures.

TFC targets are from HMRC "Tax-Free Childcare statistics: June 2025"
(published 27 August 2025, covering the 2024-25 outturn):

https://www.gov.uk/government/statistics/tax-free-childcare-statistics-june-2025
https://assets.publishing.service.gov.uk/media/689f2a02b4b6acd341133a76/Tables_and_Statistics_June_2025.ods


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

**Extended (working parent).** From DfE, "Funded early education and
childcare", reporting year 2025 — the January 2025 census:

  3 and 4-year-olds registered for the working parent entitlement   379,000
  2-year-olds registered for the working parent entitlement         242,500
  => matching the ages the model places on this scheme              621,500

https://explore-education-statistics.service.gov.uk/find-statistics/funded-early-education-and-childcare/2025

The prior 740 thousand was inherited from before this module existed and
could not be traced to any release.

**No extended spending target.** The prior £2.5bn was equally untraceable,
and the only figure derivable from DfE is a full-usage construction that does
not match what the model pays. ``extended_childcare_entitlement`` gives 3 and
4-year-olds 30 weekly hours (1,140 a year), because extended-eligible
children are excluded from the separate universal variable; 2-year-olds get
15 (570). At the 2024-25 funding rates that is:

    379,000 x 1,140 x 5.88 = £2.540bn
    242,500 x   570 x 8.28 = £1.145bn
                             £3.685bn

against £2.778bn modelled on enhanced_frs_2024_25 — the weighted sum of
``extended_childcare_entitlement``, a model output rather than a DfE outturn,
and 75% of the full-usage ceiling above.
An earlier draft of this module set £2.415bn by giving both age groups 570
hours, which is 65% of the comparable model quantity and would have pulled
the calibrated hours distribution against a number the model cannot reach.
Even correctly constructed, £3.685bn assumes every registered child took
every funded hour: it is a ceiling, not an outturn, and calibrating
``maximum_extended_childcare_hours_usage`` to a ceiling biases hours upward.
No published outturn exists to replace it. The constraint stays out until one
does.

*Caseload basis is January 2025, not January 2024.* Every other target here
is for 2024. January 2024 cannot serve this programme: the 2-year-old working
parent entitlement began in April 2024, so that census counts only 3 and
4-year-olds (361,800) and misses half the scheme the model implements. The
model evaluates at annual period 2024 while this comparator is an
end-of-expansion stock, and the bias from that mismatch is not quantified —
an annual-average 2024 count would need monthly 2-year-old registrations
that DfE does not publish.

**Not covered at all: under-2s.** DfE reports 195,100 one-year-olds and
29,200 children aged 9 to 11 months registered in January 2025. The model
places nobody under 2 on this scheme, so those 224,300 children are outside
both the target and the model. That is a coverage gap in the model rather
than a calibration error, and no change to these targets addresses it.

**Universal and early learning for 2-year-olds.** Corrected against DfE's
published figures in a follow-up change; see "Funded early education and
childcare", reporting year 2026
(https://explore-education-statistics.service.gov.uk/find-statistics/funded-early-education-and-childcare/2026).
"""

# Spending in £bn, caseload in thousands of children, both for 2024.
TARGETS = {
    "spending": {
        "tfc": 0.6322,  # HMRC £632.2m, 2024-25
        # No extended entry: the only derivable figure is a full-usage ceiling
        # that the model pays 75% of. See the module docstring.
        "targeted": 0.6,
        "universal": 1.7,
    },
    "caseload": {
        "tfc": 1_085.02,  # HMRC 1,085,020 children, 2024-25
        "extended": 621.5,  # DfE Jan 2025: 379,000 + 242,500
        "targeted": 130,
        "universal": 490,
    },
}

# Fraction by which the built dataset may differ from a target before the
# check fails. The previous check allowed any ratio in (0, 2), which cannot
# detect even a doubling — and did not: Tax-Free Childcare spending shipped
# at 1.87x target in enhanced_frs_2024_25 v1.56.16 with 0.13 to spare.
#
# WHICH BUILD THIS CHECK VALIDATES
#
# Two builds run the test, and they are not the same artefact.
#
#   pull_request.yaml sets TESTING=1, which cuts calibration from 512 epochs
#   to 32 (datasets/create_datasets.py). Those weights are under-converged by
#   construction — test_vehicle_ownership.py and test_scotland_babies.py say
#   so explicitly — and the ratios they produce can sit a long way from the
#   release: about 1.12x for Tax-Free Childcare spending where the release
#   was 1.87x. A green pull request is a smoke check, not validation.
#
#   push.yaml runs on every release (it triggers on the pyproject.toml bump
#   that versioning.yaml pushes to main), builds at the full 512 epochs with
#   no TESTING flag, runs `make test`, and only then runs `make upload`. A
#   failing test there stops the upload. That is the release gate, and it has
#   been wired that way since July 2025; the 1.87x artefact passed through it
#   because the tolerance let it, not because the gate was missing.
#
# test_release_gate_is_wired in tests/test_childcare.py asserts that
# ordering, so a change to push.yaml that ran tests after the upload, set
# TESTING on the release build, or let the test step fail without stopping
# the job would itself fail CI.
#
# Consequences for anyone editing this file:
#
#   - Measure KNOWN_MISSES and tolerance overrides on the release build (the
#     push.yaml log, or the published artefact), never on a pull-request run.
#   - Tightening a tolerance can block the next release. That is the point,
#     but it means the fix that lets the target pass must merge first. The
#     1.87x Tax-Free Childcare spending miss on the v1.56.16 artefact was
#     corrected by #473, merged 28 August 2026, before these tolerances
#     landed.
#
# 0.4 is a first step, not a resting place: a 39% error still passes. It is
# tightened per target below only where a release-representative measurement
# exists.
DEFAULT_TOLERANCE = 0.4
TOLERANCES: dict[tuple[str, str], float] = {
    # Both HMRC figures are exact outturns, which is why they are held
    # tighter than the rest. 0.25 is a QA judgement, not a measured bound:
    # the 1.02x seen on a local build with the corrected inputs applied
    # (policyengine-uk 2.93.0, which takes the 20% rate on gross spend, plus
    # #473's routed-spend adjustment) is not reproducible from a cited
    # release log. It leaves room for a fresh 512-epoch calibration to move
    # the weights; replace it with a measured bound, tightening to about
    # 0.15, once a push.yaml log records the release ratios.
    ("spending", "tfc"): 0.25,
    ("caseload", "tfc"): 0.25,
    # No override for extended, targeted or universal: their caseload targets
    # are January census headcounts against an annual model period, and the
    # only release-build measurements to hand were taken before the take-up
    # corrections in #474. Set from the first push.yaml log after that lands.
}


def tolerance(metric: str, programme: str) -> float:
    """Allowed fractional deviation from target for one programme and metric."""
    return TOLERANCES.get((metric, programme), DEFAULT_TOLERANCE)


# Targets the release build is known not to meet, with the issue tracking
# each. Listed rather than silently tolerated so that CI reports the gap
# without blocking the release. A companion test fails if an entry here
# starts passing, so the list cannot outlive the problems it records.
#
# Measure entries on the release build, not a pull-request run: an earlier
# entry for Tax-Free Childcare spending was removed on the strength of a
# 1.12x pull-request build while the release still sat at 1.87x.
#
# Deliberately empty for Tax-Free Childcare despite the 1.87x on main: an
# entry would wave the miss through the gate, and the correction (#473) is
# ready to merge. Blocking the release until it does is the intended outcome.
KNOWN_MISSES: dict[tuple[str, str], str] = {}
