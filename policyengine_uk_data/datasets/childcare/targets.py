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

**Universal and early learning for 2-year-olds.** From DfE, "Funded early
education and childcare", reporting year 2026, national figures for January
2024 (``data/headline_figures_feeac_2011_2026.csv`` in the release bundle):

  release       https://explore-education-statistics.service.gov.uk/find-statistics/funded-early-education-and-childcare/2026
  the data      https://explore-education-statistics.service.gov.uk/data-catalogue/funded-early-education-and-childcare/2026

  registered for the universal entitlement, excluding reception   778,327
  registered for the working parent entitlement, aged 3 to 4      361,790
  => registered for the universal entitlement only                416,537

  early learning for 2-year-olds, registered                      115,852
  early learning for 2-year-olds, eligible                        154,957

The universal figure nets off the working parent entitlement because
``universal_childcare_entitlement_eligible`` in policyengine-uk ends with
``& ~has_extended_childcare`` — the schemes are modelled as mutually
exclusive, so the comparator is children on the universal entitlement *only*,
not the 1.13 million headline. The subtraction is only correct while that
exclusion holds: an eligibility refactor that dropped it would make 416,537
the wrong comparator without any target here looking wrong.
test_universal_eligibility_still_excludes_the_working_parent_scheme in
tests/test_childcare_targets.py asserts it against the installed
policyengine-uk, so the cross-repo dependency fails loudly instead of
silently. The prior 490 thousand target was 1.18x that
figure and the prior 130 thousand target was 1.12x the EL2 count, both
unsourced.

**No universal or targeted spending target.** DfE publishes January
headcounts of children registered for at least some provision, and publishes
no per-programme spending. The only spending figure derivable from it is the
caseload at the statutory 570 hours and the DfE funding rate for the age band
(£5.88 an hour for 3 and 4-year-olds in 2024-25, £8.28 for 2-year-olds):

  universal  416,537 x 570 x 5.88 = £1.396bn
  targeted   115,852 x 570 x 8.28 = £0.547bn

Those are not observations. Being caseload times a constant, each is the
caseload target restated in pounds, and `takeup_rate.objective` sums an
equally weighted squared relative error over every entry in both dictionaries:

    for key in targets["spending"]:
        loss += (spending[key] / targets["spending"][key] - 1) ** 2
    for key in targets["caseload"]:
        loss += (caseload[key] / targets["caseload"][key] - 1) ** 2

Since the model pays every recipient of these two schemes the same per-child
amount, the spending ratio equals the caseload ratio and the loop adds the
same term twice — doubling the pull of universal and targeted against
Tax-Free Childcare and extended hours, on no extra evidence. Both also assume
every registered child took the full 570 hours, so they are upper bounds and
the duplicated term pulls weights up.

Tax-Free Childcare spending stays, because it is not redundant: it varies
with childcare expenditure rather than being its caseload times a constant,
and it is a published outturn. Restoring universal and targeted spending
needs an allocation or outturn source, not a headcount.

"""

# Spending in £bn, caseload in thousands of children, both for 2024.
TARGETS = {
    "spending": {
        "tfc": 0.6322,  # HMRC £632.2m, 2024-25
        # No extended, universal or targeted entry. Universal and targeted are
        # the caseload times a constant, which duplicates the caseload term in
        # the loss rather than adding evidence; extended's derivable figure is
        # a full-usage ceiling the model pays 75% of. See the module docstring.
    },
    "caseload": {
        "tfc": 1_085.02,  # HMRC 1,085,020 children, 2024-25
        "extended": 621.5,  # DfE Jan 2025: 379,000 + 242,500
        "targeted": 115.852,  # DfE Jan 2024: 115,852 registered
        "universal": 416.537,  # DfE Jan 2024: 778,327 - 361,790
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
#   failing test there stops the upload. That is the release gate as the
#   workflow stands for release 1.56.16 — 512 epochs and one OA clone. It has
#   not always been so: release testing briefly ran with TESTING=1 in
#   December 2025, so read the workflow rather than assuming continuity. The
#   1.87x artefact passed through the gate because the tolerance let it, not
#   because the gate was missing.
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
    # Both HMRC figures are official published outturns — the top-up rounded
    # to £0.1m — which is why they are held tighter than the rest. 0.25 is a
    # QA judgement, not a measured bound:
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


# The tolerances above are release thresholds, and a pull-request build
# cannot be held to them. It calibrates for 32 epochs rather than 512 and is
# under-converged by construction: Tax-Free Childcare spending measured 1.11x
# on the smoke build at 2505d334 and 0.60x at 39ec2161, a range no
# release-representative threshold absorbs.
#
#   1.11x  https://github.com/PolicyEngine/policyengine-uk-data/actions/runs/33193249421
#   0.60x  https://github.com/PolicyEngine/policyengine-uk-data/actions/runs/33246122109
#
# Those are different commits, so this is a range across branch builds rather
# than a demonstrated same-SHA flake. Neither commit changed how the extended
# hours are drawn or what TFC pays, which is why the spread looks like
# under-convergence — but no same-SHA rerun has been captured, so it is not
# claimed as one. Either way a 32-epoch build cannot support a 25% threshold.
#
# So the smoke build gets its own override, and a weak one on purpose. What a
# 32-epoch run can show is that the pipeline runs, the variables resolve, and
# nothing has collapsed or run away. It cannot show that the numbers are
# right — only push.yaml's 512-epoch build can, and that is what the
# tolerances above gate.
#
# Scoped to Tax-Free Childcare spending, the one check that needs it. The
# other six ran at 0.79 to 1.12 on the same build and keep their release
# thresholds, so a regression in any of them still fails the pull request.
# Add an entry only for a miss a release build does not share: a smoke
# failure that reproduces at 512 epochs is a real one.
SMOKE_TOLERANCE = 0.6
SMOKE_TOLERANCES: dict[tuple[str, str], float] = {
    ("spending", "tfc"): SMOKE_TOLERANCE,
}


def tolerance(metric: str, programme: str, smoke: bool = False) -> float:
    """Allowed fractional deviation from target for one programme and metric.

    ``smoke`` selects the reduced-epoch pull-request contract, which exists
    only for the targets in SMOKE_TOLERANCES. Everything else keeps its
    release threshold in both builds, so this loosens exactly what it
    documents and nothing else.
    """
    if smoke and (metric, programme) in SMOKE_TOLERANCES:
        return SMOKE_TOLERANCES[(metric, programme)]
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
# Deliberately empty for Tax-Free Childcare: the 1.87x on the v1.56.16
# artefact was corrected by #473, merged 28 August 2026, so there is nothing
# left to wave through. An entry here would only hide the next regression.
KNOWN_MISSES: dict[tuple[str, str], str] = {}
