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
and the September 2025 "30 free hours for under-5s" boost in uptake.

Other programme targets are at their prior DfE values.
"""

# Spending in £bn, caseload in thousands of children, both for 2024.
TARGETS = {
    "spending": {
        "tfc": 0.63,
        "extended": 2.5,
        "targeted": 0.6,
        "universal": 1.7,
    },
    "caseload": {
        "tfc": 1_085,
        "extended": 740,
        "targeted": 130,
        "universal": 490,
    },
}

# Fraction by which the built dataset may differ from a target before the
# check fails. The check previously allowed any ratio in (0, 2), which cannot
# detect even a doubling — Tax-Free Childcare spending was passing at 1.96x.
TOLERANCE = 0.4

# Targets the built dataset is known not to meet, with the issue tracking each.
# Listed rather than silently tolerated so that CI reports the gap without
# failing, and so that closing the gap is a visible change.
KNOWN_MISSES = {
    ("spending", "tfc"): (
        "TFC spending is ~1.9x its target while caseload is within 2% of its "
        "own, so the gap is the average award rather than take-up. No uniform "
        "take-up rate hits both. See PolicyEngine/policyengine-uk-data#470."
    ),
}
