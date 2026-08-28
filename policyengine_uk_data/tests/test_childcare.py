"""Check built childcare spending and caseloads against their calibration targets.

The targets live in ``datasets/childcare/targets.py`` and are shared with the
take-up optimisation, so the two cannot drift apart again.
"""

import os
from pathlib import Path

import pytest
import yaml

from policyengine_uk_data.datasets.childcare.targets import (
    KNOWN_MISSES,
    TARGETS,
    tolerance,
)

PROGRAMMES = {
    "tfc": ("tax_free_childcare", "is_child_receiving_tax_free_childcare"),
    "extended": (
        "extended_childcare_entitlement",
        "is_child_receiving_extended_childcare",
    ),
    "targeted": (
        "targeted_childcare_entitlement",
        "is_child_receiving_targeted_childcare",
    ),
    "universal": (
        "universal_childcare_entitlement",
        "is_child_receiving_universal_childcare",
    ),
}

# Iterate over the targets that exist, not every programme: only Tax-Free
# Childcare has a spending target, because it is the only programme with a
# published expenditure outturn. See targets.py for why the rest were dropped.
CASES = [
    (metric, programme)
    for metric in ("spending", "caseload")
    for programme in TARGETS[metric]
]

# Which build produced the dataset under test. pull_request.yaml sets
# TESTING=1 for a 32-epoch smoke build; push.yaml builds the release at 512
# epochs with no flag and gates the upload on this suite.
BUILD = (
    "smoke (TESTING=1, 32 epochs)"
    if os.environ.get("TESTING") == "1"
    else ("release (512 epochs)")
)
PUSH_WORKFLOW = Path(__file__).resolve().parents[2] / ".github/workflows/push.yaml"


def measure(baseline, metric: str, programme: str) -> float:
    """Built value for one programme: spending in £bn, caseload in thousands."""
    spending_variable, caseload_variable = PROGRAMMES[programme]
    if metric == "spending":
        return baseline.calculate(spending_variable, 2024).sum() / 1e9
    return baseline.calculate(caseload_variable, 2024).sum() / 1e3


@pytest.mark.parametrize("metric,programme", CASES)
def test_childcare_hits_its_calibration_target(baseline, metric, programme):
    target = TARGETS[metric][programme]
    actual = measure(baseline, metric, programme)
    ratio = actual / target
    allowed = tolerance(metric, programme)

    known_miss = KNOWN_MISSES.get((metric, programme))
    if known_miss is not None:
        pytest.xfail(
            f"{metric}/{programme} is a known miss ({ratio:.2f}x): {known_miss}"
        )

    assert abs(ratio - 1) < allowed, (
        f"{programme} {metric} is {actual:.3f} against a target of {target:.3f} "
        f"({ratio:.2f}x), outside the ±{allowed:.0%} tolerance"
    )


def test_known_misses_are_still_missing(baseline):
    """Fail once a known miss is fixed, so the exemption gets removed.

    Without this a target could be met while CI still reports it as expected
    to fail, and the exemption would outlive the problem.
    """
    for (metric, programme), reason in KNOWN_MISSES.items():
        ratio = measure(baseline, metric, programme) / TARGETS[metric][programme]
        assert abs(ratio - 1) >= tolerance(metric, programme), (
            f"{programme} {metric} now hits its target ({ratio:.2f}x). "
            f"Remove it from KNOWN_MISSES in datasets/childcare/targets.py. "
            f"Recorded reason: {reason}"
        )


def test_report_ratios(baseline, capsys):
    """Record every programme's deviation from target, pass or fail.

    The check above reports only whether a programme is inside its tolerance.
    Printing the ratios makes the current build's actual position visible in
    CI, which is what a future tightening of the tolerances needs.
    """
    lines = [
        "",
        f"childcare calibration check — build: {BUILD}",
        f"{'metric':10s} {'programme':11s} {'built':>10s} {'target':>10s} {'ratio':>7s}",
    ]
    for metric, programme in CASES:
        target = TARGETS[metric][programme]
        actual = measure(baseline, metric, programme)
        lines.append(
            f"{metric:10s} {programme:11s} {actual:10.3f} {target:10.3f} {actual / target:6.2f}x"
        )
    with capsys.disabled():
        print("\n".join(lines))


def test_release_gate_is_wired():
    """The release build must run this suite, at full fidelity, before uploading.

    push.yaml is the only place the calibration targets are checked against
    the artefact users receive. This pins the three properties that make it a
    gate, so a workflow edit that broke one fails CI rather than shipping an
    unvalidated dataset: the release build is not a TESTING smoke build, the
    tests run before the upload, and a failing test stops the job.
    """
    workflow = yaml.safe_load(PUSH_WORKFLOW.read_text())
    steps = workflow["jobs"]["test"]["steps"]
    by_name = {step.get("name"): step for step in steps}
    order = [step.get("name") for step in steps]

    build = by_name["Build datasets"]
    assert build.get("env", {}).get("TESTING") != "1", (
        "the release build must not be a TESTING smoke build"
    )
    assert order.index("Run tests") < order.index("Upload data"), (
        "tests must run before the upload, or a failing target cannot block it"
    )
    assert not by_name["Run tests"].get("continue-on-error", False), (
        "a failing test must stop the job, or the gate is decorative"
    )
    assert "make test" in by_name["Run tests"]["run"]
