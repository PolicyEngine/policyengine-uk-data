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

# Iterate over the targets that exist, not every programme: extended has no
# spending target, because the only figure derivable from DfE is a full-usage
# ceiling the model pays 75% of. See targets.py.
CASES = [
    (metric, programme)
    for metric in ("spending", "caseload")
    for programme in TARGETS[metric]
]

# Which build produced the dataset under test. pull_request.yaml sets
# TESTING=1 for a 32-epoch smoke build; push.yaml builds the release at 512
# epochs with no flag and gates the upload on this suite.
# A non-testing run only tells us the dataset is not a smoke build: locally it
# may be a previously downloaded artefact rather than one built here.
SMOKE = os.environ.get("TESTING") == "1"
BUILD = (
    "smoke (TESTING=1, 32 epochs)"
    if SMOKE
    else "non-smoke, provenance unknown (a release build, or an artefact "
    "built or downloaded earlier)"
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
    allowed = tolerance(metric, programme, smoke=SMOKE)

    known_miss = KNOWN_MISSES.get((metric, programme))
    if known_miss is not None:
        pytest.xfail(
            f"{metric}/{programme} is a known miss ({ratio:.2f}x): {known_miss}"
        )

    assert abs(ratio - 1) < allowed, (
        f"{programme} {metric} is {actual:.3f} against a target of {target:.3f} "
        f"({ratio:.2f}x), outside the ±{allowed:.0%} tolerance "
        f"for the {BUILD} build"
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
    """The release build must run this suite before uploading.

    push.yaml is the only place the calibration targets are checked against
    the artefact users receive. This pins the properties that make it a gate,
    so a workflow edit that broke one fails CI rather than shipping an
    unvalidated dataset: the release build is a 512-epoch, one-OA-clone build
    rather than a TESTING smoke build, the tests run before the upload, and a
    failing test stops the job.

    "Not a smoke build" is not the same as full fidelity — the release runs
    one OA clone where the non-testing default is ten — so the contract is
    stated as what it is and the clone count is pinned with the rest.
    """
    workflow = yaml.safe_load(PUSH_WORKFLOW.read_text())
    job = workflow["jobs"]["test"]
    steps = job["steps"]
    by_name = {step.get("name"): step for step in steps}
    order = [step.get("name") for step in steps]

    # TESTING set at workflow or job scope would reach the build step just as
    # a step-level setting does, so all three scopes are checked.
    for scope in (workflow, job, by_name["Build datasets"]):
        assert scope.get("env", {}).get("TESTING") != "1", (
            "the release build must not be a TESTING smoke build"
        )
    assert order.index("Run tests") < order.index("Upload data"), (
        "tests must run before the upload, or a failing target cannot block it"
    )
    assert not by_name["Run tests"].get("continue-on-error", False), (
        "a failing test must stop the job, or the gate is decorative"
    )
    # `if: always()` or `if: failure()` on the upload would run it whatever the
    # tests did, which is the same as having no gate.
    assert "if" not in by_name["Upload data"], (
        "the upload must be unconditional on success, not run despite a failure"
    )
    # The release contract, pinned: one OA clone, stated rather than assumed.
    assert job["env"]["PE_UK_DATA_OA_CLONES"] == "1"

    # Pin the commands themselves. Without this the gate survives `echo make
    # test`, `make test || true`, or an inline `TESTING=1 make data` — each of
    # which leaves every assertion above true while the gate does nothing.
    expected = {
        "Build datasets": "uv run --frozen make data",
        "Run tests": "uv run --frozen make test",
        "Upload data": "uv run --frozen make upload",
    }
    for name, command in expected.items():
        assert by_name[name]["run"].strip() == command, (
            f"{name} must run exactly `{command}`: a wrapped, echoed or "
            "failure-swallowing variant is not a gate"
        )
