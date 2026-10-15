"""No workflow uploads a raw calibration log.

The logs hold each target's weighted estimate, nationally and for every
constituency and local authority, with no small-cell control. A target
supported by a few households, divided by a published count, could give one
household's value back, and this repository's Actions artifacts are public.
"""

from pathlib import Path

import pytest
import yaml

WORKFLOWS = sorted((Path(__file__).parents[2] / ".github" / "workflows").glob("*.y*ml"))


def _uploads(workflow: Path) -> list:
    jobs = yaml.safe_load(workflow.read_text()).get("jobs") or {}
    return [
        str(step.get("with", {}).get("path", ""))
        for job in jobs.values()
        for step in job.get("steps", [])
        if str(step.get("uses", "")).startswith("actions/upload-artifact")
    ]


def test_the_workflows_are_found():
    names = {w.name for w in WORKFLOWS}
    assert {"pull_request.yaml", "push.yaml"} <= names


@pytest.mark.parametrize("workflow", WORKFLOWS, ids=lambda w: w.name)
def test_no_workflow_uploads_a_raw_calibration_log(workflow):
    uploads = _uploads(workflow)
    assert not [p for p in uploads if "calibration_log" in p], uploads
