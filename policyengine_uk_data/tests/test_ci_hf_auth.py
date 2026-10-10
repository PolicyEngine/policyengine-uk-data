"""The release workflow reaches Hugging Face only through short-lived tokens.

`push.yaml` exchanges the job's GitHub OIDC token for a one-hour token scoped
to the private data repo (Hugging Face Trusted Publishers) in each step that
downloads or uploads, via `.github/with-hf-token.sh`. These tests stop a
long-lived Hugging Face secret from being wired back into the release job and
keep the token's scope equal to the repo the upload code writes to.
"""

import os
from pathlib import Path

import pytest
import yaml

from policyengine_uk_data.utils.hf_destinations import PRIVATE_REPO

REPO_ROOT = Path(__file__).resolve().parents[2]
PUSH_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "push.yaml"
TOKEN_HELPER = REPO_ROOT / ".github" / "with-hf-token.sh"
HF_MAKE_TARGETS = ("make download", "make upload")


def _release_job() -> dict:
    workflow = yaml.safe_load(PUSH_WORKFLOW.read_text())
    return workflow["jobs"]["test"]


def _hf_steps(job: dict) -> list[dict]:
    return [
        step
        for step in job["steps"]
        if any(target in step.get("run", "") for target in HF_MAKE_TARGETS)
    ]


def test_release_workflow_stores_no_hugging_face_secret():
    text = PUSH_WORKFLOW.read_text()
    assert "secrets.HUGGING_FACE_TOKEN" not in text
    assert "secrets.HF_TOKEN" not in text


def test_release_job_can_request_an_oidc_token():
    assert _release_job()["permissions"]["id-token"] == "write"


def test_no_release_step_sets_a_static_hugging_face_token():
    job = _release_job()
    assert "HUGGING_FACE_TOKEN" not in job.get("env", {})
    for step in job["steps"]:
        assert "HUGGING_FACE_TOKEN" not in step.get("env", {}), step["name"]


def test_both_hugging_face_steps_exist():
    runs = [step["run"] for step in _hf_steps(_release_job())]
    for target in HF_MAKE_TARGETS:
        assert sum(target in run for run in runs) == 1, target


@pytest.mark.parametrize("target", HF_MAKE_TARGETS)
def test_each_hugging_face_step_exchanges_its_own_token(target):
    (step,) = [s for s in _hf_steps(_release_job()) if target in s["run"]]
    assert step["run"].startswith(".github/with-hf-token.sh ")
    # The token is scoped to the repo the upload code writes to.
    assert step["env"]["HF_OIDC_RESOURCE"] == PRIVATE_REPO


def test_token_helper_is_executable_and_masks_the_token():
    assert TOKEN_HELPER.is_file()
    assert os.access(TOKEN_HELPER, os.X_OK)
    script = TOKEN_HELPER.read_text()
    assert "::add-mask::" in script
    assert "hf auth token" in script
