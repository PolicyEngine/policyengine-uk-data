"""Nothing built from the microdata leaves CI unless it has been reviewed.

The raw calibration logs hold each target's weighted estimate, nationally and
for every constituency and local authority, with no small-cell control. A
target supported by a few households, divided by a published count, could
give one household's value back, and this repository's Actions artifacts are
public. So the check is fail-closed rather than a list of forbidden names:

- A job **can reach the data** when it holds a data credential (the Hugging
  Face token or Google Cloud auth) or downloads artifacts that another job
  uploaded.
- In such a job, every action must be on ``NON_PUBLISHING_ACTIONS`` or be an
  artifact upload. Every upload must name literal files reviewed for that job
  in ``REVIEWED``: no directories, globs or expressions. Every run step that
  matches ``PUBLISH_COMMAND`` must use a command reviewed for that job.
- Jobs that cannot reach the data (docs, versioning, changelog) may publish;
  they hold nothing built from it.

Two parts rest on review rather than on the check:

- ``PUBLISH_COMMAND`` lists known shell routes out of a runner, so a new kind of
  command needs adding.
- A reviewed file's producer pattern shows that the job runs its summarising
  command. It does not prove that nothing overwrites the file afterwards.
"""

from __future__ import annotations

import copy
import re
from pathlib import Path

import pytest
import yaml

WORKFLOW_DIR = Path(__file__).parents[2] / ".github" / "workflows"
WORKFLOWS = sorted(WORKFLOW_DIR.glob("*.y*ml"))

DATA_CREDENTIALS = ("HUGGING_FACE_TOKEN", "google-github-actions/auth")

# Actions that publish nothing: set-up, checkout, sign-in and reading
# artifacts. setup-uv caches Python packages, not datasets.
NON_PUBLISHING_ACTIONS = {
    "actions/checkout",
    "actions/setup-python",
    "actions/setup-node",
    "astral-sh/setup-uv",
    "google-github-actions/auth",
    "actions/create-github-app-token",
    "actions/download-artifact",
}

# Reviewed routes out of each job that can reach the data:
# (workflow file, job id) -> artifact paths it may upload, mapped to a
# pattern its run steps must contain to produce that file from a
# disclosure-controlled summary, and the publishing commands it may run.
REVIEWED = {
    ("push.yaml", "test"): {
        "artifacts": {},
        # Private Hugging Face and GCS datasets (upload_completed_datasets.py,
        # data controller's destinations), then the release tag.
        "commands": ("uv run --frozen make upload", ".github/publish-git-tag.sh"),
    },
}

PUBLISH_COMMAND = re.compile(
    r"make upload|publish-git-tag|\bgh (release|gist|api|pr comment|issue comment)\b"
    r"|\bgit push\b|\bcurl\b.*(\s-T\b|--upload-file|\s-F\b|--data|\s-d\b|-X\s*(POST|PUT|PATCH))"
    r"|\baws s3\b|\bgsutil\b|\bgcloud storage\b|huggingface-cli|\bhf upload\b|\btwine\b"
)
_NOT_LITERAL = re.compile(r"[*?\[\]{}!]|\$\{\{")


def _jobs(workflow: dict) -> dict:
    return workflow.get("jobs") or {}


def reaches_data(job: dict) -> bool:
    text = yaml.safe_dump(job)
    if any(credential in text for credential in DATA_CREDENTIALS):
        return True
    return any(
        str(step.get("uses", "")).startswith("actions/download-artifact")
        for step in job.get("steps", [])
    )


def _action(uses: str) -> str:
    return uses.split("@", 1)[0]


def problems(name: str, workflow: dict) -> list:
    """Every unreviewed route out of a job of ``workflow`` that can reach the data."""
    out = []
    for job_id, job in _jobs(workflow).items():
        if not reaches_data(job):
            continue
        where = f"{name} job {job_id}"
        reviewed = REVIEWED.get((name, job_id), {"artifacts": {}, "commands": ()})
        if "uses" in job:
            out.append(f"{where} calls a reusable workflow ({job['uses']})")
        runs = "\n".join(str(s.get("run", "")) for s in job.get("steps", []))
        for step in job.get("steps", []):
            uses = str(step.get("uses", ""))
            if uses:
                action = _action(uses)
                if action == "actions/upload-artifact":
                    path = str((step.get("with") or {}).get("path", ""))
                    entries = [p.strip() for p in path.splitlines() if p.strip()]
                    if not entries:
                        out.append(f"{where} uploads an artifact with no path")
                    for entry in entries:
                        if _NOT_LITERAL.search(entry):
                            out.append(
                                f"{where} uploads a pattern or expression: {entry!r}"
                            )
                        elif entry in (".", "./") or entry.endswith("/"):
                            out.append(f"{where} uploads a directory: {entry!r}")
                        elif entry not in reviewed["artifacts"]:
                            out.append(f"{where} uploads an unreviewed file: {entry!r}")
                        elif not re.search(reviewed["artifacts"][entry], runs):
                            out.append(
                                f"{where} uploads {entry!r} without its reviewed producer"
                            )
                elif action not in NON_PUBLISHING_ACTIONS:
                    out.append(f"{where} uses an unreviewed action: {uses}")
            for line in str(step.get("run", "")).splitlines():
                if PUBLISH_COMMAND.search(line) and not any(
                    command in line for command in reviewed["commands"]
                ):
                    out.append(
                        f"{where} runs an unreviewed publishing command: {line.strip()!r}"
                    )
    return out


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def test_the_workflows_are_found():
    names = {w.name for w in WORKFLOWS}
    assert {"pull_request.yaml", "push.yaml"} <= names


@pytest.mark.parametrize("workflow", WORKFLOWS, ids=lambda w: w.name)
def test_no_unreviewed_route_out_of_a_job_that_can_reach_the_data(workflow):
    assert problems(workflow.name, _load(workflow)) == []


def test_the_dataset_building_jobs_are_recognised():
    pull_request = _jobs(_load(WORKFLOW_DIR / "pull_request.yaml"))
    push = _jobs(_load(WORKFLOW_DIR / "push.yaml"))
    docs = _jobs(_load(WORKFLOW_DIR / "docs.yaml"))
    assert reaches_data(pull_request["test"]) and reaches_data(push["test"])
    assert not reaches_data(docs["build-and-deploy"])


def test_every_reviewed_entry_names_a_real_job():
    for name, job_id in REVIEWED:
        assert job_id in _jobs(_load(WORKFLOW_DIR / name)), (name, job_id)


# A clean job that builds the datasets, for the mutation tests below.
DATA_JOB = {
    "env": {"HUGGING_FACE_TOKEN": "${{ secrets.HUGGING_FACE_TOKEN }}"},
    "steps": [
        {"uses": "actions/checkout@v6"},
        {"uses": "astral-sh/setup-uv@v8.1.0"},
        {"run": "uv run --frozen make data"},
    ],
}


def _with_step(*steps: dict) -> list:
    job = copy.deepcopy(DATA_JOB)
    job["steps"] += list(steps)
    return problems("example.yaml", {"jobs": {"build": job}})


def test_the_clean_data_job_passes():
    """Baseline for the mutation tests: each failure below is the added step's."""
    assert reaches_data(DATA_JOB)
    assert _with_step() == []


def _upload(path: str) -> dict:
    return {"uses": "actions/upload-artifact@v7", "with": {"name": "x", "path": path}}


@pytest.mark.parametrize(
    "path",
    [
        "constituency_calibration_log.csv",
        "la_calibration_log.csv",
        "renamed.csv",
        ".",
        "./",
        "policyengine_uk_data/storage/",
        "*.csv",
        "**/*.csv",
        "${{ github.workspace }}",
        "!private\nresults",
        "fit_summary.json\nconstituency_calibration_log.csv",
    ],
)
def test_an_unreviewed_upload_from_a_data_job_fails(path):
    """Sol review of #603: names, directories, globs, expressions, every line."""
    assert _with_step(_upload(path))


@pytest.mark.parametrize(
    "step",
    [
        {"uses": "actions/cache@v4", "with": {"path": "policyengine_uk_data/storage"}},
        {"uses": "actions/upload-pages-artifact@v3", "with": {"path": "."}},
        {"uses": "softprops/action-gh-release@v2", "with": {"files": "*.csv"}},
        {"run": "gh release upload v1 constituency_calibration_log.csv"},
        {"run": "curl -T la_calibration_log.csv https://example.org/"},
        {"run": "aws s3 cp . s3://bucket --recursive"},
        {"run": "git push origin HEAD:logs"},
    ],
)
def test_other_routes_out_of_a_data_job_fail(step):
    assert _with_step(step)


def test_a_reusable_workflow_in_a_data_job_fails():
    job = copy.deepcopy(DATA_JOB)
    job["uses"] = "org/repo/.github/workflows/publish.yaml@main"
    assert problems("example.yaml", {"jobs": {"build": job}})


def test_downloading_artifacts_makes_a_job_a_data_job():
    job = {
        "steps": [
            {"uses": "actions/download-artifact@v8", "with": {"path": "logs"}},
            {
                "uses": "JamesIves/github-pages-deploy-action@v4",
                "with": {"folder": "logs"},
            },
        ]
    }
    assert problems("example.yaml", {"jobs": {"publish": job}})


def test_a_job_without_data_may_publish():
    """The docs deploy holds nothing built from the microdata."""
    job = {
        "steps": [
            {"uses": "actions/checkout@v6"},
            {
                "uses": "JamesIves/github-pages-deploy-action@v4",
                "with": {"folder": "docs"},
            },
        ]
    }
    assert problems("example.yaml", {"jobs": {"docs": job}}) == []


def test_a_reviewed_file_needs_its_producer(monkeypatch):
    monkeypatch.setitem(
        REVIEWED,
        ("example.yaml", "build"),
        {
            "artifacts": {"summary.json": r"summarise \S+ --out summary\.json"},
            "commands": (),
        },
    )
    produce = {"run": "python -m tool summarise log.csv --out summary.json"}
    assert _with_step(_upload("summary.json"))
    assert _with_step(produce, _upload("summary.json")) == []
    assert _with_step(produce, _upload("summary.json\nlog.csv"))
