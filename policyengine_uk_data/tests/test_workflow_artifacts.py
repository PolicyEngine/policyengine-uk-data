"""A regression check on the routes out of CI jobs that can reach the microdata.

The raw calibration logs hold each target's weighted estimate, nationally and
for every constituency and local authority, with no small-cell control. A
target supported by a few households, divided by a published count, could
give one household's value back, and this repository's Actions artifacts are
public.

What the check does:

- A job **can reach the data** when it, or its workflow, uses any secret
  other than the GitHub token (presumed a data credential whatever it is
  called), passes secrets on, signs in to Google Cloud, or downloads
  artifacts.
- In such a job, every action must be reviewed for the inputs it is given
  (``REVIEWED_ACTIONS``), so a set-up action can't be pointed at the
  workspace. Every artifact upload must name literal files reviewed for that
  job in ``REVIEWED``: no directories, globs, exclusions or expressions.
  Every run line that matches ``PUBLISH_COMMAND`` must be, in full, a command
  reviewed for that job.
- Jobs that can't reach the data (docs, changelog) may publish.

What it does not do (Sol re-review of #603): it recognises reviewed workflow
shapes through secret, action and command names and explicit artifact
downloads. It restricts declared actions, artifact path strings and known
literal publishing commands. It does not establish data provenance or inspect
script, composite, reusable-workflow, container or service behaviour, shell
expansion, step summaries or job outputs. Producer patterns establish text
presence in an earlier run step, not execution, ordering at run time, that
the path is a regular file, or that its contents are disclosure-controlled.
Changes to any of these routes need human review.
"""

from __future__ import annotations

import copy
import re
from pathlib import Path

import pytest
import yaml

WORKFLOW_DIR = Path(__file__).parents[2] / ".github" / "workflows"
WORKFLOWS = sorted(WORKFLOW_DIR.glob("*.y*ml"))

# Any secret but these is presumed to reach the data.
NON_DATA_SECRETS = ("secrets.GITHUB_TOKEN",)
_SECRET = re.compile(r"secrets\.[A-Za-z_][A-Za-z0-9_]*")
DATA_ACTIONS = ("google-github-actions/auth",)

# Actions that publish nothing when given only these inputs. setup-uv
# caches Python packages, not datasets; its cache-path inputs are not
# reviewed, so pointing its cache at the workspace fails (Sol re-review of
# #603, finding 1). ``enable-cache`` may only turn caching off.
REVIEWED_ACTIONS = {
    "actions/checkout": {
        "fetch-depth",
        "sparse-checkout",
        "sparse-checkout-cone-mode",
        "token",
        "ref",
        "repository",
        "path",
        "persist-credentials",
    },
    "actions/setup-python": {"python-version"},
    "actions/setup-node": {"node-version"},
    "astral-sh/setup-uv": {"version", "python-version", "enable-cache"},
    "google-github-actions/auth": {"service_account", "workload_identity_provider"},
    "actions/create-github-app-token": {"app-id", "private-key"},
    "actions/download-artifact": {"name", "path", "pattern", "merge-multiple"},
}

# Reviewed routes out of each job that can reach the data:
# (workflow file, job id) -> artifact paths it may upload, each mapped to a
# pattern an earlier run step must contain to produce it; the publishing
# commands it may run, each a whole line; and actions reviewed for this job
# only, with their exact inputs.
REVIEWED = {
    ("push.yaml", "test"): {
        "artifacts": {},
        # Private Hugging Face and GCS datasets (upload_completed_datasets.py,
        # the data controller's destinations), then the release tag.
        "commands": ("uv run --frozen make upload", ".github/publish-git-tag.sh"),
        "actions": {},
    },
    ("versioning.yaml", "Versioning"): {
        "artifacts": {},
        "commands": (),
        # Commits the version bump. Its secrets are the GitHub app's, and the
        # job builds no data and downloads nothing.
        "actions": {
            "EndBug/add-and-commit": {"add": ".", "message": "Update package version"}
        },
    },
}

PUBLISH_COMMAND = re.compile(
    r"make upload|publish-git-tag|\bgh (release|gist|api|pr comment|issue comment)\b"
    r"|\bgit push\b|\bcurl\b.*(\s-T\b|--upload-file|\s-F\b|--data|\s-d\b|-X\s*(POST|PUT|PATCH))"
    r"|\baws s3\b|\bgsutil\b|\bgcloud storage\b|huggingface-cli|\bhf upload\b|\btwine\b"
)
_NOT_LITERAL = re.compile(r"[*?\[\]{}!]|\$\{\{")
_EMPTY = {"artifacts": {}, "commands": (), "actions": {}}


def _jobs(workflow: dict) -> dict:
    return workflow.get("jobs") or {}


def _triggers(workflow: dict) -> dict:
    # YAML reads a bare ``on:`` key as True.
    on = workflow.get("on", workflow.get(True)) or {}
    return on if isinstance(on, dict) else {}


def _uses_data_secret(text: str) -> bool:
    return any(s not in NON_DATA_SECRETS for s in _SECRET.findall(text))


def reaches_data(job: dict, workflow: dict | None = None) -> bool:
    workflow = workflow or {}
    shared = yaml.safe_dump(
        {"env": workflow.get("env"), "defaults": workflow.get("defaults")}
    )
    call = _triggers(workflow).get("workflow_call") or {}
    if _uses_data_secret(shared) or (isinstance(call, dict) and call.get("secrets")):
        return True
    text = yaml.safe_dump(job)
    if "secrets" in job or _uses_data_secret(text):
        return True
    return any(
        str(step.get("uses", "")).split("@", 1)[0]
        in DATA_ACTIONS + ("actions/download-artifact",)
        for step in job.get("steps", [])
    )


def _action(uses: str) -> str:
    return uses.split("@", 1)[0]


def logical_lines(script: str) -> list:
    """Shell lines with backslash continuations joined."""
    out, pending = [], ""
    for line in script.splitlines():
        if line.rstrip().endswith("\\"):
            pending += line.rstrip()[:-1] + " "
            continue
        out.append((pending + line).strip())
        pending = ""
    if pending:
        out.append(pending.strip())
    return [line for line in out if line]


def _code(script: str) -> str:
    """A run script without comment lines."""
    return "\n".join(
        line for line in logical_lines(script) if not line.lstrip().startswith("#")
    )


def problems(name: str, workflow: dict) -> list:
    """Every unreviewed route out of a job of ``workflow`` that can reach the data."""
    out = []
    for job_id, job in _jobs(workflow).items():
        if not reaches_data(job, workflow):
            continue
        where = f"{name} job {job_id}"
        reviewed = {**_EMPTY, **REVIEWED.get((name, job_id), {})}
        if "uses" in job:
            out.append(f"{where} calls a reusable workflow ({job['uses']})")
        earlier = []  # run steps before the current one, comments removed
        for step in job.get("steps", []):
            uses = str(step.get("uses", ""))
            inputs = step.get("with") or {}
            if uses:
                action = _action(uses)
                if action == "actions/upload-artifact":
                    path = str(inputs.get("path", ""))
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
                        elif not re.search(
                            reviewed["artifacts"][entry], "\n".join(earlier)
                        ):
                            out.append(
                                f"{where} uploads {entry!r} without its reviewed "
                                "producer in an earlier run step"
                            )
                elif action in reviewed["actions"]:
                    if dict(inputs) != reviewed["actions"][action]:
                        out.append(
                            f"{where} gives {action} inputs other than the reviewed ones"
                        )
                elif action in REVIEWED_ACTIONS:
                    extra = set(inputs) - REVIEWED_ACTIONS[action]
                    if extra:
                        out.append(
                            f"{where} gives {action} unreviewed inputs: "
                            + ", ".join(sorted(extra))
                        )
                    if action == "astral-sh/setup-uv" and inputs.get(
                        "enable-cache", False
                    ) not in (False, "false"):
                        out.append(f"{where} turns setup-uv's cache on explicitly")
                else:
                    out.append(f"{where} uses an unreviewed action: {uses}")
            script = str(step.get("run", ""))
            for line in logical_lines(script):
                if PUBLISH_COMMAND.search(line) and line not in reviewed["commands"]:
                    out.append(
                        f"{where} runs an unreviewed publishing command: {line!r}"
                    )
            if script and step.get("if") not in (False, "false", "${{ false }}"):
                earlier.append(_code(script))
    return out


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def test_the_workflows_are_found():
    names = {w.name for w in WORKFLOWS}
    assert {"pull_request.yaml", "push.yaml"} <= names


@pytest.mark.parametrize("workflow", WORKFLOWS, ids=lambda w: w.name)
def test_no_unreviewed_route_out_of_a_job_that_can_reach_the_data(workflow):
    assert problems(workflow.name, _load(workflow)) == []


def test_the_jobs_that_can_reach_the_data_are_recognised():
    def job(name, job_id):
        workflow = _load(WORKFLOW_DIR / name)
        return reaches_data(_jobs(workflow)[job_id], workflow)

    assert job("pull_request.yaml", "test") and job("push.yaml", "test")
    assert job("versioning.yaml", "Versioning")  # a GitHub app's secrets
    assert not job("docs.yaml", "build-and-deploy")
    assert not job("push.yaml", "gate")  # the GitHub token only


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


def _with_step(*steps: dict, workflow_extra=None) -> list:
    job = copy.deepcopy(DATA_JOB)
    job["steps"] += list(steps)
    return problems("example.yaml", {**(workflow_extra or {}), "jobs": {"build": job}})


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
        {"run": "curl \\\n  -T la_calibration_log.csv \\\n  https://example.org/"},
    ],
)
def test_other_routes_out_of_a_data_job_fail(step):
    assert _with_step(step)


@pytest.mark.parametrize(
    "inputs",
    [
        {"cache-local-path": ".", "enable-cache": True, "cache-suffix": "x"},
        {"cache-local-path": "."},
        {"enable-cache": True},
        {"cache-dependency-glob": "**/*.csv"},
    ],
)
def test_a_set_up_action_pointed_at_the_workspace_fails(inputs):
    """Sol re-review of #603, finding 1: setup-uv's post step saves its cache."""
    assert _with_step({"uses": "astral-sh/setup-uv@v8.1.0", "with": inputs})


def test_turning_setup_uvs_cache_off_passes():
    step = {"uses": "astral-sh/setup-uv@v8.1.0", "with": {"enable-cache": False}}
    assert _with_step(step) == []


@pytest.mark.parametrize(
    "script",
    [
        "uv run --frozen make upload; curl -T constituency_calibration_log.csv https://x/",
        "uv run --frozen make upload && gh release upload v1 log.csv",
        "# uv run --frozen make upload\ncurl -T log.csv https://x/",
    ],
)
def test_a_reviewed_command_does_not_excuse_another_on_its_line(script, monkeypatch):
    """Sol re-review of #603, finding 2: a reviewed command is a whole line."""
    monkeypatch.setitem(
        REVIEWED,
        ("example.yaml", "build"),
        {**_EMPTY, "commands": ("uv run --frozen make upload",)},
    )
    assert _with_step({"run": script})
    assert _with_step({"run": "uv run --frozen make upload"}) == []


def test_a_workflow_level_credential_makes_every_job_a_data_job():
    """Sol re-review of #603, finding 3."""
    job = {
        "steps": [
            {"run": "uv run --frozen make data"},
            _upload("constituency_calibration_log.csv"),
        ]
    }
    workflow = {
        "env": {"HUGGING_FACE_TOKEN": "${{ secrets.HUGGING_FACE_TOKEN }}"},
        "jobs": {"sneaky": job},
    }
    assert reaches_data(job, workflow)
    assert problems("example.yaml", workflow)


@pytest.mark.parametrize(
    "job",
    [
        {"env": {"GH_TOKEN": "${{ secrets.ANY_NAME }}"}, "steps": []},
        {"uses": "org/repo/.github/workflows/build.yaml@main", "secrets": "inherit"},
        {"steps": [{"uses": "google-github-actions/auth@v3"}]},
        {"steps": [{"uses": "actions/download-artifact@v8"}]},
    ],
)
def test_any_secret_or_data_route_makes_a_data_job(job):
    """A secret's name doesn't say what it can read."""
    assert reaches_data(job)


def test_a_job_with_only_the_github_token_is_not_a_data_job():
    job = {"env": {"GH_TOKEN": "${{ github.token }}"}, "steps": []}
    assert not reaches_data(job)
    job = {"env": {"GH_TOKEN": "${{ secrets.GITHUB_TOKEN }}"}, "steps": []}
    assert not reaches_data(job)


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


def test_a_reviewed_action_needs_its_reviewed_inputs():
    data = copy.deepcopy(_load(WORKFLOW_DIR / "versioning.yaml"))
    for step in _jobs(data)["Versioning"]["steps"]:
        if str(step.get("uses", "")).startswith("EndBug/add-and-commit"):
            step["with"]["add"] = "constituency_calibration_log.csv"
    assert problems("versioning.yaml", data)


@pytest.mark.parametrize(
    "producer",
    [
        None,  # no producer at all
        {"run": "# python -m tool summarise log.csv --out summary.json"},
        {"run": "python -m tool summarise log.csv --out summary.json", "if": False},
    ],
)
def test_a_reviewed_file_needs_its_producer_in_an_earlier_step(producer, monkeypatch):
    monkeypatch.setitem(
        REVIEWED,
        ("example.yaml", "build"),
        {**_EMPTY, "artifacts": {"summary.json": r"summarise \S+ --out summary\.json"}},
    )
    steps = [producer] if producer else []
    assert _with_step(*steps, _upload("summary.json"))
    # The producer after the upload doesn't count either.
    produce = {"run": "python -m tool summarise log.csv --out summary.json"}
    assert _with_step(_upload("summary.json"), produce)
    assert _with_step(produce, _upload("summary.json")) == []
    assert _with_step(produce, _upload("summary.json\nlog.csv"))
