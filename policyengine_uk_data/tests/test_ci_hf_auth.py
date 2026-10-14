"""CI reaches Hugging Face only through short-lived tokens.

Each step that downloads or uploads runs under `.github/with-hf-token.sh`,
which exchanges the job's GitHub OIDC token for a one-hour Hugging Face token
(Trusted Publishers). The release job (`push.yaml`) exchanges for a token that
can write to the private data repo only; pull requests (`pull_request.yaml`)
exchange for the read-only token of policyengine-auto's CI/CD identity. These
tests stop a stored Hugging Face secret from coming back, pin the wrapped
commands exactly, keep pull requests off write tokens, and run the helper
itself against a fake exchange.
"""

import json
import os
import re
import stat
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from policyengine_uk_data.utils.hf_destinations import PRIVATE_REPO

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = REPO_ROOT / ".github" / "workflows"
TOKEN_HELPER = REPO_ROOT / ".github" / "with-hf-token.sh"
HELPER = ".github/with-hf-token.sh"
CI_IDENTITY = "policyengine-auto"

# Every step that talks to Hugging Face: (workflow, job, step) -> (exact
# command, token resource).
HF_STEPS = {
    ("push.yaml", "test", "Download data inputs"): (
        f"{HELPER} uv run --frozen make download",
        PRIVATE_REPO,
    ),
    ("push.yaml", "test", "Upload data"): (
        f"{HELPER} uv run --frozen make upload",
        PRIVATE_REPO,
    ),
    ("pull_request.yaml", "test", "Download data inputs"): (
        f"{HELPER} uv run --frozen make download",
        CI_IDENTITY,
    ),
}
HF_COMMANDS = ("make download", "make upload")
TOKEN_VARIABLES = ("HUGGING_FACE_TOKEN", "HF_TOKEN")


def _workflow_files() -> list[Path]:
    return sorted([*WORKFLOWS.glob("*.yaml"), *WORKFLOWS.glob("*.yml")])


def _load(name: str) -> dict:
    return yaml.safe_load((WORKFLOWS / name).read_text())


def _all_steps():
    for path in _workflow_files():
        workflow = yaml.safe_load(path.read_text())
        for job_name, job in workflow.get("jobs", {}).items():
            for step in job.get("steps", []):
                yield path.name, job_name, job, step


def _env_scopes():
    for path in _workflow_files():
        workflow = yaml.safe_load(path.read_text())
        yield path.name, "workflow", workflow.get("env", {})
        for job_name, job in workflow.get("jobs", {}).items():
            yield path.name, job_name, job.get("env", {})
            for step in job.get("steps", []):
                yield path.name, f"{job_name}/{step.get('name')}", step.get("env", {})


def test_no_workflow_reads_a_hugging_face_secret():
    pattern = re.compile(r"secrets\.\w*(HF|HUGGING)\w*", re.IGNORECASE)
    for path in _workflow_files():
        assert not pattern.search(path.read_text()), path.name


def test_no_workflow_sets_a_static_hugging_face_token():
    for workflow, scope, env in _env_scopes():
        for variable in TOKEN_VARIABLES:
            assert variable not in (env or {}), f"{workflow} {scope} sets {variable}"


@pytest.mark.parametrize("workflow,job,step", sorted(HF_STEPS))
def test_each_hugging_face_step_runs_exactly_its_wrapped_command(workflow, job, step):
    command, resource = HF_STEPS[(workflow, job, step)]
    (match,) = [s for s in _load(workflow)["jobs"][job]["steps"] if s.get("name") == step]
    assert match["run"].strip() == command
    assert match["env"]["HF_OIDC_RESOURCE"] == resource


def test_hugging_face_commands_run_only_in_the_listed_steps():
    """A second, unwrapped `make upload` line would get no token."""
    found = set()
    for workflow, job_name, _, step in _all_steps():
        run = step.get("run", "")
        if any(command in run for command in HF_COMMANDS):
            found.add((workflow, job_name, step.get("name")))
    assert found == set(HF_STEPS)


def test_jobs_that_exchange_can_request_an_oidc_token():
    for workflow, job_name, job, step in _all_steps():
        if HELPER in step.get("run", ""):
            assert job.get("permissions", {}).get("id-token") == "write", (
                f"{workflow} {job_name}"
            )


def test_pull_requests_only_get_the_read_only_identity():
    """A repo resource yields a write token; pull requests must never get one."""
    for workflow, scope, env in _env_scopes():
        if workflow == "pull_request.yaml" and "HF_OIDC_RESOURCE" in (env or {}):
            assert env["HF_OIDC_RESOURCE"] == CI_IDENTITY, scope


def test_release_token_is_scoped_to_the_repo_the_upload_writes_to():
    for workflow, scope, env in _env_scopes():
        if workflow == "push.yaml" and "HF_OIDC_RESOURCE" in (env or {}):
            assert env["HF_OIDC_RESOURCE"] == PRIVATE_REPO, scope


def test_token_helper_is_executable():
    assert TOKEN_HELPER.is_file()
    assert os.access(TOKEN_HELPER, os.X_OK)


# The helper, run against a fake exchange. The fake `uvx` (or `pipx`) records
# its arguments and the resource it saw, then prints $FAKE_TOKEN and exits
# with $FAKE_EXIT. The child prints what it received as one JSON line.

FAKE_EXCHANGE = """#!/bin/sh
printf '%s\\n' "$*" > "$CALLS/{name}"
printf '%s' "${{HF_OIDC_RESOURCE:-}}" > "$CALLS/{name}.resource"
printf '%s' "$FAKE_TOKEN"
exit "${{FAKE_EXIT:-0}}"
"""

CHILD = (
    "import json, os, sys;"
    "print(json.dumps({'argv': sys.argv[1:], 'stdin': sys.stdin.read(),"
    " 'env': {k: os.environ.get(k) for k in ("
    "'HUGGING_FACE_TOKEN', 'HF_TOKEN', 'HF_OIDC_RESOURCE', 'HF_OIDC_OPTIONAL')}}));"
    "open(os.environ['CALLS'] + '/child', 'w').close();"
    "sys.exit(int(os.environ.get('CHILD_EXIT', '0')))"
)


@pytest.fixture
def run_helper(tmp_path):
    calls = tmp_path / "calls"
    calls.mkdir()

    def run(*args, exchange="uvx", stdin="", **env):
        bin_dir = tmp_path / f"bin-{exchange}"
        bin_dir.mkdir(exist_ok=True)
        if exchange:
            fake = bin_dir / exchange
            fake.write_text(FAKE_EXCHANGE.format(name=exchange))
            fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
        base = {
            "PATH": f"{bin_dir}:/usr/bin:/bin",
            "CALLS": str(calls),
            "FAKE_TOKEN": "hf_jwt_fake",
            "HF_OIDC_RESOURCE": CI_IDENTITY,
            "ACTIONS_ID_TOKEN_REQUEST_URL": "https://example.invalid/oidc",
        }
        base.update({k: v for k, v in env.items() if v is not None})
        for key in [k for k, v in env.items() if v is None]:
            base.pop(key, None)
        command = args or (sys.executable, "-c", CHILD)
        return subprocess.run(
            ["bash", str(TOKEN_HELPER), *command],
            input=stdin,
            capture_output=True,
            text=True,
            env=base,
        )

    run.calls = calls
    return run


def _child_output(result):
    lines = [l for l in result.stdout.splitlines() if l.startswith("{")]
    assert len(lines) == 1, result.stdout
    return json.loads(lines[0])


def test_helper_passes_the_token_arguments_stdin_and_exit_status(run_helper):
    result = run_helper(
        sys.executable, "-c", CHILD, "two words", "it's", "$HOME",
        stdin="line one\nline two\n", CHILD_EXIT="3",
    )
    assert result.returncode == 3
    out = _child_output(result)
    assert out["argv"] == ["two words", "it's", "$HOME"]
    assert out["stdin"] == "line one\nline two\n"
    assert out["env"]["HUGGING_FACE_TOKEN"] == "hf_jwt_fake"
    assert out["env"]["HF_TOKEN"] == "hf_jwt_fake"
    # huggingface_hub >= 1.19 would run its own exchange if this were set.
    assert out["env"]["HF_OIDC_RESOURCE"] is None


def test_helper_exchanges_for_the_named_resource_with_a_pinned_cli(run_helper):
    result = run_helper(HF_OIDC_RESOURCE=PRIVATE_REPO)
    assert result.returncode == 0, result.stderr
    calls = run_helper.calls
    assert (calls / "uvx").read_text().split() == [
        "--from", "huggingface_hub==2.2.0", "hf", "auth", "token",
    ]
    assert (calls / "uvx.resource").read_text() == PRIVATE_REPO


def test_helper_masks_the_token_before_the_command_runs(run_helper):
    result = run_helper()
    lines = result.stdout.splitlines()
    assert lines[0] == "::add-mask::hf_jwt_fake"
    assert lines[1].startswith("{")


def test_helper_falls_back_to_pipx_without_uvx(run_helper):
    result = run_helper(exchange="pipx")
    assert result.returncode == 0, result.stderr
    assert (run_helper.calls / "pipx").read_text().split() == [
        "run", "--spec", "huggingface_hub==2.2.0", "hf", "auth", "token",
    ]
    assert _child_output(result)["env"]["HUGGING_FACE_TOKEN"] == "hf_jwt_fake"


@pytest.mark.parametrize(
    "env",
    [
        {"FAKE_EXIT": "1"},
        {"FAKE_EXIT": "1", "FAKE_TOKEN": "hf_printed_before_failing"},
        {"FAKE_TOKEN": ""},
        {"HF_OIDC_RESOURCE": None},
        {"HF_OIDC_RESOURCE": ""},
    ],
    ids=["exchange-fails", "fails-after-output", "empty-token", "no-resource", "empty-resource"],
)
def test_helper_never_runs_the_command_without_a_token(run_helper, env):
    result = run_helper(**env)
    assert result.returncode != 0
    assert not (run_helper.calls / "child").exists()
    assert "::add-mask::" not in result.stdout


def test_helper_needs_a_command(run_helper):
    result = subprocess.run(
        ["bash", str(TOKEN_HELPER)],
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin", "HF_OIDC_RESOURCE": CI_IDENTITY},
    )
    assert result.returncode == 2


def test_helper_without_oidc_fails_unless_told_the_token_is_optional(run_helper):
    # A fork pull request has no OIDC request URL. By default the helper still
    # attempts the exchange, which fails there, rather than skip silently.
    strict = run_helper(ACTIONS_ID_TOKEN_REQUEST_URL=None, FAKE_EXIT="1")
    assert strict.returncode != 0
    assert (run_helper.calls / "uvx").exists()
    assert not (run_helper.calls / "child").exists()

    (run_helper.calls / "uvx").unlink()
    optional = run_helper(ACTIONS_ID_TOKEN_REQUEST_URL=None, HF_OIDC_OPTIONAL="1")
    assert optional.returncode == 0, optional.stderr
    assert not (run_helper.calls / "uvx").exists()
    # No token, no resource and no flag reach the command.
    assert all(value is None for value in _child_output(optional)["env"].values())
