"""The release build asks policyengine.py to qualify each published release.

policyengine.py's certify-uk-release workflow qualifies a new data release
against the latest model before it can be promoted, triggered by a
``policyengine-uk-data-release`` repository dispatch whose payload names the
version. The dispatch must come only after the data uploaded and the tag was
published, carry exactly the released version, and use a token scoped to
policyengine.py alone.
"""

import os
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
PUSH_WORKFLOW = ROOT / ".github/workflows/push.yaml"
TOKEN_STEP = "Generate a token for policyengine.py"
DISPATCH_STEP = "Ask policyengine.py to qualify this release"


def _steps():
    job = yaml.safe_load(PUSH_WORKFLOW.read_text())["jobs"]["test"]
    return [step.get("name") for step in job["steps"]], {
        step.get("name"): step for step in job["steps"]
    }


def test_qualification_is_requested_only_after_the_release_is_published():
    order, by_name = _steps()
    assert (
        order.index("Upload data")
        < order.index("Publish a git tag")
        < order.index(TOKEN_STEP)
        < order.index(DISPATCH_STEP)
    )
    # No condition: the default success() means every earlier step, the
    # upload included, succeeded.
    assert "if" not in by_name[TOKEN_STEP]
    assert "if" not in by_name[DISPATCH_STEP]
    assert not by_name[DISPATCH_STEP].get("continue-on-error", False)


def test_the_token_reaches_policyengine_py_only():
    _, by_name = _steps()
    token = by_name[TOKEN_STEP]
    assert token["uses"].startswith("actions/create-github-app-token@")
    assert token["with"]["owner"] == "PolicyEngine"
    assert token["with"]["repositories"] == "policyengine.py"
    dispatch = by_name[DISPATCH_STEP]
    assert dispatch["env"] == {
        "GH_TOKEN": "${{ steps.qualification-token.outputs.token }}"
    }
    assert "${{" not in dispatch["run"]


def _run_dispatch(tmp_path, pyproject_version):
    """Run the dispatch step's script with a fake gh that records its arguments."""

    _, by_name = _steps()
    repo = tmp_path / "repo"
    (repo / ".github").mkdir(parents=True)
    (repo / ".github/fetch_version.py").write_text(
        (ROOT / ".github/fetch_version.py").read_text()
    )
    (repo / "pyproject.toml").write_text(
        f'[project]\nversion = "{pyproject_version}"\n'
        if pyproject_version
        else "[project]\n"
    )
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "gh.log"
    (bin_dir / "gh").write_text(f'#!/usr/bin/env bash\nprintf "%s\\n" "$@" > "{log}"\n')
    (bin_dir / "gh").chmod(0o755)
    result = subprocess.run(
        ["bash", "-e", "-c", by_name[DISPATCH_STEP]["run"]],
        cwd=repo,
        env={"PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}"},
        capture_output=True,
        text=True,
    )
    return result, log.read_text().splitlines() if log.exists() else None


def test_the_dispatch_names_the_released_version(tmp_path):
    result, arguments = _run_dispatch(tmp_path, "1.59.0")
    assert result.returncode == 0, result.stderr
    assert arguments == [
        "api",
        "--method",
        "POST",
        "repos/PolicyEngine/policyengine.py/dispatches",
        "-f",
        "event_type=policyengine-uk-data-release",
        "-f",
        "client_payload[version]=1.59.0",
    ]


def test_no_dispatch_without_a_version(tmp_path):
    result, arguments = _run_dispatch(tmp_path, None)
    assert result.returncode != 0
    assert arguments is None
