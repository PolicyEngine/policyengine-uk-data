"""The final calibration fit summary and the no-worsening comparison (utils/fit_report.py)."""

import json

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from policyengine_uk_data.utils.fit_report import (
    MIN_RECORDS,
    FitPolicy,
    compare,
    load_policy,
    main,
    summarise,
)

POLICY = FitPolicy(share_drop_allowance=0.02, target_flag_threshold=0.2)


def _log(national: dict, local=(), epochs=(0, 30), records=100):
    """A calibration log: national {metric: (estimate, target)}, local [(est, target)]."""
    rows = []
    for epoch in epochs:
        for metric, (estimate, target) in national.items():
            rows.append(("UK", 0, metric, estimate, target, epoch, records))
        for i, (estimate, target) in enumerate(local):
            rows.append(
                (f"area {i}", f"E{i}", f"local/{i}", estimate, target, epoch, np.nan)
            )
    return pd.DataFrame(
        rows,
        columns=["name", "index", "metric", "estimate", "target", "epoch", "records"],
    )


LOCAL = [(100, 100)] * 10


def test_summarise_reads_the_last_epoch_and_drops_zero_targets():
    log = _log(
        {"a": (105, 100), "b": (150, 100), "zero": (5, 0)}, local=[(9, 10), (20, 10)]
    )
    log.loc[log.epoch == 0, "estimate"] = 1  # earlier epochs are ignored
    s = summarise(log)
    assert s["epoch"] == 30
    assert s["national"]["a"] == {"within": True, "error": pytest.approx(0.05)}
    assert s["national"]["b"] == {"within": False, "error": pytest.approx(0.5)}
    assert "zero" not in s["national"]
    assert s["local"] == {"E0|local/0": True, "E1|local/1": False}


def test_a_target_with_few_records_has_no_published_error():
    """Sol review of #599, P1: an estimate from a few records is not published."""
    s = summarise(_log({"rare": (123.45, 100)}, local=LOCAL, records=MIN_RECORDS - 1))
    assert s["national"]["rare"] == {"within": False, "error": None}
    assert "123.45" not in json.dumps(s)
    # A log without the records column publishes no errors at all.
    s = summarise(_log({"a": (101, 100)}, local=LOCAL).drop(columns="records"))
    assert s["national"]["a"]["error"] is None
    # Local cells give only within 10% or not.
    s = summarise(_log({"a": (101, 100)}, local=[(123.45, 100)]))
    assert s["local"] == {"E0|local/0": False}


def test_a_malformed_log_is_refused():
    with pytest.raises(ValueError, match="no rows"):
        summarise(_log({}, epochs=()))
    with pytest.raises(ValueError, match="non-finite"):
        summarise(_log({"a": (np.inf, 100)}))
    with pytest.raises(ValueError, match="repeats"):
        summarise(pd.concat([_log({"a": (1, 1)}), _log({"a": (2, 1)})]))


def _summary(national_errors, local=LOCAL, epochs=(0, 30)):
    return summarise(
        _log(
            {f"t{i}": (100 * (1 + e), 100) for i, e in enumerate(national_errors)},
            local,
            epochs,
        )
    )


def test_a_drop_beyond_the_allowance_fails_and_within_it_passes():
    base = _summary([0.0] * 100)
    head = _summary([0.5, 0.5, 0.5] + [0.0] * 97)
    result = compare(base, head, POLICY)
    assert result["status"] == "fail"
    assert (
        "national targets within 10% falls from 100.0% to 97.0%"
        in result["failures"][0]
    )
    assert [f["target"] for f in result["flags"]] == ["t0", "t1", "t2"]
    assert compare(base, _summary([0.5] + [0.0] * 99), POLICY)["status"] == "pass"


def test_added_targets_do_not_count_against_the_existing_ones():
    """Sol review of #599, P2: 100 unchanged targets plus 3 poor new ones pass."""
    base = _summary([0.0] * 100)
    head = summarise(
        _log(
            {
                **{f"t{i}": (100, 100) for i in range(100)},
                **{f"new{i}": (200, 100) for i in range(3)},
            },
            LOCAL,
        )
    )
    result = compare(base, head, POLICY)
    assert result["status"] == "pass"
    assert result["summary"]["national"]["compared"] == 100
    assert "3 national targets are new: new0, new1, new2" in result["notes"]


@pytest.mark.parametrize(
    "head_log",
    [
        _log({"a": (1, 1)}, local=()),  # only national rows
        _log({}, local=LOCAL),  # only local rows
        _log({"a": (1, 0)}, local=[(1, 0)]),  # only zero targets
    ],
)
def test_an_empty_target_group_fails(head_log):
    """Sol review of #599, P2: empty groups used to give NaN shares and pass."""
    result = compare(_summary([0.0] * 10), summarise(head_log), POLICY)
    assert result["status"] == "fail"
    assert "has no" in result["failures"][0]


def test_an_incomplete_last_epoch_fails():
    log = _log({f"t{i}": (100, 100) for i in range(10)}, LOCAL)
    log = log[~((log.epoch == 30) & (log.metric == "t9"))]
    result = compare(_summary([0.0] * 10), summarise(log), POLICY)
    assert result["status"] == "fail" and "incomplete" in result["failures"][0]


def test_builds_of_different_epochs_cannot_be_compared():
    base = _summary([0.0], epochs=(0, 30))
    assert (
        compare(base, _summary([0.0], epochs=(0, 510, 511)), POLICY)["status"] == "fail"
    )
    # 32 epochs log last at 30, or at 31 once the final epoch is saved (#594).
    result = compare(base, _summary([0.0], epochs=(0, 30, 31)), POLICY)
    assert result["status"] == "pass"
    assert "saves the final epoch" in result["notes"][0]


errors = st.lists(st.floats(0, 3), min_size=1, max_size=40)


@settings(max_examples=200, deadline=None)
@given(national=errors, local=errors)
def test_a_build_compared_with_itself_passes_with_nothing_flagged(national, local):
    s = _summary(national, local=[(100 * (1 + e), 100) for e in local])
    result = compare(s, s, POLICY)
    assert result["status"] == "pass" and result["flags"] == []
    assert all(row["change"] == 0 for row in result["summary"].values())


@settings(max_examples=200, deadline=None)
@given(base=errors, data=st.data())
def test_improving_every_target_never_fails_and_changes_are_antisymmetric(base, data):
    better = [data.draw(st.floats(0, e)) for e in base]
    forward = compare(_summary(base), _summary(better), POLICY)
    assert forward["status"] == "pass" and forward["flags"] == []
    backward = compare(_summary(better), _summary(base), POLICY)
    for name in ("national", "local"):
        assert forward["summary"][name]["change"] == pytest.approx(
            -backward["summary"][name]["change"]
        )


def test_the_shipped_policy_loads():
    policy = load_policy()
    assert policy.share_drop_allowance == 0.02
    assert policy.target_flag_threshold == 0.2


def test_the_commands_summarise_and_exit_one_on_a_failure(tmp_path):
    base_log, head_log = tmp_path / "base.csv", tmp_path / "head.csv"
    targets = {f"t{i}": (100, 100) for i in range(50)}
    _log(targets, LOCAL).to_csv(base_log, index=False)
    _log(dict(targets, **{f"t{i}": (200, 100) for i in range(5)}), LOCAL).to_csv(
        head_log, index=False
    )
    for log in (base_log, head_log):
        assert (
            main(["summarise", str(log), "--out", str(log.with_suffix(".json"))]) == 0
        )
    out = tmp_path / "fit.md"
    code = main(
        [
            "compare",
            str(tmp_path / "base.json"),
            str(tmp_path / "head.json"),
            "--markdown",
            str(out),
        ]
    )
    assert code == 1
    assert "## Calibration fit against the base build: fail" in out.read_text()
    assert (
        main(["compare", str(tmp_path / "base.json"), str(tmp_path / "base.json")]) == 0
    )


def test_the_pull_request_workflow_uploads_only_summaries():
    """The datasets are licensed microdata, and raw calibration logs stay on the runner."""
    from pathlib import Path

    import yaml

    workflow = Path(__file__).parents[2] / ".github" / "workflows" / "pull_request.yaml"
    jobs = yaml.safe_load(workflow.read_text())["jobs"]
    uploads = sorted(
        step["with"]["path"]
        for job in jobs.values()
        for step in job.get("steps", [])
        if step.get("uses", "").startswith("actions/upload-artifact")
    )
    assert uploads == ["base_fit_summary.json", "fit_summary.json"]
    for job in jobs.values():
        for step in job.get("steps", []):
            assert not step.get("uses", "").startswith("actions/cache"), step
    assert jobs["fit-comparison"]["needs"] == ["test", "base-build"]
    post = next(
        s
        for s in jobs["fit-comparison"]["steps"]
        if s.get("name", "").startswith("Post")
    )
    assert "if [ ! -s fit.md ]" in post["run"]
