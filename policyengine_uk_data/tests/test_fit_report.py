"""The final calibration fit and the no-worsening comparison (utils/fit_report.py)."""

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from policyengine_uk_data.utils.fit_report import (
    FitPolicy,
    compare_fits,
    final_fit,
    load_policy,
    main,
)

POLICY = FitPolicy(share_drop_allowance=0.02, target_flag_threshold=0.2)


def _log(national: dict, local=(), epochs=(0, 30)):
    """A calibration log: national {metric: (estimate, target)}, local [(est, target)]."""
    rows = []
    for epoch in epochs:
        for metric, (estimate, target) in national.items():
            rows.append(("UK", metric, estimate, target, epoch))
        for i, (estimate, target) in enumerate(local):
            rows.append((f"area {i}", f"local/{i}", estimate, target, epoch))
    return pd.DataFrame(rows, columns=["name", "metric", "estimate", "target", "epoch"])


def test_final_fit_reads_the_last_epoch_and_drops_zero_targets():
    log = _log(
        {"a": (105, 100), "b": (150, 100), "zero": (5, 0)}, local=[(9, 10), (20, 10)]
    )
    log.loc[log.epoch == 0, "estimate"] = 1  # earlier epochs are ignored
    fit = final_fit(log)
    assert fit.epoch == 30
    assert fit.national.to_dict() == pytest.approx({"a": 0.05, "b": 0.5})
    assert fit.national_within == 0.5
    assert fit.local_within == 0.5


def test_a_malformed_log_is_refused():
    with pytest.raises(ValueError, match="no rows"):
        final_fit(_log({}, epochs=()))
    log = _log({"a": (np.inf, 100)})
    with pytest.raises(ValueError, match="non-finite"):
        final_fit(log)
    log = pd.concat([_log({"a": (1, 1)}), _log({"a": (2, 1)})])
    with pytest.raises(ValueError, match="repeats"):
        final_fit(log)


def test_a_drop_beyond_the_allowance_fails_and_within_it_passes():
    targets = {f"t{i}": (100, 100) for i in range(100)}
    base = final_fit(_log(targets, local=[(100, 100)] * 10))
    worse = dict(targets, **{f"t{i}": (150, 100) for i in range(3)})
    head = final_fit(_log(worse, local=[(100, 100)] * 10))
    result = compare_fits(base, head, POLICY)
    assert result["status"] == "fail"
    assert (
        "national targets within 10% falls from 100.0% to 97.0%"
        in result["failures"][0]
    )
    assert [f["target"] for f in result["flags"]] == ["t0", "t1", "t2"]
    slightly = dict(targets, t0=(150, 100))
    assert (
        compare_fits(base, final_fit(_log(slightly, local=[(100, 100)] * 10)), POLICY)[
            "status"
        ]
        == "pass"
    )


def test_builds_of_different_epochs_cannot_be_compared():
    base = final_fit(_log({"a": (1, 1)}, epochs=(0, 30)))
    head = final_fit(_log({"a": (1, 1)}, epochs=(0, 510, 511)))
    assert compare_fits(base, head, POLICY)["status"] == "fail"
    # 32 epochs log last at 30, or at 31 once the final epoch is saved (#594).
    final = final_fit(_log({"a": (1, 1)}, epochs=(0, 30, 31)))
    result = compare_fits(base, final, POLICY)
    assert result["status"] == "pass"
    assert "saves the final epoch" in result["notes"][0]


def test_the_pull_request_workflow_uploads_only_aggregate_reports():
    """The datasets are licensed microdata: no job may upload or cache one."""
    from pathlib import Path

    import yaml

    workflow = Path(__file__).parents[2] / ".github" / "workflows" / "pull_request.yaml"
    jobs = yaml.safe_load(workflow.read_text())["jobs"]
    uploads = []
    for job in jobs.values():
        for step in job.get("steps", []):
            uses = step.get("uses", "")
            assert not uses.startswith("actions/cache"), step
            if uses.startswith("actions/upload-artifact"):
                uploads.append(step["with"]["path"])
    assert uploads and all(
        p.endswith(("calibration_log.csv", "distribution_report.json")) for p in uploads
    ), uploads
    assert jobs["fit-comparison"]["needs"] == ["test", "base-build"]


def test_new_and_removed_targets_are_noted_not_failed():
    base = final_fit(_log({"a": (1, 1), "old": (1, 1)}))
    head = final_fit(_log({"a": (1, 1), "new": (1, 1)}))
    result = compare_fits(base, head, POLICY)
    assert result["status"] == "pass"
    assert result["notes"] == [
        "1 national targets are new: new",
        "1 national targets are gone: old",
    ]


errors = st.lists(st.floats(0, 3), min_size=1, max_size=40)


@settings(max_examples=200, deadline=None)
@given(national=errors, local=errors)
def test_a_build_compared_with_itself_passes_with_nothing_flagged(national, local):
    fit = final_fit(
        _log(
            {f"t{i}": (100 * (1 + e), 100) for i, e in enumerate(national)},
            local=[(100 * (1 + e), 100) for e in local],
        )
    )
    result = compare_fits(fit, fit, POLICY)
    assert result["status"] == "pass" and result["flags"] == []
    assert all(row["change"] == 0 for row in result["summary"].values())


@settings(max_examples=200, deadline=None)
@given(base=errors, data=st.data())
def test_improving_every_target_never_fails_and_changes_are_antisymmetric(base, data):
    better = [data.draw(st.floats(0, e)) for e in base]
    fit_base = final_fit(
        _log({f"t{i}": (100 * (1 + e), 100) for i, e in enumerate(base)})
    )
    fit_head = final_fit(
        _log({f"t{i}": (100 * (1 + e), 100) for i, e in enumerate(better)})
    )
    forward = compare_fits(fit_base, fit_head, POLICY)
    assert forward["status"] == "pass" and forward["flags"] == []
    backward = compare_fits(fit_head, fit_base, POLICY)
    for name in ("national", "local"):
        a, b = forward["summary"][name]["change"], backward["summary"][name]["change"]
        assert a == pytest.approx(-b) or (np.isnan(a) and np.isnan(b))


def test_the_shipped_policy_loads():
    policy = load_policy()
    assert policy.share_drop_allowance == 0.02
    assert policy.target_flag_threshold == 0.2


def test_the_command_exits_one_on_a_failure(tmp_path):
    targets = {f"t{i}": (100, 100) for i in range(50)}
    _log(targets).to_csv(tmp_path / "base.csv", index=False)
    _log(dict(targets, **{f"t{i}": (200, 100) for i in range(5)})).to_csv(
        tmp_path / "head.csv", index=False
    )
    out = tmp_path / "fit.md"
    assert (
        main(
            [
                "compare",
                str(tmp_path / "base.csv"),
                str(tmp_path / "head.csv"),
                "--markdown",
                str(out),
            ]
        )
        == 1
    )
    assert "## Calibration fit against the base build: fail" in out.read_text()
    assert (
        main(["compare", str(tmp_path / "base.csv"), str(tmp_path / "base.csv")]) == 0
    )
