"""Required coverage of the national calibration targets (targets/coverage.py).

create_target_matrix used to skip a target with a warning, or with nothing at
all when its year did not resolve (#456), and to accept a column no household
contributes to (#593). These tests pin the gate that replaces that: every
offered target is in the matrix with signal, or listed in known_gaps.yaml with
the status it may have.
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from policyengine_uk_data.targets import build_loss_matrix
from policyengine_uk_data.targets.coverage import (
    EXCUSABLE,
    STATUSES,
    KnownGap,
    TargetCoverageError,
    TargetStatus,
    check_coverage,
    column_status,
    coverage_problems,
    load_known_gaps,
    parse_known_gaps,
)
from policyengine_uk_data.targets.schema import GeographicLevel, Target, Unit


def test_column_status():
    assert column_status(5.0, [0, 1, 0]) == "in_matrix"
    assert column_status(5.0, [0, 0, 0]) == "zero_signal"
    # A zero target met by a zero column needs no signal.
    assert column_status(0.0, [0, 0, 0]) == "in_matrix"
    assert column_status(float("nan"), [1, 1]) == "non_finite"
    assert column_status(5.0, [1, np.inf]) == "non_finite"


def test_the_shipped_known_gaps_load_and_cite_issues():
    gaps = load_known_gaps()
    assert gaps
    for gap in gaps:
        assert gap.status in EXCUSABLE
        assert gap.issue.startswith("https://github.com/PolicyEngine/")
        assert gap.reason
    names = [n for gap in gaps for n in gap.targets]
    assert len(names) == len(set(names))
    # The 1.58.1 inventory: 15 two-child-limit targets with no 2025 value, 19
    # targets without signal, and 3 more without signal in TESTING builds.
    assert sum(len(g.targets) for g in gaps if g.status == "no_value_for_year") == 15
    assert sum(len(g.targets) for g in gaps if g.status == "zero_signal") == 22


@pytest.mark.parametrize(
    "entry, match",
    [
        (
            {
                "status": "error",
                "reason": "x",
                "issue": "https://github.com/a",
                "targets": ["t"],
            },
            "status",
        ),
        (
            {
                "status": "non_finite",
                "reason": "x",
                "issue": "https://github.com/a",
                "targets": ["t"],
            },
            "status",
        ),
        (
            {
                "status": "zero_signal",
                "reason": "",
                "issue": "https://github.com/a",
                "targets": ["t"],
            },
            "reason",
        ),
        (
            {"status": "zero_signal", "reason": "x", "issue": "#456", "targets": ["t"]},
            "issue",
        ),
        (
            {
                "status": "zero_signal",
                "reason": "x",
                "issue": "https://github.com/a",
                "targets": [],
            },
            "no targets",
        ),
        (
            {
                "status": "zero_signal",
                "reason": "x",
                "issue": "https://github.com/a",
                "targets": ["t", "t"],
            },
            "twice",
        ),
    ],
)
def test_malformed_known_gaps_are_refused(entry, match):
    with pytest.raises(ValueError, match=match):
        parse_known_gaps({"schema_version": 1, "gaps": [entry]})


def _gap(status, *names):
    return KnownGap(status, frozenset(names), "reason", "https://github.com/x")


def test_a_gap_excuses_only_its_own_status_and_stale_gaps_are_reported():
    statuses = [
        TargetStatus("a", "in_matrix"),
        TargetStatus("b", "zero_signal"),
        TargetStatus("c", "no_value_for_year"),
        TargetStatus("d", "error", "KeyError: x"),
    ]
    problems, stale = coverage_problems(
        statuses,
        [_gap("zero_signal", "b", "c"), _gap("no_column", "gone")],
    )
    assert problems == ["c: no_value_for_year", "d: error (KeyError: x)"]
    assert stale == [
        "c: listed as zero_signal, now no_value_for_year",
        "gone: listed as no_column, now not offered by the registry",
    ]


statuses_strategy = st.dictionaries(
    st.text("abcdef", min_size=1, max_size=3), st.sampled_from(STATUSES), max_size=12
)


@settings(max_examples=300, deadline=None)
@given(statuses=statuses_strategy, data=st.data())
def test_problems_are_exactly_the_unexcused_targets_out_of_the_matrix(statuses, data):
    names = sorted(statuses)
    excused = data.draw(
        st.dictionaries(
            st.sampled_from(names + ["zz"]) if names else st.just("zz"),
            st.sampled_from(EXCUSABLE),
        )
    )
    gaps = [_gap(status, name) for name, status in excused.items()]
    rows = [TargetStatus(name, statuses[name]) for name in names]
    problems, stale = coverage_problems(rows, gaps)
    expected = [
        f"{n}: {statuses[n]}"
        for n in names
        if statuses[n] != "in_matrix" and excused.get(n) != statuses[n]
    ]
    assert problems == expected
    assert {s.split(":")[0] for s in stale} == {
        n for n, status in excused.items() if statuses.get(n) != status
    }
    # Order of the statuses never changes the verdict.
    shuffled = data.draw(st.permutations(rows))
    assert sorted(coverage_problems(shuffled, gaps)[0]) == sorted(problems)


def _target(name, values={2025: 10.0}, **kwargs):
    return Target(
        name=name,
        variable="employment_income",
        source="test",
        unit=Unit.GBP,
        values=values,
        **kwargs,
    )


@pytest.fixture
def fake_matrix(monkeypatch):
    """create_target_matrix over two households, with chosen targets and columns."""
    import policyengine_uk

    class FakeMicrosimulation:
        def __init__(self, dataset=None, reform=None):
            pass

        def calculate(self, variable, *args, **kwargs):
            assert variable == "country"
            return pd.Series(["ENGLAND", "WALES"])

    monkeypatch.setattr(policyengine_uk, "Microsimulation", FakeMicrosimulation)

    def configure(targets, columns):
        monkeypatch.setattr(
            build_loss_matrix,
            "get_all_targets",
            lambda geographic_level=None: (
                targets if geographic_level == GeographicLevel.NATIONAL else []
            ),
        )

        def compute(target, ctx, year):
            column = columns[target.name]
            if isinstance(column, Exception):
                raise column
            return None if column is None else np.asarray(column, dtype=float)

        monkeypatch.setattr(build_loss_matrix, "_compute_column", compute)
        return SimpleNamespace(time_period="2025")

    return configure


def test_a_dropped_target_fails_the_matrix_unless_it_is_a_known_gap(
    fake_matrix, monkeypatch
):
    dataset = fake_matrix(
        [_target("ok"), _target("later", values={2026: 1.0}), _target("broken")],
        {"ok": [1.0, 0.0], "later": [1.0, 1.0], "broken": KeyError("no such variable")},
    )
    with pytest.raises(TargetCoverageError) as raised:
        build_loss_matrix.create_target_matrix(dataset, time_period="2025")
    message = str(raised.value)
    assert "later: no_value_for_year (values for 2026)" in message
    assert "broken: error (KeyError: 'no such variable')" in message
    df, values = build_loss_matrix.create_target_matrix(
        dataset, time_period="2025", require_coverage=False
    )
    assert list(df.columns) == ["ok"] and list(values.index) == ["ok"]
    monkeypatch.setattr(
        "policyengine_uk_data.targets.coverage.load_known_gaps",
        lambda: [_gap("no_value_for_year", "later")],
    )
    with pytest.raises(TargetCoverageError, match="broken"):
        build_loss_matrix.create_target_matrix(dataset, time_period="2025")


def test_a_column_without_signal_fails_unless_it_is_a_known_gap(
    fake_matrix, monkeypatch
):
    dataset = fake_matrix(
        [_target("ok"), _target("flat")], {"ok": [1.0, 2.0], "flat": [0.0, 0.0]}
    )
    with pytest.raises(TargetCoverageError, match="flat: zero_signal"):
        build_loss_matrix.create_target_matrix(dataset, time_period="2025")
    monkeypatch.setattr(
        "policyengine_uk_data.targets.coverage.load_known_gaps",
        lambda: [_gap("zero_signal", "flat")],
    )
    df, values = build_loss_matrix.create_target_matrix(dataset, time_period="2025")
    # Known gaps still enter the matrix: the change is that they are declared.
    assert list(df.columns) == ["ok", "flat"]


def test_a_non_finite_column_is_never_excused(fake_matrix, monkeypatch):
    dataset = fake_matrix([_target("nan")], {"nan": [np.nan, 1.0]})
    monkeypatch.setattr(
        "policyengine_uk_data.targets.coverage.load_known_gaps",
        lambda: [_gap("zero_signal", "nan")],
    )
    with pytest.raises(TargetCoverageError, match="nan: non_finite"):
        build_loss_matrix.create_target_matrix(dataset, time_period="2025")


def test_stale_gaps_warn_without_failing(caplog):
    caplog.set_level("WARNING", logger="policyengine_uk_data.targets.coverage")
    stale = check_coverage([TargetStatus("a", "in_matrix")], [_gap("zero_signal", "a")])
    assert stale == ["a: listed as zero_signal, now in_matrix"]
    assert "Stale known calibration-target gap" in caplog.text


def test_built_data_has_no_target_outside_the_matrix_and_the_known_gaps(enhanced_frs):
    """On the built dataset the gate passes: the known gaps are complete."""
    from policyengine_uk_data.datasets.frs_release import CURRENT_FRS_RELEASE

    _, _, statuses = build_loss_matrix.build_target_matrix(
        enhanced_frs, str(CURRENT_FRS_RELEASE.calibration_year)
    )
    problems, _ = coverage_problems(statuses, load_known_gaps())
    assert problems == []
