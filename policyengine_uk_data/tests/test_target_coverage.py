"""Required coverage of the national calibration targets (targets/coverage.py).

create_target_matrix used to skip a target with a warning, or with nothing at
all when its year did not resolve (#456) or its name was already taken, and to
accept a column no household contributes to (#593). These tests pin the gate
that replaces that: every offered target is in the matrix with signal, or
listed in known_gaps.yaml with the status it may have, in the builds it may
have it.
"""

from collections import Counter
from fnmatch import fnmatchcase
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from policyengine_uk_data.targets import build_loss_matrix, registry
from policyengine_uk_data.targets.coverage import (
    BUILDS,
    EXCUSABLE,
    NEVER_EXCUSED,
    STATUSES,
    KnownGap,
    TargetCoverageError,
    TargetStatus,
    check_coverage,
    column_status,
    coverage_problems,
    current_build,
    load_known_gaps,
    parse_known_gaps,
)
from policyengine_uk_data.targets.schema import GeographicLevel, Target, Unit

UC_BANDS = "dwp/uc_payment_dist/*"


def test_column_status():
    assert column_status(5.0, [0, 1, 0]) == "in_matrix"
    assert column_status(5.0, [0, 0, 0]) == "zero_signal"
    # A zero target met by a zero column needs no signal.
    assert column_status(0.0, [0, 0, 0]) == "in_matrix"
    assert column_status(float("nan"), [1, 1]) == "non_finite"
    assert column_status(5.0, [1, np.inf]) == "non_finite"


def test_the_shipped_known_gaps_load_and_cite_issues():
    gaps = load_known_gaps()
    for gap in gaps:
        assert gap.status in EXCUSABLE
        assert gap.issue.startswith("https://github.com/PolicyEngine/")
        assert gap.reason
    listed = [(n, gap.status) for gap in gaps for n in gap.targets]
    assert len(listed) == len(set(listed))

    def named(status):
        return sum(len(g.targets) for g in gaps if g.status == status)

    # The 1.58.1 inventory at 2025, in every build: 15 two-child-limit targets
    # with no 2025 value, 19 without signal, and 156 SPI and 3 ONS targets
    # dropped under a name already taken.
    assert [named(s) for s in ("no_value_for_year", "zero_signal", "duplicate")] == [
        15,
        19,
        159,
    ]
    assert all(g.builds == set(BUILDS) for g in gaps if g.targets)
    # Only in testing builds, any UC payment band may lack signal (#598).
    [uc] = [g for g in gaps if g.patterns]
    assert (uc.status, uc.patterns, uc.builds) == (
        "zero_signal",
        (UC_BANDS,),
        {"testing"},
    )


def test_the_shipped_gaps_excuse_any_uc_band_without_signal_in_testing_builds_only():
    gaps = load_known_gaps()
    flickers = TargetStatus(
        "dwp/uc_payment_dist/COUPLE_WITH_CHILDREN_annual_payment_27_600_to_28_800",
        "zero_signal",
    )
    listed = TargetStatus(
        "dwp/uc_payment_dist/COUPLE_NO_CHILDREN_annual_payment_30_000_to_inf",
        "zero_signal",
    )
    broken = TargetStatus(flickers.name, "error", "ValueError: x")
    assert coverage_problems([flickers, listed], gaps, "testing")[0] == []
    assert coverage_problems([flickers, listed], gaps, "release")[0] == [
        f"{flickers.name}: zero_signal"
    ]
    assert coverage_problems([broken], gaps, "testing")[0] == [
        f"{flickers.name}: error (ValueError: x)"
    ]


def _entry(**kwargs):
    return {
        "status": "zero_signal",
        "reason": "x",
        "issue": "https://github.com/a",
        "targets": ["t"],
        **kwargs,
    }


@pytest.mark.parametrize(
    "entry, match",
    [
        (_entry(status="error"), "status"),
        (_entry(status="non_finite"), "status"),
        (_entry(reason=""), "reason"),
        (_entry(issue="#456"), "issue"),
        (_entry(targets=[]), "no targets"),
        (_entry(targets=["t", "t"]), "twice"),
        (_entry(builds=["nightly"]), "builds"),
        (_entry(builds=[]), "builds"),
        (_entry(builds="testing"), "builds"),
        # A pattern excuses testing builds only, and must be a pattern.
        (_entry(targets=[], patterns=["a/*"]), "testing builds"),
        (_entry(targets=[], patterns=["a/*"], builds=["release"]), "testing builds"),
        (_entry(targets=[], patterns=["a/b"], builds=["testing"]), "wildcard"),
        (
            _entry(status="error", targets=[], patterns=["a/*"], builds=["testing"]),
            "status",
        ),
    ],
)
def test_malformed_known_gaps_are_refused(entry, match):
    with pytest.raises(ValueError, match=match):
        parse_known_gaps({"schema_version": 1, "gaps": [entry]})


def _gap(status, *names, builds=BUILDS, patterns=()):
    return KnownGap(
        status,
        frozenset(names),
        "reason",
        "https://github.com/x",
        frozenset(builds),
        tuple(patterns),
    )


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


def test_a_gap_for_one_kind_of_build_excuses_nothing_in_the_other(monkeypatch):
    rows = [TargetStatus("dwp/uc_payment_dist/A", "zero_signal")]
    for gap in (
        _gap("zero_signal", rows[0].name, builds=["testing"]),
        _gap("zero_signal", builds=["testing"], patterns=[UC_BANDS]),
    ):
        assert check_coverage(rows, [gap], build="testing") == []
        with pytest.raises(TargetCoverageError, match="A: zero_signal"):
            check_coverage(rows, [gap], build="release")
    release_only = _gap("no_column", "b", builds=["release"])
    rows = [TargetStatus("b", "no_column")]
    assert check_coverage(rows, [release_only], build="release") == []
    with pytest.raises(TargetCoverageError, match="b: no_column"):
        check_coverage(rows, [release_only], build="testing")
    # Nor is it stale there: it does not apply.
    assert coverage_problems([], [release_only], "testing") == ([], [])
    # The build kind is read from TESTING, as create_datasets reads it.
    monkeypatch.setenv("TESTING", "1")
    assert current_build() == "testing"
    monkeypatch.setenv("TESTING", "0")
    assert current_build() == "release"
    monkeypatch.delenv("TESTING")
    assert current_build() == "release"
    with pytest.raises(TargetCoverageError, match="b: no_column"):
        check_coverage(rows, [_gap("no_column", "b", builds=["testing"])])


def test_a_pattern_excuses_only_its_status_and_is_stale_only_if_it_matches_nothing():
    [gap] = parse_known_gaps(
        {
            "schema_version": 1,
            "gaps": [_entry(targets=[], patterns=[UC_BANDS], builds=["testing"])],
        }
    )
    rows = [
        TargetStatus("dwp/uc_payment_dist/A", "zero_signal"),
        TargetStatus("dwp/uc_payment_dist/B", "in_matrix"),
        TargetStatus("dwp/uc_payment_dist/C", "error", "KeyError: x"),
        TargetStatus("dwp/uc_payment_dist/D", "non_finite"),
        TargetStatus("dwp/uc_payment_dist/E", "no_column"),
        TargetStatus("dwp/uc/F", "zero_signal"),
    ]
    assert coverage_problems(rows, [gap], "testing") == (
        [
            "dwp/uc_payment_dist/C: error (KeyError: x)",
            "dwp/uc_payment_dist/D: non_finite",
            "dwp/uc_payment_dist/E: no_column",
            "dwp/uc/F: zero_signal",
        ],
        [],
    )
    # Excusing nothing in one build is not stale; matching no target is.
    assert coverage_problems(rows[1:2], [gap], "testing") == ([], [])
    assert coverage_problems(
        [TargetStatus("obr/x", "in_matrix")], [gap], "testing"
    ) == (
        [],
        [f"{UC_BANDS}: listed as zero_signal, matches no target"],
    )


NAMES = ["a/x", "a/y", "b/x", "c"]
PATTERNS = ["a/*", "*/x", "z*"]


@settings(max_examples=300, deadline=None)
@given(
    rows=st.lists(st.tuples(st.sampled_from(NAMES), st.sampled_from(STATUSES))),
    build=st.sampled_from(BUILDS),
    data=st.data(),
)
def test_problems_are_exactly_the_unexcused_targets_out_of_the_matrix(
    rows, build, data
):
    # Hand-built gaps, so even statuses the file refuses: none may excuse an
    # error or a non-finite value.
    status = st.sampled_from(EXCUSABLE + NEVER_EXCUSED)
    builds = st.sampled_from([BUILDS, ("testing",), ("release",)])
    # A name may be listed under several statuses, each once.
    listed = data.draw(
        st.dictionaries(st.tuples(st.sampled_from(NAMES + ["zz"]), status), builds)
    )
    globbed = data.draw(st.dictionaries(st.sampled_from(PATTERNS), status))
    gaps = [_gap(s, n, builds=b) for (n, s), b in listed.items()]
    gaps += [_gap(s, builds=["testing"], patterns=[p]) for p, s in globbed.items()]
    statuses = [TargetStatus(n, s) for n, s in rows]
    problems, stale = coverage_problems(statuses, gaps, build)

    # The model: only the gaps for this build apply.
    named = {pair for pair, b in listed.items() if build in b}
    patterns = globbed if build == "testing" else {}

    def excused(name, status):
        return status not in NEVER_EXCUSED and (
            (name, status) in named
            or any(s == status and fnmatchcase(name, p) for p, s in patterns.items())
        )

    assert problems == [
        f"{n}: {s}" for n, s in rows if s != "in_matrix" and not excused(n, s)
    ]
    assert all(f"{n}: {s}" in problems for n, s in rows if s in NEVER_EXCUSED)
    stale_pairs = {
        (line.split(": listed as ")[0], line.split(": listed as ")[1].split(",")[0])
        for line in stale
    }
    assert stale_pairs == {
        (n, s) for n, s in named if s in NEVER_EXCUSED or (n, s) not in rows
    } | {
        (p, s)
        for p, s in patterns.items()
        if not any(fnmatchcase(n, p) for n, _ in rows)
    }
    # Order of the statuses never changes the verdict.
    shuffled = data.draw(st.permutations(statuses))
    assert sorted(coverage_problems(shuffled, gaps, build)[0]) == sorted(problems)


def _target(name, values={2025: 10.0}, **kwargs):
    return Target(
        **{
            "name": name,
            "variable": "employment_income",
            "source": "test",
            "unit": Unit.GBP,
            "values": values,
            **kwargs,
        }
    )


class FakeMicrosimulation:
    """Two households, in England and Wales."""

    def __init__(self, dataset=None, reform=None):
        pass

    def calculate(self, variable, *args, **kwargs):
        assert variable == "country"
        return pd.Series(["ENGLAND", "WALES"])


def _sources(*sources):
    """Fake source modules, one per list of targets."""
    return lambda: [SimpleNamespace(get_targets=lambda s=s: list(s)) for s in sources]


@pytest.fixture
def fake_matrix(monkeypatch):
    """The target matrix from chosen sources, with columns chosen by name."""
    import policyengine_uk

    monkeypatch.setattr(policyengine_uk, "Microsimulation", FakeMicrosimulation)

    def configure(columns, *sources):
        monkeypatch.setattr(registry, "discover_source_modules", _sources(*sources))

        def compute(target, ctx, year):
            column = columns[target.name]
            if isinstance(column, Exception):
                raise column
            return None if column is None else np.asarray(column)

        monkeypatch.setattr(build_loss_matrix, "_compute_column", compute)
        return SimpleNamespace(time_period="2025")

    return configure


def _statuses_by_name(statuses):
    by_name = {}
    for row in statuses:
        by_name.setdefault(row.name, []).append(row.status)
    return by_name


def test_a_dropped_target_fails_the_matrix_unless_it_is_a_known_gap(
    fake_matrix, monkeypatch
):
    dataset = fake_matrix(
        {"ok": [1.0, 0.0], "later": [1.0, 1.0], "broken": KeyError("no such variable")},
        [_target("ok"), _target("later", values={2026: 1.0}), _target("broken")],
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
        {"ok": [1.0, 2.0], "flat": [0.0, 0.0]}, [_target("ok"), _target("flat")]
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
    dataset = fake_matrix({"nan": [np.nan, 1.0]}, [_target("nan")])
    monkeypatch.setattr(
        "policyengine_uk_data.targets.coverage.load_known_gaps",
        lambda: [_gap("zero_signal", "nan")],
    )
    with pytest.raises(TargetCoverageError, match="nan: non_finite"):
        build_loss_matrix.create_target_matrix(dataset, time_period="2025")


def test_a_name_two_sources_emit_fails_unless_it_is_a_known_gap(
    fake_matrix, monkeypatch
):
    """Scenario A: two source modules emit the same national name."""
    dataset = fake_matrix(
        {"obr/foo": [1.0, 2.0]},
        [_target("obr/foo")],
        [_target("obr/foo", values={2025: 20.0})],
    )
    _, values, statuses = build_loss_matrix.build_target_matrix(dataset, "2025")
    assert values.to_dict() == {"obr/foo": 10.0}
    assert statuses == [
        TargetStatus(
            "obr/foo",
            "duplicate",
            "test national target for 2025 dropped for the test national one for 2025",
        ),
        TargetStatus("obr/foo", "in_matrix"),
    ]
    with pytest.raises(TargetCoverageError, match="obr/foo: duplicate"):
        build_loss_matrix.create_target_matrix(dataset, time_period="2025")
    monkeypatch.setattr(
        "policyengine_uk_data.targets.coverage.load_known_gaps",
        lambda: [_gap("duplicate", "obr/foo")],
    )
    df, _ = build_loss_matrix.create_target_matrix(dataset, time_period="2025")
    assert list(df.columns) == ["obr/foo"]


def test_a_regional_target_named_like_a_national_one_fails(fake_matrix):
    """Scenario B: the national target wins even when emitted second."""
    dataset = fake_matrix(
        {"ons/foo": [1.0, 2.0]},
        [
            _target(
                "ons/foo",
                values={2024: 5.0},
                geographic_level=GeographicLevel.REGION,
                source="ons",
            ),
            _target("ons/foo"),
        ],
    )
    _, values, statuses = build_loss_matrix.build_target_matrix(dataset, "2025")
    assert values.to_dict() == {"ons/foo": 10.0}
    assert _statuses_by_name(statuses) == {"ons/foo": ["duplicate", "in_matrix"]}
    assert statuses[0].detail == (
        "ons region target for 2024 dropped for the test national one for 2025"
    )
    with pytest.raises(TargetCoverageError, match="ons/foo: duplicate"):
        build_loss_matrix.create_target_matrix(dataset, time_period="2025")


def test_a_column_that_cannot_enter_the_matrix_has_one_error_status(fake_matrix):
    """The status follows the column into the matrix, so a column that fails
    to go in, or whose status fails, leaves only an error and no column."""
    dataset = fake_matrix(
        {
            "ok": [1.0, 2.0],
            "long": [1.0, 2.0, 3.0],
            "text": np.array(["a", "b"], dtype=object),
        },
        [_target("ok"), _target("long"), _target("text")],
    )
    df, values, statuses = build_loss_matrix.build_target_matrix(dataset, "2025")
    assert list(df.columns) == ["ok"] and list(values.index) == ["ok"]
    assert _statuses_by_name(statuses) == {
        "ok": ["in_matrix"],
        "long": ["error"],
        "text": ["error"],
    }


def _main_create_target_matrix(dataset, time_period):
    """create_target_matrix and get_all_targets as on main before the gate
    (daad71cd), with module references made explicit: the differential
    test's reference."""
    from policyengine_uk import Microsimulation

    def get_all_targets(geographic_level):
        all_targets, seen_names = [], set()
        for mod in registry.discover_source_modules():
            for target in mod.get_targets():
                if target.name in seen_names:
                    continue
                if target.geographic_level != geographic_level:
                    continue
                seen_names.add(target.name)
                all_targets.append(target)
        return all_targets

    year = int(time_period)
    sim = Microsimulation(dataset=dataset)
    sim.default_calculation_period = time_period
    ctx = build_loss_matrix._SimContext(sim, time_period, dataset, None)
    all_targets = []
    seen = set()
    for level in (
        GeographicLevel.NATIONAL,
        GeographicLevel.REGION,
        GeographicLevel.COUNTRY,
    ):
        for t in get_all_targets(geographic_level=level):
            if t.name not in seen:
                seen.add(t.name)
                all_targets.append(t)
    df = pd.DataFrame()
    target_names = []
    target_values = []
    for target in all_targets:
        try:
            val = build_loss_matrix._resolve_value(target, year)
            if val is None:
                continue
            col = build_loss_matrix._compute_column(target, ctx, year)
            if col is None:
                continue
            df[target.name] = build_loss_matrix.restrict_to_countries(
                col, ctx.country, target.countries
            )
            target_names.append(target.name)
            target_values.append(val)
        except Exception:
            pass
    return df, pd.Series(target_values, index=target_names)


def _column(kind):
    """A custom_compute giving a fixed column, no column, or an error."""
    if kind == "none":
        return lambda ctx, target, year: None
    if kind == "error":

        def raises(ctx, target, year):
            raise KeyError("no such variable")

        return raises
    return lambda ctx, target, year: np.asarray(kind, dtype=float)


emitted_target = st.builds(
    lambda name, level, year, value, column, countries: _target(
        name,
        values={year: value},
        geographic_level=level,
        countries=countries,
        custom_compute=_column(column),
    ),
    name=st.sampled_from(["a", "b", "c"]),
    level=st.sampled_from(
        [
            GeographicLevel.NATIONAL,
            GeographicLevel.REGION,
            GeographicLevel.COUNTRY,
            GeographicLevel.CONSTITUENCY,
        ]
    ),
    # 2023 carries forward to 2025; 2026 does not carry back.
    year=st.sampled_from([2023, 2025, 2026]),
    value=st.sampled_from([0.0, 10.0, float("nan")]),
    column=st.one_of(
        st.sampled_from(["none", "error"]),
        st.lists(
            st.sampled_from([0.0, 1.0, -2.5, np.nan, np.inf]), min_size=2, max_size=2
        ),
    ),
    countries=st.sampled_from([None, ("ENGLAND",)]),
)


@settings(max_examples=200, deadline=None)
@given(sources=st.lists(st.lists(emitted_target, max_size=5), min_size=1, max_size=3))
def test_without_the_gate_the_matrix_is_mains(sources):
    """require_coverage=False builds exactly the matrix main built, from the
    same sources, duplicates included; and every target offered at a matrix
    level gets exactly one status.

    The one intended difference, not generated here: a column whose status
    cannot be computed (a non-numeric column) now leaves the matrix as an
    error, where main kept it.
    """
    import policyengine_uk

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(policyengine_uk, "Microsimulation", FakeMicrosimulation)
        patch.setattr(registry, "discover_source_modules", _sources(*sources))
        dataset = SimpleNamespace(time_period="2025")
        df, values = build_loss_matrix.create_target_matrix(
            dataset, "2025", require_coverage=False
        )
        main_df, main_values = _main_create_target_matrix(dataset, "2025")
        _, _, statuses = build_loss_matrix.build_target_matrix(dataset, "2025")
    pd.testing.assert_frame_equal(df, main_df)
    pd.testing.assert_series_equal(values, main_values)
    offered = Counter(
        t.name
        for source in sources
        for t in source
        if t.geographic_level in build_loss_matrix.MATRIX_LEVELS
    )
    assert Counter(row.name for row in statuses) == offered


def test_stale_gaps_warn_without_failing(caplog):
    caplog.set_level("WARNING", logger="policyengine_uk_data.targets.coverage")
    stale = check_coverage([TargetStatus("a", "in_matrix")], [_gap("zero_signal", "a")])
    assert stale == ["a: listed as zero_signal, now in_matrix"]
    assert "Stale known calibration-target gap" in caplog.text


def test_stale_gaps_and_problems_reach_the_job_summary_once(tmp_path, monkeypatch):
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    gaps = [_gap("zero_signal", "a")]
    # The constituency and local authority calibrations both check.
    for _ in range(2):
        check_coverage([TargetStatus("a", "in_matrix")], gaps, build="release")
    text = summary.read_text()
    assert text.count("`a: listed as zero_signal, now in_matrix`") == 1
    assert text.count("### Calibration target coverage (release build)") == 1
    with pytest.raises(TargetCoverageError):
        check_coverage([TargetStatus("b", "no_column")], gaps, build="testing")
    assert (
        "Not in the matrix and not a known gap: `b: no_column`" in summary.read_text()
    )
    monkeypatch.delenv("GITHUB_STEP_SUMMARY")
    check_coverage([TargetStatus("c", "in_matrix")], [_gap("zero_signal", "c")])
    assert "`c:" not in summary.read_text()


def test_built_data_has_no_target_outside_the_matrix_and_the_known_gaps(enhanced_frs):
    """On the built dataset the gate passes: the known gaps are complete."""
    from policyengine_uk_data.datasets.frs_release import CURRENT_FRS_RELEASE

    _, _, statuses = build_loss_matrix.build_target_matrix(
        enhanced_frs, str(CURRENT_FRS_RELEASE.calibration_year)
    )
    problems, _ = coverage_problems(statuses, load_known_gaps())
    assert problems == []
