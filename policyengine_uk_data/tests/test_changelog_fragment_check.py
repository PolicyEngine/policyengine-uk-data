"""Tests for .github/check_changelog_fragment.py, the PR changelog check."""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

REPO = Path(__file__).resolve().parents[2]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / ".github" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


checker = _load("check_changelog_fragment")
bump_version = _load("bump_version")
TYPES = checker.fragment_types(REPO / "pyproject.toml")
PATTERN = checker.fragment_pattern(TYPES)
BUMP = {"breaking": "major", "added": "minor", "removed": "minor"}
relaxed = settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])

# The untyped fragments that sat on main until this check landed. Towncrier
# never compiled them, so several releases say "No significant changes."
HISTORICAL_UNTYPED = """218 281 316 317 341 368 378 402 409 414 428 429 431 433 436
462 64 68 73 bus-fare-distribution-anchoring cgt-band-donors cgt-donor-sparsity
frs-year-semantics fuel-litre-calibration fuel-uprating-household-weight
hmrc-cgt-targets longwise-local-geography national-age-bands obr-efo-fallback
scottish-ct-water-charges spi-2022-23 spi-prior-diagnostics""".split()

valid_names = st.from_regex(r"[A-Za-z0-9][A-Za-z0-9._-]{0,10}", fullmatch=True)
any_names = st.text(
    st.characters(exclude_categories=("Cs",), exclude_characters="/\0"),
    min_size=1,
    max_size=12,
)
types = st.sampled_from(TYPES)
stems = st.text(st.sampled_from("abcxyz_-019"), min_size=1, max_size=8)
needing_paths = st.one_of(
    st.sampled_from(["pyproject.toml", "uv.lock", "Makefile", ".gitattributes"]),
    stems.map(lambda s: f"policyengine_uk_data/{s}.py"),
    stems.map(lambda s: f"policyengine_uk_data/storage/{s}.md"),
)
exempt_paths = st.one_of(
    st.sampled_from(["README.md", "CLAUDE.md", ".gitignore", ".claude/settings.json"]),
    stems.map(lambda s: f"docs/{s}.md"),
    stems.map(lambda s: f".github/workflows/{s}.yaml"),
    stems.map(lambda s: f"tools/{s}/build.py"),
    any_names.map(lambda n: ("D", f"changelog.d/{n}")),
    st.sampled_from(["A", "M", "D"]).map(lambda s: (s, "changelog.d/.gitkeep")),
)


def _changes(paths):
    return [p if isinstance(p, tuple) else ("M", p) for p in paths]


def _check(changes, contents=None, at_head=None):
    contents = contents or {}
    if at_head is None:
        at_head = [p for s, p in changes if s != "D" and p.startswith("changelog.d/")]
    return checker.check(
        changes, TYPES, lambda path: contents.get(path, b"Entry."), at_head
    )


def _towncrier_parse():
    builder = pytest.importorskip("towncrier._builder")
    if not hasattr(builder, "parse_newfragment_basename"):
        pytest.skip("towncrier no longer exposes parse_newfragment_basename")
    return builder.parse_newfragment_basename


def test_types_come_from_towncrier_config():
    assert TYPES == ["breaking", "added", "changed", "fixed", "removed"]


@pytest.mark.parametrize("stem", HISTORICAL_UNTYPED)
def test_historical_untyped_fragments_are_rejected_like_towncrier_ignores_them(stem):
    path = f"changelog.d/{stem}.md"
    problems = _check([("A", path), ("A", "changelog.d/x.fixed.md")])
    assert len(problems) == 1 and problems[0].startswith(f"{path}: name it")
    assert _towncrier_parse()(f"{stem}.md", TYPES) == (None, None, None)


@pytest.mark.parametrize(
    "name",
    ["x.added.1.md", "x.added.md.bak", "x.added.mdx", "x.Added.md", "x.added"]
    + ["sub/x.added.md", "x.feature.md", "added.md", "².added.md", "x .added.md"]
    + [".x.added.md", "-x.added.md", "x\r.added.md", "é.added.md"],
)
def test_near_miss_names_are_rejected(name):
    path = f"changelog.d/{name}"
    problems = _check([("A", path), ("A", "changelog.d/y.fixed.md")])
    assert len(problems) == 1 and problems[0].startswith(f"{path}: name it")


@pytest.mark.parametrize("name", ["x.added.1.md", "x.added.md.bak"])
def test_rejected_suffixes_would_split_towncrier_from_bump_version(tmp_path, name):
    # Towncrier files these under Added, but bump_version.py sees no type and
    # bumps the patch number instead of the minor one.
    (tmp_path / name).write_text("Entry.")
    assert _towncrier_parse()(name, TYPES)[1] == "added"
    assert bump_version.infer_bump(tmp_path) == "patch"


@pytest.mark.parametrize(
    "path, needed",
    [
        ("policyengine_uk_data/datasets/frs.py", True),
        ("policyengine_uk_data/tests/test_x.py", True),
        ("policyengine_uk_data/storage/BRMA_DATA_SOURCES.md", True),
        ("pyproject.toml", True),
        ("uv.lock", True),
        ("Makefile", True),
        (".gitattributes", True),
        ("README.md", False),
        ("CHANGELOG.md", False),
        ("docs/index.md", False),
        (".github/workflows/push.yaml", False),
        (".github/bump_version.py", False),
        (".claude/settings.json", False),
        ("tools/brma_households/build.py", False),
        ("changelog.d/x.fixed.md", False),
        (".gitignore", False),
    ],
)
def test_which_paths_need_a_fragment(path, needed):
    assert checker.needs_fragment(path) is needed


@relaxed
@given(st.lists(needing_paths, min_size=1), st.lists(exempt_paths), valid_names, types)
def test_a_typed_fragment_satisfies_any_change(needing, exempt, name, t):
    fragment = ("A", f"changelog.d/{name}.{t}.md")
    assert _check(_changes(needing + exempt) + [fragment]) == []


@relaxed
@given(st.lists(needing_paths, min_size=1), st.lists(exempt_paths), valid_names, types)
def test_a_change_needing_a_fragment_fails_without_an_added_one(needing, exempt, n, t):
    # Editing a fragment already on main does not count as adding one.
    edited = ("M", f"changelog.d/{n}.{t}.md")
    problems = _check(_changes(needing + exempt) + [edited])
    assert len(problems) == 1 and "adds no changelog fragment" in problems[0]


@relaxed
@given(st.lists(exempt_paths))
def test_changes_only_to_exempt_paths_need_no_fragment(exempt):
    assert _check(_changes(exempt)) == []


@relaxed
@given(st.lists(st.one_of(needing_paths, exempt_paths)), any_names)
def test_a_badly_named_fragment_is_always_rejected(others, name):
    path = f"changelog.d/{name}"
    if PATTERN.fullmatch(path):
        return
    problems = _check(_changes(others) + [("A", path)])
    assert any(p.startswith(f"{path}: name it") for p in problems)


@relaxed
@given(st.lists(needing_paths, min_size=1), valid_names, types)
def test_an_empty_fragment_is_rejected_and_does_not_count(needing, name, t):
    path = f"changelog.d/{name}.{t}.md"
    problems = _check(_changes(needing) + [("A", path)], {path: b" \n"})
    assert problems[0] == f"{path} is empty. Describe the change in it."
    assert "adds no changelog fragment" in problems[1]


def test_a_fragment_that_is_not_utf8_is_rejected():
    path = "changelog.d/x.fixed.md"
    assert _check([("A", path)], {path: b"\xff"}) == [f"{path} is not UTF-8 text."]


@relaxed
@given(st.lists(st.one_of(needing_paths, exempt_paths)), valid_names, types, st.data())
def test_problems_do_not_depend_on_change_order(paths, name, t, data):
    changes = _changes(paths) + [("A", f"changelog.d/{name}.{t}.md")]
    shuffled = data.draw(st.permutations(changes))
    assert sorted(_check(changes)) == sorted(_check(shuffled))


@relaxed
@given(st.one_of(valid_names, any_names), types)
def test_towncrier_reads_every_accepted_name_as_the_same_entry(name, t):
    basename = f"{name}.{t}.md"
    if match := PATTERN.fullmatch(f"changelog.d/{basename}"):
        issue, category, counter = _towncrier_parse()(basename, TYPES)
        assert (issue, category, counter) == (*checker.towncrier_entry(match), 0)


@relaxed
@given(valid_names, valid_names, types)
def test_a_collision_is_reported_exactly_when_towncrier_would_collide(a, b, t):
    paths = [f"changelog.d/{a}.{t}.md", f"changelog.d/{b}.{t}.md"]
    if paths[0] == paths[1]:
        return
    parse = _towncrier_parse()
    collide = parse(Path(paths[0]).name, TYPES) == parse(Path(paths[1]).name, TYPES)
    problems = _check([("A", path) for path in paths])
    assert bool(problems) is collide
    assert all("stops versioning" in p for p in problems)


def test_a_collision_with_a_fragment_already_on_main_is_reported():
    problems = _check(
        [("A", "changelog.d/01.fixed.md")],
        at_head=["changelog.d/1.fixed.md", "changelog.d/01.fixed.md"],
    )
    assert problems == [
        "changelog.d/01.fixed.md and changelog.d/1.fixed.md: towncrier files "
        "each as 1.fixed and stops versioning with an error. Rename all but one."
    ]


@pytest.mark.parametrize(
    "names, collide",
    [
        (["1.fixed.md", "01.fixed.md"], True),
        (["0.added.md", "000.added.md"], True),
        (["1.fixed.md", "1.0.fixed.md"], False),
        (["foo.fixed.md", "foo.added.md"], False),
    ],
)
def test_towncrier_build_fails_exactly_when_the_check_reports_a_collision(
    tmp_path, names, collide
):
    pytest.importorskip("towncrier")
    (tmp_path / "pyproject.toml").write_text((REPO / "pyproject.toml").read_text())
    (tmp_path / "changelog.d").mkdir()
    for name in names:
        (tmp_path / "changelog.d" / name).write_text(f"Entry {name}.")
    build = subprocess.run(
        [sys.executable, "-m", "towncrier", "build", "--draft"]
        + ["--name", "x", "--version", "0.0.0", "--dir", str(tmp_path)],
        capture_output=True,
        text=True,
    )
    assert (build.returncode != 0) is collide
    assert bool(_check([("A", f"changelog.d/{name}") for name in names])) is collide


def test_changed_paths_keeps_raw_path_bytes(monkeypatch):
    output = b"M\0docs/a\rb.md\0A\0changelog.d/\xff.fixed.md\0D\0changelog.d/x.md\0"
    monkeypatch.setattr(
        checker.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args, 0, stdout=output),
    )
    assert checker.changed_paths("a", "b") == [
        ("M", "docs/a\rb.md"),
        ("A", os.fsdecode(b"changelog.d/\xff.fixed.md")),
        ("D", "changelog.d/x.md"),
    ]


def _git(repo, *args):
    identity = ["-c", "user.name=t", "-c", "user.email=t@t"]
    return subprocess.run(
        ["git", *identity, "-c", "commit.gpgsign=false", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


@pytest.mark.parametrize(
    "edits, expected",
    [
        ({"policyengine_uk_data/x.py": "y = 2", "changelog.d/b.fixed.md": "B."}, 0),
        ({"policyengine_uk_data/x.py": "y = 2"}, 1),
        ({"docs/a b.md": "Docs.", "docs/c\rd.md": "CR.", ".github/w.yaml": "x"}, 0),
        ({"policyengine_uk_data/x.py": "y = 2", "changelog.d/b.md": "B."}, 1),
        ({"changelog.d/1.fixed.md": "A.", "changelog.d/01.fixed.md": "B."}, 1),
        # Same content, so git would report a rename unless told not to.
        (
            {
                "changelog.d/old.md": None,
                "changelog.d/old.fixed.md": "An untyped fragment already on main.",
                "policyengine_uk_data/x.py": "y = 2",
            },
            0,
        ),
    ],
)
def test_main_reads_the_pull_requests_own_diff(tmp_path, monkeypatch, edits, expected):
    files = {
        "pyproject.toml": (REPO / "pyproject.toml").read_text(),
        "changelog.d/.gitkeep": "",
        "changelog.d/old.md": "An untyped fragment already on main.",
        "policyengine_uk_data/x.py": "y = 1",
    }
    for path, text in files.items():
        (tmp_path / path).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / path).write_text(text)
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-qm", "base")
    base = _git(tmp_path, "rev-parse", "HEAD")
    for path, text in edits.items():
        target = tmp_path / path
        if text is None:
            target.unlink()
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text)
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-qm", "change")
    monkeypatch.chdir(tmp_path)
    assert checker.main([base, "HEAD"]) == expected


@settings(relaxed, max_examples=50)
@given(valid_names, types)
def test_accepted_names_give_bump_version_the_same_type(tmp_path_factory, name, t):
    directory = tmp_path_factory.mktemp("changelog.d")
    (directory / f"{name}.{t}.md").write_text("Entry.")
    assert bump_version.infer_bump(directory) == BUMP.get(t, "patch")
