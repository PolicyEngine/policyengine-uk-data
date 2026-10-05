"""Calibration targets that share a matrix column must agree.

A target whose column is a plain sum (or recipient count) of one variable
shares that column with every other such target on the same variable. Two of
them with different values for one year contradict each other: no weights
meet both, so the optimiser splits the difference and distorts everything
else. obr/ni (total NICs, £200bn in 2025-26) and obr/ni_employee (£50bn) did
exactly that on ni_employee.

The column a target gets is decided by ``build_loss_matrix._compute_column``,
so these tests ask that dispatcher rather than restating its rules: every
custom compute function is swapped for a sentinel, so only targets that reach
the plain-variable fallbacks get a column key.
"""

import math
import random
from unittest.mock import patch

import pytest
import requests

from policyengine_uk_data.targets import build_loss_matrix as blm
from policyengine_uk_data.targets.schema import Target, Unit
from policyengine_uk_data.targets.sources import obr

# Targets allowed to share a plain-variable column with different values: the
# frozenset of their names, mapped to the reason. None are intended today.
INTENDED_SHARED_COLUMNS: dict[frozenset[str], str] = {}


@pytest.fixture
def column_of(monkeypatch):
    """Map a target to ("sum" | "count", variable, countries), or None."""
    for name in dir(blm):
        if name.startswith("compute_"):
            monkeypatch.setattr(blm, name, lambda *args, **kwargs: None)
    monkeypatch.setattr(blm, "_compute_simple_gbp", lambda t, ctx: ("sum", t.variable))
    monkeypatch.setattr(
        blm, "_compute_simple_count", lambda t, ctx: ("count", t.variable)
    )

    def column(target):
        if target.custom_compute is not None:
            return None
        key = blm._compute_column(target, None, None)
        if key is None:
            return None
        # Targets restricted to different countries get different columns
        # (the ``countries`` field proposed in #490 and #530).
        countries = getattr(target, "countries", None)
        return key + (tuple(countries) if countries else None,)

    return column


def conflicts(targets, column) -> set[tuple]:
    """(column, year, names) for every shared column whose values disagree."""
    groups: dict[tuple, list[Target]] = {}
    for target in targets:
        if (key := column(target)) is not None:
            groups.setdefault(key, []).append(target)
    found = set()
    for key, group in groups.items():
        for year in {y for t in group for y in t.values}:
            values = {
                t.name: v
                for t in group
                if (v := blm._resolve_value(t, year)) is not None
            }
            first = next(iter(values.values()), None)
            if any(not math.isclose(v, first, rel_tol=1e-9) for v in values.values()):
                names = frozenset(values)
                if names not in INTENDED_SHARED_COLUMNS:
                    found.add((key, year, names))
    return found


@pytest.fixture(scope="module")
def obr_targets():
    obr._download_workbook.cache_clear()

    def get(*args, **kwargs):
        raise requests.ConnectionError("offline")

    with (
        patch.object(obr.requests, "get", side_effect=get),
        patch.object(obr.time, "sleep", lambda s: None),
    ):
        targets = obr.get_targets()
    obr._download_workbook.cache_clear()
    return targets


def test_obr_targets_agree_on_shared_columns(obr_targets, column_of):
    assert conflicts(obr_targets, column_of) == set()


def test_total_nics_on_ni_employee_is_flagged(obr_targets, column_of):
    """The check fails on the mapping this PR removed."""
    total_nics = Target(
        name="obr/ni",
        variable="ni_employee",
        source="obr",
        unit=Unit.GBP,
        values={2025: 200.08e9},
    )
    found = conflicts(obr_targets + [total_nics], column_of)
    assert {(key, names) for key, _, names in found} == {
        (("sum", "ni_employee", None), frozenset({"obr/ni", "obr/ni_employee"}))
    }
    assert 2025 in {year for _, year, _ in found}


def test_calibration_targets_agree_on_shared_columns(column_of):
    """Every national, regional and country target the national matrix uses."""
    assert conflicts(blm.calibration_targets(), column_of) == set()


def _brute_force(targets, column) -> set[tuple]:
    """Reference: compare every pair of targets directly."""
    found = set()
    for i, a in enumerate(targets):
        for b in targets[i + 1 :]:
            if column(a) is None or column(a) != column(b):
                continue
            for year in set(a.values) | set(b.values):
                va, vb = blm._resolve_value(a, year), blm._resolve_value(b, year)
                if None not in (va, vb) and not math.isclose(va, vb, rel_tol=1e-9):
                    found.add((column(a), year))
    return found


def test_conflict_check_matches_pairwise_definition(column_of):
    """Seeded random target sets: the grouped check flags exactly the
    (column, year) pairs a pairwise comparison does. Covers custom columns,
    counts against sums, and the nearest-earlier-year fallback."""
    rng = random.Random(0)
    for _ in range(300):
        targets = []
        for i in range(rng.randint(0, 8)):
            is_count = rng.random() < 0.3
            years = rng.sample(range(2020, 2027), rng.randint(1, 3))
            targets.append(
                Target(
                    name=f"synthetic/{i}",
                    variable=rng.choice(["a", "b", "c"]),
                    source="synthetic",
                    unit=Unit.COUNT if is_count else Unit.GBP,
                    is_count=is_count,
                    values={y: rng.choice([1.0, 2.0]) for y in years},
                    custom_compute=(lambda *a: None) if rng.random() < 0.2 else None,
                )
            )
        grouped = {(key, year) for key, year, _ in conflicts(targets, column_of)}
        assert grouped == _brute_force(targets, column_of)
