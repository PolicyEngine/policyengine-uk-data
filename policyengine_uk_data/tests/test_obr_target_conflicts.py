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
from itertools import combinations
from unittest.mock import patch

import pytest
import requests
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

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
        # create_target_matrix zeroes a column outside the target's countries
        # (np.isin, so their order does not matter), so targets restricted to
        # different countries get different columns.
        countries = target.countries
        return key + (frozenset(countries) if countries is not None else None,)

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
            if any(
                not math.isclose(a, b, rel_tol=1e-9)
                for a, b in combinations(values.values(), 2)
            ):
                names = frozenset(values)
                if names not in INTENDED_SHARED_COLUMNS:
                    found.add((key, year, names))
    return found


@pytest.fixture(scope="module")
def obr_targets():
    def get(*args, **kwargs):
        raise requests.ConnectionError("offline")

    # The manifest loader reads verified committed bytes without downloading.
    with patch.object(requests, "get", side_effect=get):
        targets = obr.get_targets()
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
    targets = blm.calibration_targets()
    # Guard against passing on a registry whose sources failed silently.
    assert {"obr/ni_employee", "obr/ni_employer", "obr/vat"} <= {
        t.name for t in targets
    }
    assert conflicts(targets, column_of) == set()


def _brute_force(targets, column) -> set[tuple]:
    """Reference: compare every pair of targets directly, in every year listed
    by any target on their column (a carried-forward pair can disagree in a
    year only a third target lists)."""
    years_by_column: dict[tuple, set[int]] = {}
    for t in targets:
        if (key := column(t)) is not None:
            years_by_column.setdefault(key, set()).update(t.values)
    found = set()
    for i, a in enumerate(targets):
        for b in targets[i + 1 :]:
            if column(a) is None or column(a) != column(b):
                continue
            for year in years_by_column[column(a)]:
                va, vb = blm._resolve_value(a, year), blm._resolve_value(b, year)
                if None not in (va, vb) and not math.isclose(va, vb, rel_tol=1e-9):
                    found.add((column(a), year))
    return found


@st.composite
def synthetic_targets(draw) -> list[Target]:
    targets = []
    for i in range(draw(st.integers(0, 8))):
        is_count = draw(st.booleans())
        years = draw(
            st.lists(st.integers(2020, 2026), min_size=1, max_size=3, unique=True)
        )
        targets.append(
            Target(
                name=f"synthetic/{i}",
                variable=draw(st.sampled_from(["a", "b", "c"])),
                source="synthetic",
                unit=Unit.COUNT if is_count else Unit.GBP,
                is_count=is_count,
                countries=draw(
                    st.sampled_from(
                        [
                            None,
                            ("ENGLAND",),
                            ("WALES",),
                            ("ENGLAND", "WALES"),
                            ("WALES", "ENGLAND"),
                        ]
                    )
                ),
                carry_forward=draw(st.booleans()),
                values={y: draw(st.sampled_from([1.0, 2.0])) for y in years},
                custom_compute=(lambda *a: None)
                if draw(st.integers(0, 4)) == 0
                else None,
            )
        )
    return targets


@settings(
    max_examples=300,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(targets=synthetic_targets())
def test_conflict_check_matches_pairwise_definition(column_of, targets):
    """The grouped check flags exactly the (column, year) pairs a pairwise
    comparison does. Covers custom columns, counts against sums, country
    restrictions in either order, ``carry_forward=False`` and the
    nearest-earlier-year fallback. The VOA population uprating in
    ``_resolve_value`` is not drawn: it looks population totals up."""
    grouped = {(key, year) for key, year, _ in conflicts(targets, column_of)}
    assert grouped == _brute_force(targets, column_of)


def test_conflict_check_compares_every_pair(column_of):
    """isclose is not transitive: with the middle value first, comparing each
    value to the first passes, but the outer pair is more than 1e-9 apart."""
    targets = [
        Target(
            name=f"synthetic/{i}",
            variable="a",
            source="synthetic",
            unit=Unit.GBP,
            values={2025: value},
        )
        for i, value in enumerate([1 + 0.9e-9, 1.0, 1 + 1.8e-9])
    ]
    assert {year for _, year, _ in conflicts(targets, column_of)} == {2025}


def test_carried_values_are_compared_in_every_listed_year(column_of):
    """a and b disagree in 2024 and, carried forward, in 2025, a year only c
    lists; both checks must report both years (Sol review r2)."""
    targets = [
        Target(
            name=name,
            variable="a",
            source="synthetic",
            unit=Unit.GBP,
            values=values,
        )
        for name, values in [
            ("synthetic/a", {2024: 1.0}),
            ("synthetic/b", {2024: 1 + 1.8e-9}),
            ("synthetic/c", {2025: 1 + 0.9e-9}),
        ]
    ]
    expected = {(("sum", "a", None), 2024), (("sum", "a", None), 2025)}
    assert {(k, y) for k, y, _ in conflicts(targets, column_of)} == expected
    assert _brute_force(targets, column_of) == expected
