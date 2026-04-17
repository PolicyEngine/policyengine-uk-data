"""
Panel ID contract for UK datasets.

When this repo is extended to produce per-year snapshots (see issue #345),
a given household, benefit unit and person must carry the same identifier
across every yearly snapshot so that downstream consumers can join rows
into panel trajectories.

The identifiers used are the same ones built by ``create_frs`` in
``policyengine_uk_data/datasets/frs.py``:

- ``household_id`` on the household table
- ``benunit_id`` on the benefit unit table
- ``person_id`` on the person table

They are deterministic functions of the raw FRS ``sernum``, so they are
stable by construction as long as each yearly snapshot is derived from
the same FRS base. This module gives us a way to enforce that contract
at save time and in tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


PANEL_ID_COLUMNS: dict[str, str] = {
    "household": "household_id",
    "benunit": "benunit_id",
    "person": "person_id",
}


@dataclass(frozen=True)
class PanelIDs:
    """Sorted, deduplicated ID arrays for the three entity tables."""

    household: np.ndarray
    benunit: np.ndarray
    person: np.ndarray

    def as_dict(self) -> dict[str, np.ndarray]:
        return {
            "household": self.household,
            "benunit": self.benunit,
            "person": self.person,
        }


def get_panel_ids(dataset) -> PanelIDs:
    """Extract sorted unique IDs for each entity from a ``UKSingleYearDataset``.

    Args:
        dataset: an object exposing ``person``, ``benunit`` and
            ``household`` DataFrames (e.g. ``UKSingleYearDataset``).

    Returns:
        ``PanelIDs`` with one sorted ``int64`` array per entity.

    Raises:
        KeyError: if any of the expected ID columns is missing.
    """
    tables = {
        "household": dataset.household,
        "benunit": dataset.benunit,
        "person": dataset.person,
    }
    ids: dict[str, np.ndarray] = {}
    for entity, column in PANEL_ID_COLUMNS.items():
        df = tables[entity]
        if column not in df.columns:
            raise KeyError(
                f"Expected ID column '{column}' on the {entity} table "
                f"but columns were {list(df.columns)}."
            )
        ids[entity] = np.sort(np.unique(df[column].to_numpy().astype("int64")))
    return PanelIDs(**ids)


def assert_panel_id_consistency(
    base,
    other,
    *,
    entities: Iterable[str] = ("household", "benunit", "person"),
    label_base: str = "base",
    label_other: str = "other",
) -> None:
    """Assert that two datasets carry identical panel IDs.

    This is the save-time check for the panel-data pipeline: every yearly
    snapshot must contain the same households, benefit units and persons
    as the base snapshot, so that IDs can be used as panel keys.

    Args:
        base: the reference dataset (typically the base-year snapshot).
        other: a dataset being checked against the base.
        entities: which entities to check. Defaults to all three.
        label_base: human-readable label for ``base`` in error messages.
        label_other: human-readable label for ``other`` in error messages.

    Raises:
        AssertionError: if any entity's IDs differ between the two
            datasets. The message names the first missing and first extra
            IDs it finds, to keep output bounded when sets diverge.
    """
    base_ids = get_panel_ids(base)
    other_ids = get_panel_ids(other)

    problems: list[str] = []
    for entity in entities:
        a = base_ids.as_dict()[entity]
        b = other_ids.as_dict()[entity]
        if a.shape == b.shape and np.array_equal(a, b):
            continue

        missing = np.setdiff1d(a, b, assume_unique=True)
        extra = np.setdiff1d(b, a, assume_unique=True)
        parts = [
            f"{entity} IDs differ between {label_base} and {label_other}: "
            f"{len(a)} vs {len(b)} unique values."
        ]
        if missing.size:
            parts.append(
                f"First missing in {label_other}: {missing[:5].tolist()}"
                f"{' ...' if missing.size > 5 else ''}"
            )
        if extra.size:
            parts.append(
                f"First extra in {label_other}: {extra[:5].tolist()}"
                f"{' ...' if extra.size > 5 else ''}"
            )
        problems.append(" ".join(parts))

    if problems:
        raise AssertionError("\n".join(problems))
