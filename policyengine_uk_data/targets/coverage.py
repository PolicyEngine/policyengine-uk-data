"""Required coverage of the national calibration targets.

Every target the registry offers must enter the national loss matrix with
a finite value and a column that some household contributes to. The
exceptions are the targets ``targets/known_gaps.yaml`` lists, each with the
one status it may have, a reason and an issue. ``create_target_matrix``
checks this, so a build that would calibrate without a target, or against
one the weights cannot move, fails instead of logging a warning.

    python -m policyengine_uk_data.targets.coverage <dataset.h5> <year> [out.json]

reports every target's status on a dataset: names, statuses and counts only,
never household values.
"""

from __future__ import annotations

import json
import logging
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import yaml

logger = logging.getLogger(__name__)

KNOWN_GAPS_PATH = Path(__file__).with_name("known_gaps.yaml")
IN_MATRIX = "in_matrix"
# Statuses a known gap may excuse. An error or a non-finite value is never
# a known gap: it is a broken target.
EXCUSABLE = ("no_value_for_year", "no_column", "zero_signal")
NEVER_EXCUSED = ("error", "non_finite")
STATUSES = (IN_MATRIX,) + EXCUSABLE + NEVER_EXCUSED


class TargetCoverageError(Exception):
    """The national target matrix is missing targets it must have."""


@dataclass(frozen=True)
class TargetStatus:
    name: str
    status: str
    detail: str = ""


@dataclass(frozen=True)
class KnownGap:
    status: str
    targets: frozenset
    reason: str
    issue: str


def parse_known_gaps(raw) -> list:
    if (raw or {}).get("schema_version") != 1:
        raise ValueError("known gaps: schema_version must be 1")
    gaps, seen = [], set()
    for index, entry in enumerate(raw.get("gaps") or []):
        where = f"known gaps[{index}]"
        if entry.get("status") not in EXCUSABLE:
            raise ValueError(f"{where}: status must be one of {', '.join(EXCUSABLE)}")
        if not entry.get("reason") or not str(entry.get("issue", "")).startswith(
            "https://github.com/"
        ):
            raise ValueError(f"{where}: needs a reason and an issue URL")
        targets = list(entry.get("targets") or [])
        if not targets:
            raise ValueError(f"{where}: lists no targets")
        repeated = seen.intersection(targets) | {
            t for t, n in Counter(targets).items() if n > 1
        }
        if repeated:
            raise ValueError(f"{where}: {sorted(repeated)} listed twice")
        seen.update(targets)
        gaps.append(
            KnownGap(
                entry["status"],
                frozenset(targets),
                str(entry["reason"]),
                str(entry["issue"]),
            )
        )
    return gaps


def load_known_gaps(path=None) -> list:
    with open(path or KNOWN_GAPS_PATH) as f:
        return parse_known_gaps(yaml.safe_load(f))


def column_status(value, column) -> str:
    """A matrix column's status: non-finite, without signal, or fine.

    A column with no non-zero entry has no signal unless its target is
    zero too: the weights cannot move it.
    """
    column = np.asarray(column, dtype=float)
    if not np.isfinite(value) or not np.isfinite(column).all():
        return "non_finite"
    if value != 0 and not np.any(column != 0):
        return "zero_signal"
    return IN_MATRIX


def coverage_problems(statuses: Sequence[TargetStatus], gaps: Iterable[KnownGap]):
    """(problems, stale): targets out of place, and known gaps that no longer apply."""
    gaps = list(gaps)
    excused = {name: gap.status for gap in gaps for name in gap.targets}
    problems, matched = [], set()
    for row in statuses:
        if row.status == IN_MATRIX:
            continue
        if excused.get(row.name) == row.status:
            matched.add(row.name)
            continue
        detail = f" ({row.detail})" if row.detail else ""
        problems.append(f"{row.name}: {row.status}{detail}")
    present = {row.name: row.status for row in statuses}
    stale = []
    for gap in gaps:
        for name in sorted(gap.targets - matched):
            now = present.get(name, "not offered by the registry")
            stale.append(f"{name}: listed as {gap.status}, now {now}")
    return problems, stale


def check_coverage(
    statuses: Sequence[TargetStatus], gaps: Optional[Iterable[KnownGap]] = None
) -> list:
    """Raise unless every target is in the matrix or a known gap; return stale gaps."""
    problems, stale = coverage_problems(
        statuses, load_known_gaps() if gaps is None else gaps
    )
    counts = Counter(row.status for row in statuses)
    logger.info(
        "Calibration targets: %s",
        ", ".join(f"{counts[s]} {s}" for s in STATUSES if counts[s]),
    )
    for line in stale:
        logger.warning("Stale known calibration-target gap, remove it: %s", line)
    if problems:
        raise TargetCoverageError(
            f"{len(problems)} calibration targets are missing or cannot be fitted, "
            "and targets/known_gaps.yaml does not list them: " + "; ".join(problems)
        )
    return stale


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) not in (2, 3):
        print(__doc__)
        return 2
    from policyengine_uk.data import UKSingleYearDataset

    from policyengine_uk_data.targets.build_loss_matrix import build_target_matrix

    _, _, statuses = build_target_matrix(
        UKSingleYearDataset(file_path=argv[0]), argv[1]
    )
    problems, stale = coverage_problems(statuses, load_known_gaps())
    report = {
        "dataset": Path(argv[0]).name,
        "year": int(argv[1]),
        "by_status": Counter(row.status for row in statuses),
        "problems": problems,
        "stale_known_gaps": stale,
        "targets": [row.__dict__ for row in statuses],
    }
    if len(argv) == 3:
        Path(argv[2]).write_text(json.dumps(report, indent=1))
    print(
        json.dumps(
            {k: report[k] for k in ("by_status", "problems", "stale_known_gaps")},
            indent=1,
        )
    )
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
