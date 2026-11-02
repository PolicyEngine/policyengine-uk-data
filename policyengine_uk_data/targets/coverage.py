"""Required coverage of the national calibration targets.

Every target the registry offers must enter the national loss matrix with
a finite value and a column that some household contributes to, under a
name no other target has. The exceptions are the targets
``targets/known_gaps.yaml`` lists, each with the one status it may have, a
reason and an issue, for every build or only for testing or release builds.
``create_target_matrix`` checks this, so a build that would calibrate
without a target, or against one the weights cannot move, fails instead of
logging a warning.

    python -m policyengine_uk_data.targets.coverage <dataset.h5> <year> [out.json]

reports every target's status on a dataset: names, statuses and counts only,
never household values. Set TESTING=1 to check it as a testing build.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import yaml

logger = logging.getLogger(__name__)

KNOWN_GAPS_PATH = Path(__file__).with_name("known_gaps.yaml")
IN_MATRIX = "in_matrix"
# Statuses a known gap may excuse. An error or a non-finite value is never
# a known gap: it is a broken target.
EXCUSABLE = ("duplicate", "no_value_for_year", "no_column", "zero_signal")
NEVER_EXCUSED = ("error", "non_finite")
STATUSES = (IN_MATRIX,) + EXCUSABLE + NEVER_EXCUSED
# Pull request builds set TESTING=1 (smaller, faster); release builds don't.
BUILDS = ("testing", "release")


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
    builds: frozenset = frozenset(BUILDS)
    # fnmatch-style globs over target names; testing builds only.
    patterns: tuple = ()


def current_build() -> str:
    """The kind of build running, read from TESTING as the build itself does."""
    return "testing" if os.environ.get("TESTING", "0") == "1" else "release"


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
        builds = entry.get("builds", list(BUILDS))
        if not isinstance(builds, list) or not builds or not set(builds) <= set(BUILDS):
            raise ValueError(f"{where}: builds must list testing and/or release")
        targets = list(entry.get("targets") or [])
        patterns = list(entry.get("patterns") or [])
        if not targets and not patterns:
            raise ValueError(f"{where}: lists no targets")
        if patterns and set(builds) != {"testing"}:
            # A release gap names each target, so stale entries show up.
            raise ValueError(f"{where}: patterns may only excuse testing builds")
        if any(not set("*?[") & set(pattern) for pattern in patterns):
            raise ValueError(f"{where}: a pattern needs a wildcard")
        # A name may be listed under two statuses: a target without signal
        # whose duplicate is dropped, say. Under one status, it is listed once.
        names = [(name, entry["status"]) for name in targets + patterns]
        repeated = seen.intersection(names) | {
            t for t, n in Counter(names).items() if n > 1
        }
        if repeated:
            raise ValueError(
                f"{where}: {sorted(name for name, _ in repeated)} listed twice"
            )
        seen.update(names)
        gaps.append(
            KnownGap(
                entry["status"],
                frozenset(targets),
                str(entry["reason"]),
                str(entry["issue"]),
                frozenset(builds),
                tuple(patterns),
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


def coverage_problems(
    statuses: Sequence[TargetStatus],
    gaps: Iterable[KnownGap],
    build: Optional[str] = None,
):
    """(problems, stale) in one kind of build, by default the one running.

    Problems are the targets out of the matrix that no gap for this build
    excuses under their own status. Stale gaps are named targets that no
    longer have that status, and patterns that match no offered target:
    what a pattern excuses varies from one testing build to the next, so
    matching nothing in one build is not stale.
    """
    build = build or current_build()
    gaps = [gap for gap in gaps if build in gap.builds]
    named = {(name, gap.status) for gap in gaps for name in gap.targets}
    patterns = [(pattern, gap.status) for gap in gaps for pattern in gap.patterns]
    problems, matched, present = [], set(), defaultdict(list)
    for row in statuses:
        present[row.name].append(row.status)
        if row.status == IN_MATRIX:
            continue
        if row.status not in NEVER_EXCUSED:
            if (row.name, row.status) in named:
                matched.add((row.name, row.status))
                continue
            if any(
                status == row.status and fnmatchcase(row.name, pattern)
                for pattern, status in patterns
            ):
                continue
        detail = f" ({row.detail})" if row.detail else ""
        problems.append(f"{row.name}: {row.status}{detail}")
    stale = []
    for gap in gaps:
        for name in sorted(gap.targets):
            if (name, gap.status) in matched:
                continue
            now = ", ".join(present.get(name, ["not offered by the registry"]))
            stale.append(f"{name}: listed as {gap.status}, now {now}")
        for pattern in gap.patterns:
            if not any(fnmatchcase(name, pattern) for name in present):
                stale.append(f"{pattern}: listed as {gap.status}, matches no target")
    return problems, stale


def check_coverage(
    statuses: Sequence[TargetStatus],
    gaps: Optional[Iterable[KnownGap]] = None,
    build: Optional[str] = None,
) -> list:
    """Raise unless every target is in the matrix or a known gap; return stale gaps."""
    build = build or current_build()
    problems, stale = coverage_problems(
        statuses, load_known_gaps() if gaps is None else gaps, build
    )
    counts = Counter(row.status for row in statuses)
    logger.info(
        "Calibration targets (%s build): %s",
        build,
        ", ".join(f"{counts[s]} {s}" for s in STATUSES if counts[s]),
    )
    for line in stale:
        logger.warning("Stale known calibration-target gap, remove it: %s", line)
    write_step_summary(build, stale, problems)
    if problems:
        raise TargetCoverageError(
            f"{len(problems)} calibration targets are missing or cannot be fitted, "
            "and targets/known_gaps.yaml does not list them: " + "; ".join(problems)
        )
    return stale


def write_step_summary(build: str, stale: Sequence[str], problems: Sequence[str]):
    """Add stale gaps and problems to the GitHub Actions job summary, once each.

    A long build log buries a warning; the job summary shows it on the run's
    page. Lines already in the summary (the local-authority calibration
    repeats the constituency check) are not written again.
    """
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    lines = [f"- Stale, remove from targets/known_gaps.yaml: `{s}`" for s in stale]
    lines += [f"- Not in the matrix and not a known gap: `{p}`" for p in problems]
    if not path or not lines:
        return
    summary = Path(path)
    written = summary.read_text() if summary.exists() else ""
    lines = [line for line in lines if line not in written]
    if lines:
        with summary.open("a") as f:
            f.write(f"\n### Calibration target coverage ({build} build)\n\n")
            f.write("\n".join(lines) + "\n")


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
        "build": current_build(),
        "by_status": Counter(row.status for row in statuses),
        "problems": problems,
        "stale_known_gaps": stale,
        "targets": [row.__dict__ for row in statuses],
    }
    if len(argv) == 3:
        Path(argv[2]).write_text(json.dumps(report, indent=1))
    print(
        json.dumps(
            {
                k: report[k]
                for k in ("build", "by_status", "problems", "stale_known_gaps")
            },
            indent=1,
        )
    )
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
