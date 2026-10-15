"""Final calibration fit from a build's log, and the comparison of two builds.

A build writes ``constituency_calibration_log.csv``: each target's estimate
and value every tenth epoch, and at the last one. ``final_fit`` reads the
last epoch. A target's error is |estimate / target - 1|, and targets of zero
are left out. The national rows are those named ``UK``.

``compare_fits`` checks that a pull request does not worsen the fit of the
build it would replace, both built the same way: same environment, epochs
and clone count. Two rules come from ``targets/fit_policy.yaml``:

- **Fail** when the share of national (or local) targets within 10% falls
  by more than the allowance.
- **Flag** each national target whose error rises by more than the
  same-code spread, for review.

Per-target moves are not yet gates: builds of the same code currently
differ by up to 0.62 on one target (#598).

    python -m policyengine_uk_data.utils.fit_report compare BASE.csv HEAD.csv \\
        [--markdown out.md]
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml

POLICY_PATH = Path(__file__).parents[1] / "targets" / "fit_policy.yaml"
WITHIN = 0.10
_COLUMNS = ["name", "metric", "estimate", "target", "epoch"]


@dataclass(frozen=True)
class Fit:
    epoch: int
    national: pd.Series  # error by target name
    local_errors: np.ndarray

    @property
    def national_within(self) -> float:
        return (
            float(np.mean(self.national <= WITHIN))
            if len(self.national)
            else float("nan")
        )

    @property
    def local_within(self) -> float:
        return (
            float(np.mean(self.local_errors <= WITHIN))
            if len(self.local_errors)
            else float("nan")
        )


@dataclass(frozen=True)
class FitPolicy:
    share_drop_allowance: float
    target_flag_threshold: float


def load_policy(path=None) -> FitPolicy:
    with open(path or POLICY_PATH) as f:
        raw = yaml.safe_load(f)
    if raw.get("schema_version") != 1:
        raise ValueError("fit policy: schema_version must be 1")
    return FitPolicy(
        share_drop_allowance=float(raw["share_within_10pct"]["allowed_drop"]),
        target_flag_threshold=float(raw["national_target"]["flag_rise"]),
    )


def errors(frame: pd.DataFrame) -> pd.Series:
    """|estimate / target - 1| for rows with a non-zero target."""
    frame = frame[frame.target != 0]
    return (frame.estimate / frame.target - 1).abs()


def final_fit(log) -> Fit:
    """The fit at a calibration log's last epoch."""
    df = (
        pd.read_csv(log, usecols=_COLUMNS) if not isinstance(log, pd.DataFrame) else log
    )
    if df.empty:
        raise ValueError("the calibration log has no rows")
    last = int(df.epoch.max())
    rows = df[df.epoch == last]
    national = rows[rows.name == "UK"]
    if national.metric.duplicated().any():
        raise ValueError(
            "the calibration log repeats a national target at its last epoch"
        )
    national_errors = errors(national.set_index("metric"))
    local_errors = errors(rows[rows.name != "UK"]).to_numpy()
    if not np.isfinite(national_errors).all() or not np.isfinite(local_errors).all():
        raise ValueError("the calibration log has non-finite estimates")
    return Fit(last, national_errors, local_errors)


def compare_fits(base: Fit, head: Fit, policy: FitPolicy) -> dict:
    """Failures, flags and the summary for a head build against its base."""
    failures, notes = [], []
    # A log's last epoch is a multiple of ten or, since #594, the final one;
    # the same epoch count gives last epochs less than ten apart.
    if abs(base.epoch - head.epoch) >= 10:
        failures.append(
            f"The builds ran different epochs (logs end at {base.epoch} and "
            f"{head.epoch}), so their fits cannot be compared."
        )
    elif base.epoch != head.epoch:
        notes.append(
            f"The base log ends at epoch {base.epoch} and the head's at {head.epoch}: "
            "one of them saves the final epoch (#594)."
        )
    summary = {}
    for name, b, h in (
        ("national", base.national_within, head.national_within),
        ("local", base.local_within, head.local_within),
    ):
        summary[name] = {"base": b, "head": h, "change": h - b}
        if h < b - policy.share_drop_allowance:
            failures.append(
                f"The share of {name} targets within 10% falls from {b:.1%} to {h:.1%}, "
                f"by more than the {policy.share_drop_allowance:.1%} allowed."
            )
    common = base.national.index.intersection(head.national.index)
    added = sorted(set(head.national.index) - set(common))
    removed = sorted(set(base.national.index) - set(common))
    if added:
        notes.append(f"{len(added)} national targets are new: {', '.join(added[:10])}")
    if removed:
        notes.append(
            f"{len(removed)} national targets are gone: {', '.join(removed[:10])}"
        )
    # Largest rise first; ties keep the log's order, so a comparison is
    # reproducible.
    rise = (head.national[common] - base.national[common]).sort_values(
        ascending=False, kind="mergesort"
    )
    flags = [
        {"target": t, "base": float(base.national[t]), "head": float(head.national[t])}
        for t, r in rise.items()
        if r > policy.target_flag_threshold
    ]
    return {
        "status": "fail" if failures else "pass",
        "failures": failures,
        "notes": notes,
        "flags": flags,
        "summary": summary,
        "targets_compared": len(common),
    }


def render(result: dict, policy: FitPolicy) -> str:
    lines = [
        f"## Calibration fit against the base build: {result['status']}",
        "",
        "| Targets within 10% | Base | Head | Change |",
        "|---|---:|---:|---:|",
    ]
    for name, row in result["summary"].items():
        lines.append(
            f"| {name.capitalize()} | {row['base']:.1%} | {row['head']:.1%} | {100 * row['change']:+.1f}pp |"
        )
    lines.append("")
    lines.append(
        f"A drop of more than {policy.share_drop_allowance:.1%} fails. "
        f"{result['targets_compared']} national targets were compared."
    )
    if result["failures"]:
        lines += ["", "### Failures", ""] + [f"- {f}" for f in result["failures"]]
    if result["notes"]:
        lines += ["", "### Notes", ""] + [f"- {n}" for n in result["notes"]]
    flags = result["flags"]
    lines += [
        "",
        f"### National targets whose error rises by more than {policy.target_flag_threshold:.2f} ({len(flags)})",
        "",
    ]
    if flags:
        lines += ["| Target | Base error | Head error |", "|---|---:|---:|"]
        lines += [
            f"| `{f['target']}` | {f['base']:.3f} | {f['head']:.3f} |"
            for f in flags[:30]
        ]
        if len(flags) > 30:
            lines.append(f"\n{len(flags) - 30} more in the job log.")
    else:
        lines.append("None.")
    lines += [
        "",
        "Flags are for review; builds of the same code differ this much (#598).",
    ]
    return "\n".join(lines) + "\n"


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m policyengine_uk_data.utils.fit_report"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser(
        "compare", help="compare a head build's final fit with its base's"
    )
    c.add_argument("base")
    c.add_argument("head")
    c.add_argument("--markdown", help="write the comparison here as markdown")
    c.add_argument("--policy", help="fit policy YAML (default targets/fit_policy.yaml)")
    args = parser.parse_args(argv)
    policy = load_policy(args.policy)
    result = compare_fits(final_fit(args.base), final_fit(args.head), policy)
    text = render(result, policy)
    if args.markdown:
        Path(args.markdown).write_text(text)
    print(text)
    for flag in result["flags"]:
        print(f"flag: {flag['target']} {flag['base']:.4f} -> {flag['head']:.4f}")
    return 1 if result["status"] == "fail" else 0


if __name__ == "__main__":
    sys.exit(main())
