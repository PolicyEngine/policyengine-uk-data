"""Final calibration fit of a build, summarised safely, and the comparison of two.

A build writes ``constituency_calibration_log.csv``: each target's estimate
and value every tenth epoch and at the last one, nationally (rows named
``UK``) and for each constituency. The log stays on the runner that built
it. ``summarise`` reduces its last epoch to what the comparison needs, and
only the summary is uploaded: for every national and local target, whether
its estimate is within 10% of its value (|estimate / target - 1| <= 0.1),
with zero targets left out. No estimate or error leaves the runner. An
estimate built from a few households, or dominated by one, divided by a
published count, could give that household's value back, and the dataset
repeats households (output-area clones, SPI and CGT donors), so a count of
rows can't show how many respondents support a figure (Sol re-review of
#599). Whether an estimate is within 10% of a public target is a bound, not
a value.

``compare`` checks that a pull request does not worsen the fit of the build
it would replace, built the same way (same environment, epochs and clones),
using the policy in ``targets/fit_policy.yaml``:

- **Fail** when the share of national, or of local, targets within 10%
  falls by more than the allowance, over the targets both builds have.
  Added and removed targets are reported, not counted.
- **Flag**, for review, each national target that is within 10% on the
  base and not on the head. Per-target moves do not fail while builds of
  the same code move 1 to 10 national targets out of the band (#598).

    python -m policyengine_uk_data.utils.fit_report summarise LOG.csv --out summary.json
    python -m policyengine_uk_data.utils.fit_report compare BASE.json HEAD.json [--markdown out.md]
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml

POLICY_PATH = Path(__file__).parents[1] / "targets" / "fit_policy.yaml"
SCHEMA = "policyengine-uk-data-fit-summary/2"
WITHIN = 0.10
_COLUMNS = ["name", "index", "metric", "estimate", "target", "epoch"]


@dataclass(frozen=True)
class FitPolicy:
    share_drop_allowance: float


def load_policy(path=None) -> FitPolicy:
    with open(path or POLICY_PATH) as f:
        raw = yaml.safe_load(f)
    if raw.get("schema_version") != 1:
        raise ValueError("fit policy: schema_version must be 1")
    if raw.get("national_target", {}).get("flag") != "left_band":
        raise ValueError("fit policy: national_target.flag must be left_band")
    return FitPolicy(
        share_drop_allowance=float(raw["share_within_10pct"]["allowed_drop"]),
    )


def summarise(log) -> dict:
    """The last epoch of a calibration log, reduced to what may leave the runner."""
    if isinstance(log, pd.DataFrame):
        df = log
    else:
        df = pd.read_csv(log, usecols=_COLUMNS)
    if df.empty:
        raise ValueError("the calibration log has no rows")
    epochs = sorted(df.epoch.unique())
    last = int(epochs[-1])
    rows = df[df.epoch == last]
    previous = int((df.epoch == epochs[-2]).sum()) if len(epochs) > 1 else None
    national = rows[rows.name == "UK"]
    if national.metric.duplicated().any():
        raise ValueError(
            "the calibration log repeats a national target at its last epoch"
        )
    national = national[national.target != 0].set_index("metric")
    local = rows[(rows.name != "UK") & (rows.target != 0)]
    n_errors = (national.estimate / national.target - 1).abs()
    l_errors = (local.estimate / local.target - 1).abs()
    if not np.isfinite(n_errors).all() or not np.isfinite(l_errors).all():
        raise ValueError("the calibration log has non-finite estimates")
    return {
        "schema": SCHEMA,
        "epoch": last,
        "rows_last_epoch": int(len(rows)),
        "rows_previous_epoch": previous,
        "national": {
            metric: bool(error <= WITHIN) for metric, error in n_errors.items()
        },
        "local": {
            f"{i}|{m}": bool(e <= WITHIN)
            for i, m, e in zip(local["index"], local.metric, l_errors)
        },
    }


def _share(flags: list) -> float:
    return float(np.mean(flags)) if flags else float("nan")


def compare(base: dict, head: dict, policy: FitPolicy) -> dict:
    """Failures, flags and the summary for a head build against its base."""
    failures, notes = [], []
    for side, s in (("base", base), ("head", head)):
        if s.get("schema") != SCHEMA:
            failures.append(f"The {side} fit summary is missing or malformed.")
            continue
        if not s["national"] or not s["local"]:
            missing = "national" if not s["national"] else "local"
            failures.append(f"The {side} build's last epoch has no {missing} targets.")
        previous = s["rows_previous_epoch"]
        if previous and s["rows_last_epoch"] < previous:
            failures.append(
                f"The {side} build's last logged epoch ({s['epoch']}) has fewer rows "
                f"({s['rows_last_epoch']}) than the one before ({previous}): the log "
                "is incomplete."
            )
    if failures:
        return {
            "status": "fail",
            "failures": failures,
            "notes": [],
            "flags": [],
            "summary": {},
            "targets_compared": 0,
        }
    # A log's last epoch is a multiple of ten or, since #594, the final one;
    # the same epoch count gives last epochs less than ten apart.
    if abs(base["epoch"] - head["epoch"]) >= 10:
        failures.append(
            f"The builds' logs end at epochs {base['epoch']} and {head['epoch']}, so "
            "they ran different epoch counts and their fits cannot be compared."
        )
    elif base["epoch"] != head["epoch"]:
        notes.append(
            f"The base log ends at epoch {base['epoch']} and the head's at "
            f"{head['epoch']}: one of them saves the final epoch (#594)."
        )
    summary = {}
    for name in ("national", "local"):
        b, h = base[name], head[name]
        common = sorted(set(b) & set(h))
        added, removed = sorted(set(h) - set(b)), sorted(set(b) - set(h))
        before = _share([bool(b[k]) for k in common])
        after = _share([bool(h[k]) for k in common])
        summary[name] = {
            "base": before,
            "head": after,
            "change": after - before,
            "compared": len(common),
        }
        if added:
            notes.append(
                f"{len(added)} {name} targets are new: {', '.join(added[:10])}"
            )
        if removed:
            notes.append(
                f"{len(removed)} {name} targets are gone: {', '.join(removed[:10])}"
            )
        if not common:
            failures.append(f"The builds have no {name} targets in common.")
        elif after < before - policy.share_drop_allowance:
            failures.append(
                f"The share of {name} targets within 10% falls from {before:.1%} to "
                f"{after:.1%}, over the {len(common)} targets both builds have, by "
                f"more than the {policy.share_drop_allowance:.1%} allowed."
            )
    common = sorted(set(base["national"]) & set(head["national"]))
    flags = [
        {"target": t} for t in common if base["national"][t] and not head["national"][t]
    ]
    entered = sum(1 for t in common if head["national"][t] and not base["national"][t])
    if entered:
        notes.append(f"{entered} national targets come within 10% on the head.")
    return {
        "status": "fail" if failures else "pass",
        "failures": failures,
        "notes": notes,
        "flags": flags,
        "summary": summary,
        "targets_compared": summary["national"]["compared"],
    }


def render(result: dict, policy: FitPolicy) -> str:
    lines = [f"## Calibration fit against the base build: {result['status']}", ""]
    if result["summary"]:
        lines += [
            "| Targets within 10% (those both builds have) | Base | Head | Change |",
            "|---|---:|---:|---:|",
        ]
        for name, row in result["summary"].items():
            lines.append(
                f"| {name.capitalize()} ({row['compared']:,}) | {row['base']:.1%} "
                f"| {row['head']:.1%} | {100 * row['change']:+.1f}pp |"
            )
        lines += ["", f"A drop of more than {policy.share_drop_allowance:.1%} fails."]
    if result["failures"]:
        lines += ["", "### Failures", ""] + [f"- {f}" for f in result["failures"]]
    if result["notes"]:
        lines += ["", "### Notes", ""] + [f"- {n}" for n in result["notes"]]
    flags = result["flags"]
    lines += [
        "",
        f"### National targets within 10% on the base and not on the head ({len(flags)})",
        "",
    ]
    if flags:
        lines += [f"- `{f['target']}`" for f in flags[:30]]
        if len(flags) > 30:
            lines.append(f"\n{len(flags) - 30} more in the job log.")
    else:
        lines.append("None.")
    lines += [
        "",
        "Flags are for review: builds of the same code move 1 to 10 national "
        "targets out of the band (#598). No target's estimate or error is "
        "published.",
    ]
    return "\n".join(lines) + "\n"


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m policyengine_uk_data.utils.fit_report"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("summarise", help="reduce a calibration log to a safe summary")
    s.add_argument("log")
    s.add_argument("--out", required=True)
    c = sub.add_parser(
        "compare", help="compare a head build's fit summary with its base's"
    )
    c.add_argument("base")
    c.add_argument("head")
    c.add_argument("--markdown", help="write the comparison here as markdown")
    c.add_argument("--policy", help="fit policy YAML (default targets/fit_policy.yaml)")
    args = parser.parse_args(argv)
    if args.command == "summarise":
        summary = summarise(args.log)
        Path(args.out).write_text(json.dumps(summary))
        print(
            f"epoch {summary['epoch']}: {len(summary['national'])} national, "
            f"{len(summary['local'])} local targets"
        )
        return 0
    policy = load_policy(args.policy)
    base = json.loads(Path(args.base).read_text())
    head = json.loads(Path(args.head).read_text())
    result = compare(base, head, policy)
    text = render(result, policy)
    if args.markdown:
        Path(args.markdown).write_text(text)
    print(text)
    for flag in result["flags"]:
        print(f"flag: {flag['target']} left the 10% band")
    return 1 if result["status"] == "fail" else 0


if __name__ == "__main__":
    sys.exit(main())
