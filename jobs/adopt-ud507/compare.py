"""Compare #507's four builds. Aggregates only: counts, totals and hashes, never records.

PASS needs all of:
- provenance: every build exited 0; branch1/branch2 share the branch commit and the seeded calibrate.py, main1/main2
  share the main commit and the unseeded one; clean trees; the same thread settings and cached models; every target
  download served from the cache (no MISS) with the same bodies in all four; no OBR receipts failure (notice 7);
- branch1 and branch2: byte-identical constituency and local authority weights, identical eFRS and FRS tables,
  identical long-geography file, and identical income tax and household net income totals;
- main1 and main2: different weights (main's dropout is unseeded).
It reports main's spread in income tax and household net income, and how far the weights moved.

Usage (with a build worktree's venv, which has h5py, pandas and tables):
  ~/PolicyEngine/_worktrees/ud507-branch/.venv/bin/python compare.py
Reads builds/<label>/ and impact/<label>.json for branch1, branch2, main1, main2. Writes RESULT.md and RESULT.json.
Exit 0 = PASS, 1 = FAIL.
"""

import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

JOB = Path(__file__).resolve().parent
PAIRS = {"branch": ("branch1", "branch2"), "main": ("main1", "main2")}
WEIGHT_FILES = ("parliamentary_constituency_weights.h5", "local_authority_weights.h5")
STORES = ("enhanced_frs_2024_25.h5", "frs_2024_25.h5")
METRICS = ("income_tax", "household_net_income")
YEARS = ("2025", "2026")
failures, notes, report = [], [], {}


def build(label):
    return JOB / "builds" / label


def text(label, name):
    path = build(label) / name
    if path.exists():
        return path.read_text()
    zipped = path.with_name(path.name + ".gz")
    if zipped.exists():
        with gzip.open(zipped, "rt") as source:
            return source.read()
    return ""


def check(ok, message):
    if not ok:
        failures.append(message)
    return ok


def downloads(label):
    lines = text(label, "http.log").splitlines()
    misses = [line for line in lines if line.startswith("MISS")]
    # "cache <status> <bytes>b <sha12> <url>": the body each URL was served.
    return misses, Counter(tuple(line.split()[1:]) for line in lines)


def weights_diff(a, b):
    out = {}
    for name in WEIGHT_FILES:
        with (
            h5py.File(build(a) / name, "r") as fa,
            h5py.File(build(b) / name, "r") as fb,
        ):
            check(set(fa) == set(fb), f"{name}: keys differ between {a} and {b}")
            check(
                set(fa) == {"2025"} and set(fb) == {"2025"},
                f"{name}: missing expected 2025 weight arrays",
            )
            for key in sorted(set(fa) & set(fb)):
                x, y = fa[key][:], fb[key][:]
                areas = 650 if name.startswith("parliamentary") else 360
                check(
                    x.ndim == 2 and x.shape[0] == areas and x.size > 0,
                    f"{a}/{name}/{key}: empty or invalid weight shape",
                )
                check(
                    y.ndim == 2 and y.shape[0] == areas and y.size > 0,
                    f"{b}/{name}/{key}: empty or invalid weight shape",
                )
                check(
                    np.isfinite(x).all() and np.isfinite(y).all(),
                    f"{name}/{key}: weights include non-finite values",
                )
                same_layout = x.dtype == y.dtype and x.shape == y.shape
                identical = same_layout and x.tobytes() == y.tobytes()
                entry = {
                    "shape": list(x.shape),
                    "dtype": str(x.dtype),
                    "byte_identical": identical,
                }
                if same_layout and not identical:
                    entry["cells_differing"] = int((x != y).sum())
                    entry["sum_abs_diff_over_total"] = float(
                        np.abs(x.astype(float) - y).sum() / x.astype(float).sum()
                    )
                out[f"{name}/{key}"] = entry
    return out


def tables_diff(a, b):
    out = {}
    for name in STORES:
        with (
            pd.HDFStore(build(a) / name, "r") as sa,
            pd.HDFStore(build(b) / name, "r") as sb,
        ):
            check(
                set(sa.keys()) == set(sb.keys()),
                f"{name}: tables differ between {a} and {b}",
            )
            expected_tables = {"/household", "/person", "/benunit", "/time_period"}
            check(
                set(sa.keys()) == expected_tables and set(sb.keys()) == expected_tables,
                f"{name}: incomplete expected dataset tables",
            )
            for key in sorted(set(sa.keys()) & set(sb.keys())):
                x, y = sa[key], sb[key]
                if isinstance(x, pd.Series):
                    x, y = x.to_frame(), y.to_frame()
                differing = sorted(
                    c
                    for c in set(x.columns) | set(y.columns)
                    if c not in x or c not in y or not x[c].equals(y[c])
                )
                out[f"{name}{key}"] = {
                    "rows": [len(x), len(y)],
                    "columns_differing": differing,
                    "layout_identical": x.index.equals(y.index)
                    and x.columns.equals(y.columns),
                }
    return out


def household_weight_distance(a, b):
    x, y = (pd.read_hdf(build(label) / STORES[0], "household") for label in (a, b))
    if (
        len(x) != len(y)
        or not (x.household_id.to_numpy() == y.household_id.to_numpy()).all()
    ):
        return {"aligned": False}
    w, v = x.household_weight.to_numpy(float), y.household_weight.to_numpy(float)
    return {
        "aligned": True,
        "rows": len(w),
        "rows_differing": int((w != v).sum()),
        "totals": [float(w.sum()), float(v.sum())],
        "sum_abs_diff_over_total": float(np.abs(w - v).sum() / w.sum()),
    }


def long_geography_digest(label):
    with gzip.open(build(label) / "local_geography_weights.csv.gz", "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def totals(label):
    path = JOB / "impact" / f"{label}.json"
    if not check(path.exists(), f"impact/{label}.json missing (run placebo/totals.py)"):
        return None
    data = json.loads(path.read_text())
    check(
        data["dataset_sha256"] == text(label, "dataset.sha256").strip(),
        f"{label}: totals came from a different dataset",
    )
    check(
        data["policyengine_uk_version"] == "2.122.2"
        and data["policyengine_core_version"] == "3.32.13",
        f"{label}: totals used a different engine environment",
    )
    return {y: {m: data[y][m]["total_bn"] for m in METRICS} for y in YEARS}


labels = [label for pair in PAIRS.values() for label in pair]
expected_refs = json.loads((JOB / "expected_refs.json").read_text())

# Provenance.
served = {}
for label in labels:
    check(
        (build(label) / "finished").exists()
        and text(label, "build.log").rstrip().endswith("exit 0"),
        f"{label}: build did not finish with exit 0",
    )
    check(
        text(label, "status").strip() == "",
        f"{label}: tracked files modified in the build worktree",
    )
    kind = "branch" if label.startswith("branch") else "main"
    check(
        text(label, "commit").strip() == expected_refs[kind],
        f"{label}: build is not of the exact requested {kind} commit",
    )
    check(
        text(label, "calibrate.sha256")
        == text(label, "calibrate_committed.sha256")
        != "",
        f"{label}: calibration source does not match its committed source",
    )
    check(
        "Failed to download/parse OBR receipts" not in text(label, "build.log"),
        f"{label}: OBR receipts fell back (notice 7)",
    )
    misses, served[label] = downloads(label)
    check(not misses, f"{label}: {len(misses)} downloads missed the frozen cache")
    check(
        text(label, "models_after.sha256") == text(label, "models_before.sha256"),
        f"{label}: an imputation model changed during the build",
    )
    check(
        text(label, "targets_before.sha256") == text(label, "targets_after.sha256"),
        f"{label}: the frozen target cache changed during the build",
    )
for kind, (a, b) in PAIRS.items():
    check(
        text(a, "commit") == text(b, "commit") != "",
        f"{kind} builds are not of one commit",
    )
    source = text(a, "calibrate_source")
    seeded = "dropout_generator = torch.Generator()" in source
    check(
        seeded == (kind == "branch") and source == text(b, "calibrate_source"),
        f"{kind} builds: wrong calibrate.py",
    )
check(
    text("branch1", "commit") != text("main1", "commit"),
    "branch and main builds share a commit",
)
check(
    len({text(label, "env") for label in labels}) == 1,
    "thread settings differ between builds",
)
check("torch_threads 1" in text("branch1", "env"), "builds did not use one CPU thread")
check(
    len({text(label, "lock.sha256") for label in labels}) == 1,
    "builds used different dependency locks",
)
check(
    len({text(label, "models_before.sha256") for label in labels}) == 1,
    "builds started from different cached models",
)
check(
    all(
        text(label, "models_before.sha256")
        == (JOB / "frozen/models.sha256").read_text()
        for label in labels
    ),
    "builds did not use the frozen model manifest",
)
check(
    all(
        text(label, "targets_before.sha256")
        == (JOB / "frozen/targets.sha256").read_text()
        for label in labels
    ),
    "builds did not use the frozen target manifest",
)
check(
    len({frozenset(served[label].items()) for label in labels}) == 1,
    "builds were served different target downloads",
)

# Outputs.
for kind, (a, b) in PAIRS.items():
    weights = weights_diff(a, b)
    tables = tables_diff(a, b)
    entry = {
        "weights": weights,
        "tables_differing": {
            k: v["columns_differing"]
            for k, v in tables.items()
            if v["columns_differing"]
        },
        "household_weight": household_weight_distance(a, b),
        "long_geography_identical": long_geography_digest(a)
        == long_geography_digest(b),
    }
    ta, tb = totals(a), totals(b)
    if ta and tb:
        entry["totals_bn"] = {
            y: {m: [ta[y][m], tb[y][m], tb[y][m] - ta[y][m]] for m in METRICS}
            for y in YEARS
        }
        entry["spread_bn"] = {
            y: {m: abs(tb[y][m] - ta[y][m]) for m in METRICS} for y in YEARS
        }
    report[kind] = entry
    all_identical = all(w["byte_identical"] for w in weights.values())
    check(
        len(weights) == len(WEIGHT_FILES), f"{kind}: incomplete weight-array comparison"
    )
    check(
        all(v["layout_identical"] for v in tables.values()),
        f"{kind}: dataset row or column layouts differ",
    )
    if kind == "branch":
        check(all_identical, "branch builds: weights differ")
        check(not entry["tables_differing"], "branch builds: eFRS or FRS tables differ")
        check(
            entry["long_geography_identical"],
            "branch builds: long-geography file differs",
        )
        if "totals_bn" in entry:
            check(
                all(entry["totals_bn"][y][m][2] == 0 for y in YEARS for m in METRICS),
                "branch builds: totals differ",
            )
    else:
        check(
            not all_identical,
            "main builds: weights identical, so main's dropout did not vary",
        )
        check(
            not any(key.startswith(STORES[1]) for key in entry["tables_differing"]),
            "main builds differ before calibration: an additional source of noise",
        )

# All four pre-calibration datasets must agree. Person and benefit-unit tables
# must also agree after calibration; only household weights and the monetary
# household proxies rescaled from those weights may change.
for label in labels[1:]:
    comparison = tables_diff("branch1", label)
    protected = {
        key: value
        for key, value in comparison.items()
        if key.startswith(STORES[1])
        or key in (f"{STORES[0]}/person", f"{STORES[0]}/benunit")
    }
    check(
        bool(protected)
        and all(
            not value["columns_differing"] and value["layout_identical"]
            for value in protected.values()
        ),
        f"branch1 vs {label}: inputs or non-household calibrated tables differ",
    )

report["verdict"] = "PASS" if not failures else "FAIL"
report["failures"], report["notes"] = failures, notes
(JOB / "RESULT.json").write_text(json.dumps(report, indent=1))

md = [f"# uk-data#507 build comparison: {report['verdict']}", ""]
md += [f"- FAIL: {f}" for f in failures] + [f"- Note: {n}" for n in notes]
md += [
    "",
    "| Pair | Weight arrays byte-identical | Household weights differing | Σ|Δw|/Σw | eFRS/FRS tables with differing columns |",
    "|---|---|---|---|---|",
]
for kind in PAIRS:
    e = report[kind]
    hw = e["household_weight"]
    md.append(
        f"| {kind} | {sum(w['byte_identical'] for w in e['weights'].values())} of {len(e['weights'])} "
        f"| {hw.get('rows_differing', 'n/a')} of {hw.get('rows', 'n/a')} | {hw.get('sum_abs_diff_over_total', float('nan')):.2e} "
        f"| {', '.join(e['tables_differing']) or 'none'} |"
    )
md += [
    "",
    "| Pair | Year | Measure | Build 1 (£bn) | Build 2 (£bn) | Difference (£bn) |",
    "|---|---|---|---|---|---|",
]
for kind in PAIRS:
    for y, measures in report[kind].get("totals_bn", {}).items():
        for m, (x, z, d) in measures.items():
            md.append(f"| {kind} | {y} | {m} | {x:.3f} | {z:.3f} | {d:+.3f} |")
(JOB / "RESULT.md").write_text("\n".join(md) + "\n")
print("\n".join(md))
raise SystemExit(0 if not failures else 1)
