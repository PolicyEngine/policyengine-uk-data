"""Single-age population and HBAI income distribution of a built dataset.

Calibration fits grouped targets. These checks measure what the groups can
hide, on the built dataset:

- **HBAI.** Person-weighted medians, means and Gini coefficients of
  equivalised household income, before and after housing costs. Also the
  share of individuals below 60% of the median, for everyone, HBAI
  dependent children, working-age adults and those at State Pension age.
  Each is compared with DWP's HBAI FYE 2025 estimate and 95% confidence
  interval (``targets/reference_data/hbai_fye_2025.yaml``), at period 2024.
- **Single ages.** Weighted people by sex and single year of age, at the
  calibration year, against the ONS 2024-based projection the calibration
  targets come from. Ages from 80 up are one band, because the dataset's
  oldest records are in the 80-84 band (#593).

``compare`` sets a pull request's build against its base's. While the
same-code spread of these measures is unmeasured (#598), it reports moves
and flags a measure that leaves the HBAI confidence interval, but fails
only on a broken or mismatched report.

    python -m policyengine_uk_data.utils.distribution_validation measure DATASET.h5 --out report.json
    python -m policyengine_uk_data.utils.distribution_validation compare BASE.json HEAD.json [--markdown out.md]

Reports hold aggregates only: no household's values.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Mapping, Optional

import numpy as np
import yaml

SCHEMA = "policyengine-uk-data-distribution/1"
HBAI_REFERENCE = (
    Path(__file__).parents[1] / "targets" / "reference_data" / "hbai_fye_2025.yaml"
)
# The FRS loader annualises weekly amounts with 365.25 / 7 (datasets/frs.py).
WEEKS_PER_YEAR = 365.25 / 7
TOP_AGE_BAND = 80
SEXES = {"male": ("MALE", "Males"), "female": ("FEMALE", "Females")}


def weighted_median(values, weights) -> float:
    values, weights = np.asarray(values, float), np.asarray(weights, float)
    order = np.argsort(values, kind="mergesort")
    cumulative = np.cumsum(weights[order])
    return float(values[order][np.searchsorted(cumulative, cumulative[-1] / 2)])


def weighted_gini(values, weights) -> float:
    """Gini coefficient of a weighted distribution (0 to 1)."""
    values, weights = np.asarray(values, float), np.asarray(weights, float)
    order = np.argsort(values, kind="mergesort")
    x, w = values[order], weights[order]
    cw = np.cumsum(w)
    cxw = np.cumsum(x * w)
    total_w, total_xw = cw[-1], cxw[-1]
    # Area under the Lorenz curve by the trapezium rule over weighted points.
    lorenz = cxw / total_xw
    previous = np.concatenate([[0.0], lorenz[:-1]])
    area = np.sum(w / total_w * (lorenz + previous) / 2)
    return float(1 - 2 * area)


def hbai_incomes(bhc, ahc, equivalisation_bhc, equivalisation_ahc):
    """Equivalised weekly incomes as HBAI defines them.

    HBAI resets negative incomes before housing costs to zero, and derives
    the income after housing costs from the adjusted figure, so it can still
    be negative (DWP, HBAI background information and methodology, FYE 2025,
    "Negative incomes"). ``bhc`` and ``ahc`` are annual household incomes.
    """
    bhc, ahc = np.asarray(bhc, float), np.asarray(ahc, float)
    housing_costs = bhc - ahc
    adjusted = np.maximum(bhc, 0.0)
    return (
        adjusted / np.asarray(equivalisation_bhc, float) / WEEKS_PER_YEAR,
        (adjusted - housing_costs)
        / np.asarray(equivalisation_ahc, float)
        / WEEKS_PER_YEAR,
    )


def hbai_measures(income_bhc, income_ahc, weights, child, pensioner) -> dict:
    """The HBAI reference measures from person-level arrays.

    ``income_*`` are each person's household equivalised income in £ a week;
    ``weights`` are person weights; ``child`` and ``pensioner`` are the HBAI
    groups (working-age adults are everyone else).
    """
    child, pensioner = np.asarray(child, bool), np.asarray(pensioner, bool)
    groups = {
        "all": np.ones_like(child),
        "children": child,
        "working_age": ~child & ~pensioner,
        "pensioners": pensioner,
    }
    out = {}
    for suffix, income in (("bhc", income_bhc), ("ahc", income_ahc)):
        income = np.asarray(income, float)
        median = weighted_median(income, weights)
        out[f"median_{suffix}"] = median
        out[f"mean_{suffix}"] = float(np.average(income, weights=weights))
        out[f"gini_{suffix}"] = 100 * weighted_gini(income, weights)
        low = income < 0.6 * median
        for group, mask in groups.items():
            w = weights * mask
            out[f"relative_low_income_{suffix}_{group}"] = float(
                100 * np.sum(w * low) / np.sum(w)
            )
    return out


def load_hbai_reference(path=None) -> dict:
    with open(path or HBAI_REFERENCE) as f:
        raw = yaml.safe_load(f)
    if raw.get("schema_version") != 1:
        raise ValueError("HBAI reference: schema_version must be 1")
    for name, m in raw["measures"].items():
        if not m["low"] <= m["estimate"] <= m["high"]:
            raise ValueError(f"HBAI reference {name}: estimate outside its interval")
    return raw


def single_ages(age, is_male, weights, reference: Mapping) -> dict:
    """People by sex and single age, ages from ``TOP_AGE_BAND`` up as one band.

    ``reference`` maps (sex, age) to the ONS count, with ages from
    ``TOP_AGE_BAND`` up summed by the caller.
    """
    age = np.minimum(np.floor(np.asarray(age, float)), TOP_AGE_BAND).astype(int)
    out = {}
    for sex, mask in (("male", is_male), ("female", ~np.asarray(is_male, bool))):
        counts = np.bincount(
            age[mask], weights=np.asarray(weights)[mask], minlength=TOP_AGE_BAND + 1
        )
        out[sex] = {
            str(a): {"value": float(counts[a]), "reference": float(reference[(sex, a)])}
            for a in range(TOP_AGE_BAND + 1)
        }
    return out


def ons_single_ages(year: int) -> dict:
    """The ONS 2024-based projection by sex and age, with 80+ as one band."""
    import pandas as pd

    from policyengine_uk_data.targets.sources.ons_demographics import (
        _download_uk_projection,
        _year_column,
    )

    df = _download_uk_projection()
    column = _year_column(df, year)
    if column is None:
        raise ValueError(f"the ONS projection has no {year}")
    ages = pd.to_numeric(
        df["Age"].astype(str).str.extract(r"(\d+)")[0], errors="coerce"
    )
    out = {}
    for sex, (_, ons_sex) in SEXES.items():
        rows = df["Sex"] == ons_sex
        for a in range(TOP_AGE_BAND):
            out[(sex, a)] = float(df.loc[rows & (ages == a), column].sum())
        out[(sex, TOP_AGE_BAND)] = float(
            df.loc[rows & (ages >= TOP_AGE_BAND), column].sum()
        )
    return out


def summarise_single_ages(rows: Mapping) -> dict:
    errors = {
        (sex, int(a)): cell["value"] / cell["reference"] - 1
        for sex, by_age in rows.items()
        for a, cell in by_age.items()
        if cell["reference"] > 0
    }
    under = [abs(e) for (_, a), e in errors.items() if a < TOP_AGE_BAND]
    teens = [abs(e) for (_, a), e in errors.items() if 15 <= a <= 19]
    return {
        "mean_abs_error_0_79": float(np.mean(under)),
        "max_abs_error_0_79": float(np.max(under)),
        "max_abs_error_15_19": float(np.max(teens)),
        "error_80_plus": {
            sex: float(
                rows[sex][str(TOP_AGE_BAND)]["value"]
                / rows[sex][str(TOP_AGE_BAND)]["reference"]
                - 1
            )
            for sex in rows
        },
    }


def measure(
    dataset_path, hbai_period: Optional[int] = None, age_year: int = 2025
) -> dict:
    """The distribution report of a built dataset (aggregates only)."""
    from policyengine_uk import Microsimulation

    reference = load_hbai_reference()
    period = hbai_period or int(reference["period"])
    sim = Microsimulation(dataset=str(dataset_path))

    def person(variable, year):
        return np.asarray(sim.calculate(variable, year, map_to="person").values, float)

    weights = person("household_weight", period)
    income_bhc, income_ahc = hbai_incomes(
        person("hbai_household_net_income", period),
        person("hbai_household_net_income_ahc", period),
        person("household_equivalisation_bhc", period),
        person("household_equivalisation_ahc", period),
    )
    measures = hbai_measures(
        income_bhc,
        income_ahc,
        weights,
        person("is_hbai_dependent_child", period) > 0,
        person("is_SP_age", period) > 0,
    )
    hbai = {}
    for name, ref in reference["measures"].items():
        value = measures[name]
        hbai[name] = {
            "value": value,
            "reference": ref["estimate"],
            "low": ref["low"],
            "high": ref["high"],
            "inside": bool(ref["low"] <= value <= ref["high"]),
        }
    gender = np.asarray(sim.calculate("gender", age_year).values).astype(str)
    ages = single_ages(
        person("age", age_year),
        gender == "MALE",
        person("household_weight", age_year),
        ons_single_ages(age_year),
    )
    data = Path(dataset_path).read_bytes()
    return {
        "schema": SCHEMA,
        "dataset": Path(dataset_path).name,
        "sha256": hashlib.sha256(data).hexdigest(),
        "periods": {"hbai": period, "single_age": age_year},
        "hbai": hbai,
        "single_age": ages,
        "summary": {
            "hbai_outside_interval": sorted(
                n for n, m in hbai.items() if not m["inside"]
            ),
            "single_age": summarise_single_ages(ages),
        },
    }


def _finite(x) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x)


def compare(base: Mapping, head: Mapping) -> dict:
    """A head build's distribution against its base's.

    Fails only on a broken or mismatched report; flags a measure that
    leaves its HBAI interval, or moves away from its reference.
    """
    failures, flags = [], []
    for side, report in (("base", base), ("head", head)):
        if report.get("schema") != SCHEMA:
            failures.append(f"The {side} distribution report is missing or malformed.")
    if failures:
        return {
            "status": "fail",
            "failures": failures,
            "flags": [],
            "rows": [],
            "single_age": {},
        }
    required = set(load_hbai_reference()["measures"])
    for side, report in (("base", base), ("head", head)):
        missing = sorted(required - set(report.get("hbai") or {}))
        if missing:
            failures.append(f"The {side} report has no {', '.join(missing)}.")
        summary = (report.get("summary") or {}).get("single_age") or {}
        for key in ("mean_abs_error_0_79", "max_abs_error_0_79", "max_abs_error_15_19"):
            if not _finite(summary.get(key)):
                failures.append(
                    f"The {side} report's single-age {key} is missing or not finite."
                )
    if failures:
        return {
            "status": "fail",
            "failures": failures,
            "flags": [],
            "rows": [],
            "single_age": {},
        }
    if base["periods"] != head["periods"]:
        failures.append(
            f"The reports measured different periods ({base['periods']} and {head['periods']})."
        )
    rows = []
    for name, h in head["hbai"].items():
        b = base["hbai"].get(name)
        if b is None or not (_finite(h["value"]) and _finite(b["value"])):
            failures.append(f"HBAI {name} is missing or not finite on one side.")
            continue
        worse = abs(h["value"] - h["reference"]) - abs(b["value"] - b["reference"])
        row = {
            "measure": name,
            "base": b["value"],
            "head": h["value"],
            "reference": h["reference"],
            "low": h["low"],
            "high": h["high"],
            "base_inside": b["inside"],
            "head_inside": h["inside"],
            "further_from_reference": worse,
        }
        rows.append(row)
        if b["inside"] and not h["inside"]:
            flags.append(
                f"HBAI {name} leaves its 95% interval: {b['value']:.2f} -> {h['value']:.2f} "
                f"(HBAI {h['reference']:.2f}, {h['low']:.2f} to {h['high']:.2f})."
            )
    sb, sh = base["summary"]["single_age"], head["summary"]["single_age"]
    single = {
        k: {"base": sb[k], "head": sh[k]}
        for k in ("mean_abs_error_0_79", "max_abs_error_0_79", "max_abs_error_15_19")
    }
    return {
        "status": "fail" if failures else "pass",
        "failures": failures,
        "flags": flags,
        "rows": rows,
        "single_age": single,
    }


def render(result: Mapping) -> str:
    lines = [
        f"## Income distribution and single ages against the base build: {result['status']}",
        "",
    ]
    if result["failures"]:
        lines += ["### Failures", ""] + [f"- {f}" for f in result["failures"]] + [""]
    if result["rows"]:
        lines += [
            "HBAI FYE 2025 (DWP), period 2024: equivalised income in £ a week, rates and Gini in per cent. "
            "A value outside HBAI's 95% interval is marked (outside).",
            "",
            "| Measure | Base | Head | HBAI | 95% interval |",
            "|---|---:|---:|---:|---|",
        ]
        for r in result["rows"]:
            mark = lambda inside: "" if inside else " (outside)"  # noqa: E731
            lines.append(
                f"| {r['measure']} | {r['base']:.2f}{mark(r['base_inside'])} | {r['head']:.2f}{mark(r['head_inside'])} "
                f"| {r['reference']:.2f} | {r['low']:.2f} to {r['high']:.2f} |"
            )
        lines.append("")
    if result["single_age"]:
        lines += [
            "| Single ages against the ONS projection | Base | Head |",
            "|---|---:|---:|",
        ]
        labels = {
            "mean_abs_error_0_79": "Mean error, ages 0 to 79",
            "max_abs_error_0_79": "Largest error, ages 0 to 79",
            "max_abs_error_15_19": "Largest error, ages 15 to 19",
        }
        for key, label in labels.items():
            cell = result["single_age"][key]
            lines.append(
                f"| {label} | {100 * cell['base']:.1f}% | {100 * cell['head']:.1f}% |"
            )
        lines.append("")
    lines += ["### Flags", ""] + ([f"- {f}" for f in result["flags"]] or ["None."])
    lines += [
        "",
        "These measures report and flag but do not fail yet: their same-code spread is unmeasured (#598).",
    ]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m policyengine_uk_data.utils.distribution_validation"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    m = sub.add_parser("measure", help="measure a built dataset")
    m.add_argument("dataset")
    m.add_argument("--out", required=True)
    c = sub.add_parser("compare", help="compare a head build's report with its base's")
    c.add_argument("base")
    c.add_argument("head")
    c.add_argument("--markdown")
    args = parser.parse_args(argv)
    if args.command == "measure":
        report = measure(args.dataset)
        Path(args.out).write_text(json.dumps(report, indent=1))
        print(json.dumps(report["summary"], indent=1))
        return 0
    base = json.loads(Path(args.base).read_text())
    head = json.loads(Path(args.head).read_text())
    result = compare(base, head)
    text = render(result)
    if args.markdown:
        Path(args.markdown).write_text(text)
    print(text)
    return 1 if result["status"] == "fail" else 0


if __name__ == "__main__":
    sys.exit(main())
