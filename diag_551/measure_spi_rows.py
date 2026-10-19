"""DIAGNOSTIC ONLY (#551): aggregates of a built enhanced FRS, FRS rows vs SPI-synthetic rows.

Aggregates only. Every cell reports the number of distinct survey households behind
it, and every figure from fewer than 10 is suppressed (null). Quantiles are reported
only with at least 10 people on each side. Capital-gains clones and CGT band donors
repeat survey households, so they are left out of the household counts; weighted
figures use the build's calibrated household weights. SPI-row incomes are drawn from
SPI 2022-23 records and are not uprated to the 2024-25 FRS year.

    python diag_551/measure_spi_rows.py ENHANCED_FRS.h5 --out spi_rows_report.json
"""

import argparse
import json

import numpy as np
import pandas as pd

MIN_HOUSEHOLDS = 10
QUANTILES = (0.1, 0.25, 0.5, 0.75, 0.9)
STATUS_GROUPS = {
    "child": ["CHILD"],
    "employee": ["FT_EMPLOYED", "PT_EMPLOYED"],
    "self_employed": ["FT_SELF_EMPLOYED", "PT_SELF_EMPLOYED"],
    "out_of_work": [
        "UNEMPLOYED",
        "RETIRED",
        "STUDENT",
        "CARER",
        "LONG_TERM_DISABLED",
        "SHORT_TERM_DISABLED",
        "OTHER_INACTIVE",
    ],
}
INCOMES = [
    "employment_income",
    "self_employment_income",
    "private_pension_income",
    "savings_interest_income",
    "dividend_income",
    "property_income",
]


def households(d):
    return int(d.loc[d.survey, "person_household_id"].nunique())


def wquantiles(x, w):
    order = np.argsort(x, kind="stable")
    x, w = np.asarray(x)[order], np.asarray(w)[order]
    cum = np.cumsum(w) / w.sum()
    out = {}
    for q in QUANTILES:
        i = min(int(np.searchsorted(cum, q)), len(x) - 1)
        out[f"p{int(q * 100)}"] = float(x[i]) if min(i, len(x) - 1 - i) >= 10 else None
    return out


def wspearman(x, y, w):
    rx = pd.Series(x).rank().to_numpy()
    ry = pd.Series(y).rank().to_numpy()
    mx, my = np.average(rx, weights=w), np.average(ry, weights=w)
    cov = np.average((rx - mx) * (ry - my), weights=w)
    var = np.average((rx - mx) ** 2, weights=w) * np.average((ry - my) ** 2, weights=w)
    return float(cov / np.sqrt(var))


def cell(d, compute):
    n = households(d)
    return {"households": n, **(compute(d) if n >= MIN_HOUSEHOLDS else {})}


def hours_and_pay(d):
    hourly = (d.employment_income / d.hours_worked).to_numpy()
    out = {}
    for label, w in (
        ("weighted", d.household_weight.to_numpy()),
        ("unweighted", np.ones(len(d))),
    ):
        if w.sum() <= 0:
            continue
        out[label] = {
            "hourly_pay": wquantiles(hourly, w),
            "share_below_5": float((w * (hourly < 5)).sum() / w.sum()),
            "share_below_11_44": float((w * (hourly < 11.44)).sum() / w.sum()),
            "share_above_100": float((w * (hourly > 100)).sum() / w.sum()),
            "spearman_hours_pay": wspearman(
                d.hours_worked.to_numpy(), d.employment_income.to_numpy(), w
            ),
            "median_pay": wquantiles(d.employment_income.to_numpy(), w)["p50"],
            "share_pay_above_150k": float(
                (w * (d.employment_income > 150_000)).sum() / w.sum()
            ),
        }
    return out


def measure(path):
    with pd.HDFStore(path, mode="r") as store:
        person = store["/person"][
            ["person_household_id", "employment_status", "hours_worked", *INCOMES]
        ]
        household = store["/household"][
            [
                "household_id",
                "household_weight",
                "household_is_spi_synthetic",
                "household_is_capital_gains_clone",
                "household_is_cgt_band_donor",
            ]
        ]
    p = person.merge(household, left_on="person_household_id", right_on="household_id")
    p["survey"] = ~p.household_is_capital_gains_clone & ~p.household_is_cgt_band_donor
    p["source"] = np.where(p.household_is_spi_synthetic, "SPI", "FRS")
    status = p.employment_status.astype(str)
    p["group"] = "other"
    for group, statuses in STATUS_GROUPS.items():
        p.loc[status.isin(statuses), "group"] = group
    w = "household_weight"
    report = {"schema": "diag-551-spi-rows/1", "min_households": MIN_HOUSEHOLDS}

    report["households_weighted_m"] = {
        src: float(
            household.loc[household.household_is_spi_synthetic == spi, w].sum() / 1e6
        )
        for src, spi in (("FRS", False), ("SPI", True))
    }

    hp = {}
    for src in ("FRS", "SPI"):
        for label, statuses in (
            ("FT", ["FT_EMPLOYED"]),
            ("PT", ["PT_EMPLOYED"]),
            ("all", ["FT_EMPLOYED", "PT_EMPLOYED"]),
        ):
            d = p[
                (p.source == src)
                & status.isin(statuses)
                & (p.employment_income > 0)
                & (p.hours_worked > 0)
            ]
            hp[f"{src}|{label}"] = cell(d, hours_and_pay)
    report["employees_hours_and_pay"] = hp

    tail = {}
    for src in ("FRS", "SPI"):
        for threshold in (100_000, 150_000):
            d = p[(p.source == src) & (p.employment_income > threshold)]
            tail[f"{src}|{threshold}"] = cell(
                d,
                lambda d: {
                    "people_m_weighted": float(d[w].sum() / 1e6),
                    "people_unweighted": int(d.survey.sum()),
                    "pay_bn_weighted": float((d[w] * d.employment_income).sum() / 1e9),
                },
            )
    report["pay_top_tail"] = tail

    status_rows = {}
    for (src, group), d in p.groupby(["source", "group"]):

        def shares(d):
            out = {"people_m": float(d[w].sum() / 1e6)}
            for col, label in (
                ("employment_income", "share_pay_positive"),
                ("self_employment_income", "share_profit_positive"),
            ):
                positive = d[d[col] > 0]
                out[label] = (
                    float(positive[w].sum() / d[w].sum())
                    if households(positive) >= MIN_HOUSEHOLDS
                    else None
                )
            out["income_bn"] = {
                c.replace("_income", ""): float((d[w] * d[c]).sum() / 1e9)
                for c in INCOMES
            }
            return out

        status_rows[f"{src}|{group}"] = cell(d, shares)
    report["status_consistency"] = status_rows

    dividends = {}
    for src in ("FRS", "SPI"):
        d = p[(p.source == src) & (p.dividend_income > 0)]
        dividends[f"{src}|positive"] = cell(
            d,
            lambda d: {
                "people_m_weighted": float(d[w].sum() / 1e6),
                "dividends_bn_weighted": float((d[w] * d.dividend_income).sum() / 1e9),
                "median_dividend": wquantiles(
                    d.dividend_income.to_numpy(), d[w].to_numpy()
                )["p50"],
            },
        )
        both = d[d.employment_income > 0]
        dividends[f"{src}|pay_and_dividends"] = cell(
            both,
            lambda d: {
                "people_m_weighted": float(d[w].sum() / 1e6),
                "spearman_pay_dividends": wspearman(
                    d.employment_income.to_numpy(),
                    d.dividend_income.to_numpy(),
                    d[w].to_numpy(),
                ),
            },
        )
    report["dividends"] = dividends

    spi = p[p.source == "SPI"]
    se = spi.employment_status.astype(str).isin(STATUS_GROUPS["self_employed"])
    profit, pay = spi.self_employment_income, spi.employment_income
    rule = se | ((profit > 0) & (profit > pay))
    formula = profit != 0
    report["spi_gainful_self_employment"] = {
        name: cell(spi[m], lambda d: {"people_m_weighted": float(d[w].sum() / 1e6)})
        for name, m in (
            ("rule_only", rule & ~formula),
            ("formula_only", formula & ~rule),
            ("both", rule & formula),
        )
    }
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    with open(args.out, "w") as f:
        json.dump(measure(args.dataset), f, indent=1, sort_keys=True)


if __name__ == "__main__":
    main()
