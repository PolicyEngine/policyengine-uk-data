"""Verify the PR's rounded £0.8bn/£3bn claim from saved aggregates only."""

import json
from pathlib import Path


def main():
    evidence = Path.home() / "reviews/frs-cvpay-property-income-2026-09-30/impact"
    labels = ("main_matched", "main_matched_2", "main_a", "main_b")
    data = {
        label: json.loads((evidence / f"efrs_{label}.json").read_text())
        for label in labels
    }
    heads = {entry["policyengine_uk_head"] for entry in data.values()}
    assert heads == {"44240bd8d7d7e97f2980ccc8134e76e61541966f"}
    report = {
        "sources": str(evidence),
        "policyengine_uk_head": heads.pop(),
        "years": {},
    }
    for year in ("2025", "2026"):
        report["years"][year] = {}
        for metric in ("income_tax_bn", "household_net_income_bn"):
            values = {label: entry[year][metric] for label, entry in data.items()}
            low, high = min(values, key=values.get), max(values, key=values.get)
            report["years"][year][metric] = {
                "values_bn": values,
                "minimum_build": low,
                "maximum_build": high,
                "range_bn": values[high] - values[low],
            }
    assert (
        abs(report["years"]["2026"]["income_tax_bn"]["range_bn"] - 0.784596992) < 1e-10
    )
    assert (
        abs(report["years"]["2026"]["household_net_income_bn"]["range_bn"] - 2.95108608)
        < 1e-10
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
