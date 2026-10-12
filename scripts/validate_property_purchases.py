"""Validate purchase imputation on a local enhanced FRS, printing aggregates only.

Run from the repository root with the desired policyengine-uk on PYTHONPATH::

    python scripts/validate_property_purchases.py /path/enhanced_frs_2023_24.h5 \
        --years 2023 2024 2025

Only four household columns are read. No dataset is downloaded or written.
Later years uprate stocks and weights with the model's dataset indices before
re-imputing purchases against that year's assumptions. This is a validation
of annual imputation, rather than a rebuilt or recalibrated dataset.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Make direct execution from scripts/ resolve this checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from policyengine_uk import parameters as model_parameters

from policyengine_uk_data.datasets.imputations.property_purchases import (
    SDLT_REGIONS,
    impute_property_purchase_values,
)
from policyengine_uk_data.parameters import load_parameter

PROPERTY_STOCKS = {
    "additional_residential": "other_residential_property_value",
    "non_residential": "non_residential_property_value",
}
STOCK_INDEX = "gov.economic_assumptions.yoy_growth.obr.per_capita.gdp"
WEIGHT_INDEX = "gov.economic_assumptions.yoy_growth.ons.population"


def growth_factor(parameters, index, source_year, year):
    """Use the same year-on-year factors as dataset projection."""
    return float(
        np.prod(
            [
                1 + parameters.get_child(index)(str(y))
                for y in range(source_year + 1, year + 1)
            ]
        )
    )


def validate(household, source_year, year, parameters):
    """Return weighted E+NI aggregates and checks; never return record values."""
    if year < source_year:
        raise ValueError("Validation years must not precede the dataset year")
    stock_factor = growth_factor(parameters, STOCK_INDEX, source_year, year)
    weight_factor = growth_factor(parameters, WEIGHT_INDEX, source_year, year)
    weights = household.household_weight.to_numpy(dtype=float) * weight_factor
    sdlt_mask = household.region.astype(str).isin(SDLT_REGIONS).to_numpy()
    stamp_duty = parameters(str(year)).gov.hmrc.stamp_duty
    rows = []
    for property_type, stock_name in PROPERTY_STOCKS.items():
        assumptions = load_parameter("stochastic", "additional_property_purchases", year)[
            property_type
        ]
        stock = household[stock_name].to_numpy(dtype=float) * stock_factor
        prices = impute_property_purchase_values(
            stock, weights, sdlt_mask, year, property_type
        )
        repeat = impute_property_purchase_values(
            stock, weights, sdlt_mask, year, property_type
        )
        purchased = prices > 0
        minimum = assumptions["minimum_price"]
        eligible = (stock >= minimum) & (weights > 0)
        fitted_weights = weights * (purchased & sdlt_mask)
        transactions = float(fitted_weights.sum())
        value = float(np.dot(prices, weights * sdlt_mask))
        checks = {
            "reproducible": bool(np.array_equal(prices, repeat)),
            "owners_only": bool(np.all(~purchased | eligible)),
            "price_bounds": bool(np.all(prices <= stock)),
            "minimum_price": bool(np.all(prices[purchased] >= minimum)),
            "transactions_within_5_percent": bool(
                abs(transactions / assumptions["transactions"] - 1) <= 0.05
            ),
            "value_within_5_percent": bool(
                abs(value / assumptions["value"] - 1) <= 0.05
            ),
        }
        if not all(checks.values()):
            raise ValueError(f"Aggregate validation failed for {year} {property_type}")
        if property_type == "additional_residential":
            purchase = stamp_duty.residential.purchase.additional
            liability = purchase.rate.calc(prices) * (prices >= purchase.min)
        else:
            liability = stamp_duty.non_residential.purchase.calc(prices)
        bands = assumptions["price_bands"]
        counts = np.array([band["transactions"] for band in bands], dtype=float)
        means = np.array([band["value"] for band in bands]) / counts
        raw_prices = np.random.default_rng(assumptions["price_seed"]).choice(
            means, size=len(stock), p=counts / counts.sum()
        )
        uncapped = purchased & (prices < stock) & (prices > minimum)
        price_scale = (
            float(np.median(prices[uncapped] / raw_prices[uncapped]))
            if uncapped.any()
            else None
        )
        rows.append(
            {
                "year": year,
                "property_type": property_type,
                "target_transactions": assumptions["transactions"],
                "transactions": transactions,
                "target_value": assumptions["value"],
                "value": value,
                "reference_gross_receipts": assumptions["revenue"],
                "model_gross_sdlt": float(np.dot(liability, weights * sdlt_mask)),
                "two_percentage_point_hrad_exposure": (
                    0.02 * value if property_type == "additional_residential" else None
                ),
                "eligible_weighted_owners": float(np.dot(eligible, weights * sdlt_mask)),
                "eligible_mean_stock": float(
                    np.dot(stock, weights * (eligible & sdlt_mask))
                    / np.dot(eligible, weights * sdlt_mask)
                ),
                "selected_stock_capacity": float(np.dot(stock, fitted_weights)),
                "weighted_purchase_rate": float(
                    transactions / np.dot(eligible, weights * sdlt_mask)
                ),
                "price_scale": price_scale,
                "weighted_stock_capped_share": float(
                    np.dot(purchased & (prices == stock), weights * sdlt_mask)
                    / transactions
                ),
                "stock_uprating_factor": stock_factor,
                "weight_uprating_factor": weight_factor,
                "checks": checks,
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--years", type=int, nargs="+")
    args = parser.parse_args()
    source_year = int(pd.read_hdf(args.dataset, key="time_period").iloc[0])
    household = pd.read_hdf(
        args.dataset,
        key="household",
        columns=["household_weight", "region", *PROPERTY_STOCKS.values()],
    )
    rows = [
        row
        for year in args.years or [source_year]
        for row in validate(household, source_year, year, model_parameters)
    ]
    print(
        json.dumps(
            {
                "source_year": source_year,
                "scope": "England and Northern Ireland; weighted aggregates only",
                "receipts_note": (
                    "Model gross SDLT uses the policyengine-uk 1 January higher-rates "
                    "scale (the full tax, including standard rates). HMRC receipts "
                    "cover a financial year, policy changes and replacement-home "
                    "refund timing, so they are reference benchmarks, not fitted targets."
                ),
                "results": rows,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
