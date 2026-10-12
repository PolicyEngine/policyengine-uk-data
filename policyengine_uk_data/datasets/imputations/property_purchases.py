"""Impute one annual purchase from each kind of household property stock.

HMRC targets cover England and Northern Ireland. Independent seeded event
draws are calibrated there; the resulting rates also apply in Scotland and
Wales. Prices are separate seeded draws, never the household's entire stock
by default. Assumptions and sources live in stochastic parameters.
"""

import numpy as np

from policyengine_uk_data.parameters import load_parameter

SDLT_REGIONS = (
    "NORTH_EAST",
    "NORTH_WEST",
    "YORKSHIRE",
    "EAST_MIDLANDS",
    "WEST_MIDLANDS",
    "EAST_OF_ENGLAND",
    "LONDON",
    "SOUTH_EAST",
    "SOUTH_WEST",
    "NORTHERN_IRELAND",
)


def impute_property_purchase_values(
    stock: np.ndarray,
    weights: np.ndarray,
    sdlt_mask: np.ndarray,
    year: int,
    property_type: str,
) -> np.ndarray:
    """Return seeded purchase prices, fitting weighted events and capped value.

    Purchase hazard is proportional to stock, an explicit portfolio-turnover
    assumption. The closest weighted count determines the event threshold.
    The value target is fitted by multiplying sampled band-average prices,
    with the stock cap applied throughout. Zero-weight donors never purchase.
    """
    stock = np.asarray(stock, dtype=float)
    weights = np.asarray(weights, dtype=float)
    sdlt_mask = np.asarray(sdlt_mask, dtype=bool)
    if (
        stock.ndim != 1
        or stock.shape != weights.shape
        or stock.shape != sdlt_mask.shape
    ):
        raise ValueError(
            "Property stocks, weights and SDLT mask must be aligned vectors"
        )
    if (
        not np.all(np.isfinite(stock))
        or not np.all(np.isfinite(weights))
        or np.any(stock < 0)
        or np.any(weights < 0)
    ):
        raise ValueError("Property stocks and weights must be finite and non-negative")

    assumptions = load_parameter("stochastic", "additional_property_purchases", year)[
        property_type
    ]
    target_count = assumptions["transactions"]
    target_value = assumptions["value"]
    minimum_price = assumptions["minimum_price"]
    owners = (stock >= minimum_price) & (weights > 0)
    eligible = np.flatnonzero(owners & sdlt_mask)
    if weights[eligible].sum() < target_count:
        raise ValueError("Insufficient weighted property owners for purchase target")

    reference_stock = np.average(stock[eligible], weights=weights[eligible])
    hazard = (np.maximum(stock, minimum_price) / reference_stock) ** assumptions[
        "purchase_stock_exponent"
    ]
    uniform = np.random.default_rng(assumptions["purchase_seed"]).random(len(stock))
    events = -np.log1p(-uniform) / hazard
    ranked = eligible[np.argsort(events[eligible], kind="stable")]
    counts = np.concatenate(([0.0], np.cumsum(weights[ranked])))
    n_purchases = int(np.argmin(np.abs(counts - target_count)))
    if n_purchases == len(ranked):
        threshold = np.inf
    elif n_purchases == 0:
        threshold = 0.0
    else:
        threshold = (events[ranked[n_purchases - 1]] + events[ranked[n_purchases]]) / 2
    purchased = owners & (events < threshold)
    fitted_weights = weights * (purchased & sdlt_mask)
    capacity = np.dot(stock, fitted_weights)
    if capacity < target_value:
        raise ValueError(
            "Selected property stock cannot support the purchase value target"
        )
    if minimum_price * fitted_weights.sum() > target_value:
        raise ValueError("Purchase value target is below the minimum transaction price")

    bands = assumptions["price_bands"]
    band_counts = np.array([band["transactions"] for band in bands], dtype=float)
    band_prices = np.array([band["value"] for band in bands]) / band_counts
    price_rng = np.random.default_rng(assumptions["price_seed"])
    prices = price_rng.choice(
        band_prices, size=len(stock), p=band_counts / band_counts.sum()
    )

    def capped_prices(scale):
        return np.minimum(stock, np.maximum(minimum_price, prices * scale))

    def weighted_value(scale):
        return np.dot(capped_prices(scale), fitted_weights)

    lower, upper = 0.0, 1.0
    while weighted_value(upper) < target_value:
        upper *= 2
    for _ in range(80):
        midpoint = (lower + upper) / 2
        if weighted_value(midpoint) < target_value:
            lower = midpoint
        else:
            upper = midpoint
    return np.where(purchased, capped_prices(upper), 0.0)


def impute_property_purchases(dataset, year=None):
    """Add annual purchase inputs after wealth imputation, before calibration."""
    dataset = dataset.copy()
    year = int(dataset.time_period) if year is None else year
    household = dataset.household
    sdlt_mask = household["region"].astype(str).isin(SDLT_REGIONS).values
    for property_type, stock_name, purchase_name in (
        (
            "additional_residential",
            "other_residential_property_value",
            "additional_residential_property_purchased",
        ),
        (
            "non_residential",
            "non_residential_property_value",
            "non_residential_property_purchased",
        ),
    ):
        household[purchase_name] = impute_property_purchase_values(
            household[stock_name].values,
            household["household_weight"].values,
            sdlt_mask,
            year,
            property_type,
        )
    return dataset
