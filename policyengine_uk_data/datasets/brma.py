"""Broad Rental Market Area (BRMA) assignment for FRS benefit units.

The FRS identifies only a household's region, but Local Housing Allowance
rates vary by BRMA. Each benefit unit is given a BRMA drawn within its
region, in proportion to the number of private-rented households in each
BRMA with the number of bedrooms matching its LHA category (shared
accommodation and one-bedroom categories both use one-bedroom homes).

The counts come from the censuses (England and Wales 2021, Scotland 2022,
Northern Ireland 2021) mapped to BRMAs; ``storage/BRMA_DATA_SOURCES.md``
gives the sources, method and validation. A BRMA that crosses a region
boundary appears once per region, holding the households on that side.

These weights replace the row counts of the 2019-20 LHA list of rents. Its
Scottish, Welsh and Northern Ireland rows were copies of English BRMAs'
lists, so their counts said nothing about those nations' rental markets.

Private renters who report a rent are then redrawn given that rent
(``assign_private_renter_brmas``). The probability of each BRMA is the census
share of private-rented homes with the household's number of bedrooms, times
the density of its rent on the BRMA's list of rents for homes of that size
(``storage/brma_private_rents.csv``). Reported rents differ from the rents on
the rent officers' lists: ``ReportedRentModel`` measures that difference from
the survey itself. Without this step a
household's rent says nothing about its Local Housing Allowance rate.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from policyengine_uk_data.storage import STORAGE_FOLDER

BRMA_HOUSEHOLDS_PATH = STORAGE_FOLDER / "brma_private_rented_households.csv"

# Census bedroom band whose private-rented households weight each LHA category.
LHA_CATEGORY_BEDROOMS = {"A": "1", "B": "1", "C": "2", "D": "3", "E": "4+"}


def load_brma_weights(path: Path = BRMA_HOUSEHOLDS_PATH) -> pd.DataFrame:
    """Return BRMA sampling weights by region and LHA category.

    Columns: ``region``, ``lha_category``, ``brma``, ``weight`` (private-rented
    households). Every row has a positive weight. Northern Ireland's census
    has no bedrooms question, so its rows (bedrooms ``all``) weight every
    category.
    """
    households = pd.read_csv(path, dtype={"bedrooms": str})
    bands = pd.DataFrame(
        LHA_CATEGORY_BEDROOMS.items(), columns=["lha_category", "bedrooms"]
    )
    by_band = bands.merge(households, on="bedrooms")
    any_band = households[households.bedrooms == "all"].merge(
        bands[["lha_category"]], how="cross"
    )
    weights = pd.concat([by_band, any_band], ignore_index=True)
    weights = weights[weights.households > 0]
    return weights.rename(columns={"households": "weight"})[
        ["region", "lha_category", "brma", "weight"]
    ].reset_index(drop=True)


def assign_brmas(
    region: np.ndarray,
    lha_category: np.ndarray,
    rng: np.random.Generator,
    weights: pd.DataFrame | None = None,
) -> np.ndarray:
    """Draw a BRMA for each benefit unit from its region × LHA category cell.

    Args:
        region: Region name of each benefit unit's household.
        lha_category: LHA category (A-E) of each benefit unit.
        rng: Generator for the draws.
        weights: Output of ``load_brma_weights``; loaded from storage if omitted.

    Raises:
        ValueError: If a benefit unit's cell has no BRMA with a positive weight.
    """
    if weights is None:
        weights = load_brma_weights()
    if hasattr(region, "decode_to_str"):  # EnumArray of region codes
        region = region.decode_to_str()
    region = np.asarray(region).astype(str)
    lha_category = np.asarray(lha_category).astype(str)
    brma = np.empty(len(region), dtype=object)
    for (cell_region, category), cell in weights.groupby(
        ["region", "lha_category"], sort=True
    ):
        mask = (region == cell_region) & (lha_category == category)
        if mask.any():
            p = cell.weight.to_numpy(dtype=float)
            brma[mask] = rng.choice(
                cell.brma.to_numpy(), size=mask.sum(), p=p / p.sum()
            )
    missing = pd.isna(brma)
    if missing.any():
        cells = sorted(set(zip(region[missing], lha_category[missing])))
        raise ValueError(f"No BRMA weights for region × LHA category cells {cells}.")
    return brma


def pick_household_brmas(
    brma: np.ndarray, household_id: np.ndarray, rng: np.random.Generator
) -> pd.Series:
    """Give each household the BRMA of one of its benefit units, chosen at random.

    ``brma`` and ``household_id`` are per benefit unit; returns a Series
    indexed by household ID.
    """
    units = pd.DataFrame({"brma": brma, "household_id": household_id})
    return units.groupby("household_id").brma.aggregate(
        lambda x: x.sample(n=1, random_state=rng).iloc[0]
    )


BRMA_RENTS_PATH = STORAGE_FOLDER / "brma_private_rents.csv"

# Census bedroom band of a home, and the list-of-rents category that covers
# self-contained homes of that size.
BEDROOM_BAND_CATEGORY = {"1": "B", "2": "C", "3": "D", "4+": "E"}


def bedroom_band(bedrooms: np.ndarray) -> np.ndarray:
    """Census bedroom band (``1``, ``2``, ``3``, ``4+``) for a number of bedrooms."""
    bedrooms = np.asarray(bedrooms, dtype=float)
    if not (np.isfinite(bedrooms) & (bedrooms >= 1)).all():
        raise ValueError("Every home needs at least one bedroom.")
    return np.where(
        bedrooms >= 4, "4+", np.minimum(bedrooms, 3).astype(int).astype(str)
    )


@dataclass(frozen=True)
class ReportedRentModel:
    """How households' reported rents relate to their BRMA's list of rents.

    The log of a reported rent is the log of a rent on the list for the
    household's BRMA and bedrooms, plus ``region_shift`` for its region, plus
    noise. A share ``below_market_share`` of households pay
    ``below_market_discount`` log points less than that, with noise standard
    deviation ``below_market_noise``; the rest have noise standard deviation
    ``noise``.
    """

    region_shift: dict[str, float]
    noise: float
    below_market_share: float
    below_market_discount: float
    below_market_noise: float


def _rent_cells(
    region, bedrooms, households: pd.DataFrame | None, rents: pd.DataFrame | None
):
    """Each household's BRMA prior, log median rent and log rent spread.

    Returns the BRMA names and three arrays of shape (households, BRMAs). The
    prior is the share of its region's private-rented households with its
    number of bedrooms that live in each BRMA. Outside the prior's support the
    median is 0 and the spread 1; those cells never contribute.
    """
    if households is None:
        households = pd.read_csv(BRMA_HOUSEHOLDS_PATH, dtype={"bedrooms": str})
    if rents is None:
        rents = pd.read_csv(BRMA_RENTS_PATH)
    bands = pd.Series(list(BEDROOM_BAND_CATEGORY), name="bedrooms")
    any_band = households[households.bedrooms == "all"].drop(columns="bedrooms")
    counts = pd.concat(
        [households[households.bedrooms != "all"], any_band.merge(bands, how="cross")]
    ).pivot_table(
        index=["region", "bedrooms"],
        columns="brma",
        values="households",
        aggfunc="sum",
        fill_value=0,
    )
    if hasattr(region, "decode_to_str"):  # EnumArray of region codes
        region = region.decode_to_str()
    region = np.asarray(region).astype(str)
    band = bedroom_band(bedrooms)
    cell = counts.index.get_indexer(pd.MultiIndex.from_arrays([region, band]))
    if (cell < 0).any():
        missing = sorted(set(zip(region[cell < 0], band[cell < 0])))
        raise ValueError(f"No BRMA weights for region × bedrooms cells {missing}.")
    category = counts.index.get_level_values("bedrooms").map(BEDROOM_BAND_CATEGORY)
    by_category = rents.set_index(["lha_category", "brma"])
    log_median = np.log(
        by_category.median_weekly_rent.unstack().reindex(
            category, columns=counts.columns
        )
    ).to_numpy()
    log_sd = (
        by_category.log_sd.unstack()
        .reindex(category, columns=counts.columns)
        .to_numpy()
    )
    supported = counts.to_numpy() > 0
    unlisted = supported & ~(np.isfinite(log_median) & (log_sd > 0))
    if unlisted[cell].any():
        rows, columns = np.nonzero(
            unlisted & np.isin(np.arange(len(counts)), cell)[:, None]
        )
        missing = sorted(set(zip(category[rows], counts.columns[columns])))
        raise ValueError(f"No list of rents for LHA category × BRMA cells {missing}.")
    prior = counts.to_numpy() / counts.to_numpy().sum(axis=1, keepdims=True)
    return (
        counts.columns.to_numpy(),
        prior[cell],
        np.where(supported, log_median, 0.0)[cell],
        np.where(supported, log_sd, 1.0)[cell],
    )


def _log_rent_density(log_rent, shift, log_median, log_sd, shape) -> np.ndarray:
    """Log density of each reported rent in each BRMA, up to a constant."""
    noise, share, discount, wide_noise = shape
    error = log_rent[:, None] - shift[:, None] - log_median

    def log_normal(error, variance):
        return -0.5 * error**2 / variance - 0.5 * np.log(variance)

    with np.errstate(divide="ignore"):
        return np.logaddexp(
            np.log1p(-share) + log_normal(error, log_sd**2 + noise**2),
            np.log(share) + log_normal(error + discount, log_sd**2 + wide_noise**2),
        )


def _shifts(region, model: ReportedRentModel) -> np.ndarray:
    if hasattr(region, "decode_to_str"):
        region = region.decode_to_str()
    region = np.asarray(region).astype(str)
    missing = sorted(set(region) - set(model.region_shift))
    if missing:
        raise ValueError(f"No reported-rent shift for regions {missing}.")
    return np.array([model.region_shift[r] for r in region], dtype=float)


def brma_probabilities(
    region: np.ndarray,
    bedrooms: np.ndarray,
    weekly_rent: np.ndarray,
    model: ReportedRentModel,
    households: pd.DataFrame | None = None,
    rents: pd.DataFrame | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Each private-renting household's probability of living in each BRMA.

    P(BRMA | region, bedrooms, rent) is proportional to the census
    private-rented households in the BRMA with that many bedrooms, times the
    density of the household's rent under ``model`` and the BRMA's list of
    rents for homes of that size.

    Args:
        region: Region name of each household.
        bedrooms: Number of bedrooms in each household's home (at least 1).
        weekly_rent: Weekly rent each household reports (positive).
        model: Output of ``fit_reported_rent_model``.
        households: Census table (``BRMA_HOUSEHOLDS_PATH``) if omitted.
        rents: List-of-rents table (``BRMA_RENTS_PATH``) if omitted.

    Returns:
        BRMA names, and probabilities of shape (households, BRMAs). Each row
        sums to 1 and is zero outside the household's region × bedrooms cell.

    Raises:
        ValueError: If a rent is not positive, or a household's region,
            region × bedrooms cell or a BRMA in that cell has no data.
    """
    weekly_rent = np.asarray(weekly_rent, dtype=float)
    if not (np.isfinite(weekly_rent) & (weekly_rent > 0)).all():
        raise ValueError("Every household needs a positive weekly rent.")
    brmas, prior, log_median, log_sd = _rent_cells(region, bedrooms, households, rents)
    shape = (
        model.noise,
        model.below_market_share,
        model.below_market_discount,
        model.below_market_noise,
    )
    with np.errstate(divide="ignore"):
        log_p = np.log(prior) + _log_rent_density(
            np.log(weekly_rent), _shifts(region, model), log_median, log_sd, shape
        )
    p = np.exp(log_p - log_p.max(axis=1, keepdims=True))
    return brmas, p / p.sum(axis=1, keepdims=True)


def fit_reported_rent_model(
    region: np.ndarray,
    bedrooms: np.ndarray,
    weekly_rent: np.ndarray,
    weight: np.ndarray,
    households: pd.DataFrame | None = None,
    rents: pd.DataFrame | None = None,
) -> ReportedRentModel:
    """Fit ``ReportedRentModel`` to private renters' reported rents.

    Maximises the weighted likelihood of the rents, each household's BRMA
    being unobserved with the census prior. The fit has one shift per region
    and four shared shape parameters, each rounded to three decimal places so
    that the BRMA probabilities do not depend on the optimiser's last digits.

    Raises:
        RuntimeError: If the optimiser does not converge.
    """
    from scipy.optimize import minimize

    if hasattr(region, "decode_to_str"):
        region = region.decode_to_str()
    region = np.asarray(region).astype(str)
    weight = np.asarray(weight, dtype=float)
    weekly_rent = np.asarray(weekly_rent, dtype=float)
    positive = np.isfinite(weekly_rent) & (weekly_rent > 0)
    if not (positive.all() and (weight >= 0).all() and weight.sum() > 0):
        raise ValueError("Rents must be positive and weights non-negative.")
    log_rent = np.log(weekly_rent)
    _, prior, log_median, log_sd = _rent_cells(region, bedrooms, households, rents)
    # Keep each household's supported BRMAs only: far fewer than all 200.
    widest = (prior > 0).sum(axis=1).max()
    keep = np.argsort(-prior, axis=1, kind="stable")[:, :widest]
    prior, log_median, log_sd = (
        np.take_along_axis(a, keep, axis=1) for a in (prior, log_median, log_sd)
    )
    names, index = np.unique(region, return_inverse=True)
    with np.errstate(divide="ignore"):
        log_prior = np.log(prior)

    def loss(x):
        shape = (x[-4], x[-3], x[-2], x[-4] + x[-1])
        log_p = log_prior + _log_rent_density(
            log_rent, x[index], log_median, log_sd, shape
        )
        peak = log_p.max(axis=1)
        likelihood = peak + np.log(np.exp(log_p - peak[:, None]).sum(axis=1))
        return -(weight * likelihood).sum() / weight.sum()

    # Start each region's shift at its median gap to the prior-mean list median.
    gap = log_rent - (prior * log_median).sum(axis=1)
    start = [np.median(gap[index == i]) for i in range(len(names))]
    result = minimize(
        loss,
        np.array(start + [0.2, 0.15, 0.5, 0.5]),
        method="L-BFGS-B",
        bounds=[(-3, 3)] * len(names) + [(0.01, 2), (0.001, 0.5), (0, 3), (0.05, 3)],
    )
    if not result.success:
        raise RuntimeError(f"Reported-rent model did not converge: {result.message}")
    x = np.round(result.x, 3)
    return ReportedRentModel(
        region_shift=dict(zip(names.tolist(), x[: len(names)].tolist())),
        noise=float(x[-4]),
        below_market_share=float(x[-3]),
        below_market_discount=float(x[-2]),
        below_market_noise=float(round(x[-4] + x[-1], 3)),
    )


def draw_brmas(
    brmas: np.ndarray, probabilities: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Draw one BRMA per row of ``probabilities``; never one with probability 0."""
    cumulative = np.cumsum(probabilities, axis=1)
    cumulative /= cumulative[:, -1:]
    drawn = (cumulative <= rng.random(len(probabilities))[:, None]).sum(axis=1)
    return np.asarray(brmas, dtype=object)[drawn]


def assign_private_renter_brmas(
    region: np.ndarray,
    bedrooms: np.ndarray,
    weekly_rent: np.ndarray,
    weight: np.ndarray,
    rng: np.random.Generator,
    households: pd.DataFrame | None = None,
    rents: pd.DataFrame | None = None,
) -> np.ndarray:
    """Draw each private-renting household's BRMA given its rent and bedrooms.

    Fits ``ReportedRentModel`` to these households, then draws each one's BRMA
    from ``brma_probabilities``. A household that pays more than is usual for
    its region and number of bedrooms is more likely to land in a dearer BRMA,
    so its rent and its Local Housing Allowance rate move together.
    """
    model = fit_reported_rent_model(
        region, bedrooms, weekly_rent, weight, households, rents
    )
    brmas, p = brma_probabilities(
        region, bedrooms, weekly_rent, model, households, rents
    )
    return draw_brmas(brmas, p, rng)
