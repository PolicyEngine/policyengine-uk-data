import logging

import numpy as np
import pandas as pd
from policyengine_uk.data import UKSingleYearDataset

# Fit a spline to each income band's percentiles
from scipy.interpolate import UnivariateSpline

from policyengine_uk_data.storage import STORAGE_FOLDER
from policyengine_uk_data.targets.sources.hmrc_cgt import load_size_bands
from policyengine_uk_data.utils.stack import stack_datasets

capital_gains = pd.read_csv(
    STORAGE_FOLDER / "capital_gains_distribution_advani_summers.csv.gz"
)
capital_gains["maximum_total_income"] = capital_gains.minimum_total_income.shift(
    -1
).fillna(np.inf)

# Silence verbose logging
logging.getLogger("root").setLevel(logging.WARNING)

# --- Has-gains twins --------------------------------------------------------
#
# Advani & Summers give, for each band of total income, the share of people
# with gains and the percentiles of those gains. Each household's first adult
# draws a gain from their band's distribution. Where that gain is one HMRC's
# statistics describe (above the lowest size-of-gain band limit, the annual
# exempt amount), the household is stacked a second time as a "has-gains
# twin" carrying the gain, and the household's mass is split between the two
# records using the table's incidence propensity adjusted for assignment
# to first adults. Rejecting draws at or under the limit lowers realised
# incidence. Calibration then moves weight between a household and its twin,
# pulled by the HMRC targets in targets/sources/hmrc_cgt.py, without
# disturbing any other target.
#
# Draws at or under that limit, including the net losses in the table's
# lower percentiles, get no twin. HMRC publishes liable taxpayers only, so
# no target describes those people, and a record no target describes takes
# whatever weight suits the other targets: the 1.58.0 build ended with 1.4m
# people holding £15bn of losses, and 0.4m with gains under the exempt
# amount.
#
# The original and twin carry complementary household_calibration_prior
# values into calibration. Survey weights are also split; zero-weight
# originals stay at zero and their twins receive prior-equivalent weights.
# These weights must not be used to regenerate the already split priors.
# Twins of zero-weight rows used to
# keep a zero weight, and so received a full zero-weight prior each, the same
# as their no-gains originals. The calibration therefore started with half
# of the SPI rows' mass holding gains: 6.4m CGT taxpayers and £365bn of gains
# against targets of 378,000 and £68bn.


def _first_adult_incomes(dataset: UKSingleYearDataset) -> tuple[np.ndarray, np.ndarray]:
    """Each person's total income, and whether they are their household's
    first adult."""
    from policyengine_uk import Microsimulation

    sim = Microsimulation(dataset=dataset)
    return (
        sim.calculate("total_income").values,
        sim.calculate("adult_index").values == 1,
    )


def _income_band(total_income: np.ndarray) -> np.ndarray:
    """Row of the Advani & Summers table for each income (-1 below the first)."""
    return (
        np.searchsorted(
            capital_gains.minimum_total_income.values, total_income, side="right"
        )
        - 1
    )


def _households(dataset: UKSingleYearDataset, household_ids) -> UKSingleYearDataset:
    """A copy of the households with these ids, in the dataset's order."""
    person = dataset.person[dataset.person.person_household_id.isin(household_ids)]
    return UKSingleYearDataset(
        person=person.reset_index(drop=True).copy(),
        benunit=dataset.benunit[
            dataset.benunit.benunit_id.isin(person.person_benunit_id)
        ]
        .reset_index(drop=True)
        .copy(),
        household=dataset.household[dataset.household.household_id.isin(household_ids)]
        .reset_index(drop=True)
        .copy(),
        fiscal_year=dataset.time_period,
    )


def draw_gains(income_band: np.ndarray, gainer: np.ndarray) -> np.ndarray:
    """Draw a gain for each gainer from their income band's distribution.

    Assumes that the capital gains distribution is the same for all years.
    """
    gains = np.zeros(len(income_band))

    # Draw imputation quantiles from a seeded generator so the build is
    # reproducible: an unseeded global np.random made capital gains (and hence
    # CGT revenue) differ between otherwise identical builds.
    cg_rng = np.random.default_rng(0)

    for i, row in enumerate(capital_gains.itertuples()):
        spline = UnivariateSpline(
            [0.05, 0.1, 0.25, 0.5, 0.75, 0.90, 0.95],
            [row.p05, row.p10, row.p25, row.p50, row.p75, row.p90, row.p95],
            k=1,
        )
        in_band = gainer & (income_band == i)
        gains[in_band] = spline(cg_rng.random(int(in_band.sum())))

    return gains


def has_gains_shares(
    income_band: np.ndarray, first_adult: np.ndarray, person_weight: np.ndarray
) -> np.ndarray:
    """Twin propensity among first adults, before rejecting small gains.

    The table gives the share of all people in a band with gains, but only
    first adults are given gains, so the share among them is the table's
    share times the band's people per first adult.
    Rejecting draws at or under the HMRC band limit lowers realised incidence.
    """
    n_bands = len(capital_gains)
    in_table = income_band >= 0
    people = np.bincount(
        income_band[in_table], weights=person_weight[in_table], minlength=n_bands
    )
    gainers = in_table & first_adult
    first_adults = np.bincount(
        income_band[gainers], weights=person_weight[gainers], minlength=n_bands
    )
    share = np.divide(
        capital_gains.percent_with_gains.values * people,
        first_adults,
        out=np.zeros(n_bands),
        where=first_adults > 0,
    )
    if (share >= 1).any():
        raise ValueError(
            "An income band has too few first adults to hold its share of "
            "people with gains."
        )
    return share


def impute_capital_gains(dataset: UKSingleYearDataset) -> UKSingleYearDataset:
    """Stack a has-gains twin for each household whose first adult draws a
    gain above the lowest HMRC size-of-gain band limit."""
    from policyengine_uk_data.utils.calibrate import (
        initialize_weight_priors,
        prior_equivalent_weights,
    )

    dataset = dataset.copy()
    total_income, first_adult = _first_adult_incomes(dataset)
    household_of_person = pd.Index(dataset.household.household_id).get_indexer(
        dataset.person.person_household_id
    )
    weight = dataset.household.household_weight.to_numpy(dtype=float)
    priors = initialize_weight_priors(
        weight, prior_weights=dataset.household.get("household_calibration_prior")
    )

    income_band = _income_band(total_income)
    gains = draw_gains(income_band, first_adult)
    share = has_gains_shares(income_band, first_adult, weight[household_of_person])

    gainer = first_adult & (gains > load_size_bands().lower_limit.min())
    twin_weight = np.zeros(len(weight))
    twin_weight[household_of_person[gainer]] = share[income_band[gainer]]
    twin_prior = priors * twin_weight
    twin_weight *= prior_equivalent_weights(weight)
    has_twin = twin_weight > 0

    logging.info("Stacking %s has-gains twins", int(has_twin.sum()))
    twin = _households(dataset, dataset.household.household_id[has_twin])
    twin.person["capital_gains"] = gains[has_twin[household_of_person]]
    twin.household["household_weight"] = twin_weight[has_twin]
    twin.household["household_calibration_prior"] = twin_prior[has_twin]
    twin.household["household_is_capital_gains_clone"] = True

    dataset.person["capital_gains"] = 0.0
    dataset.household["household_weight"] = np.where(
        weight > 0, weight - twin_weight, 0.0
    )
    dataset.household["household_is_capital_gains_clone"] = False
    dataset.household["household_calibration_prior"] = priors - twin_prior

    data = stack_datasets(dataset, twin)
    data.validate()
    return data


# --- HMRC size-band donor stack ------------------------------------------
#
# The Advani & Summers spline stops at each income band's p95, so the twins
# above cannot carry the large gains HMRC observes: no drawn gain exceeds
# ~£2m, and fewer than a hundred exceed £500,000, while HMRC attributes
# three quarters of individuals' chargeable gains in 2024-25 to gains of
# £500,000 or more (Capital Gains Tax statistics, Table 2.1a). Calibration
# cannot fix that on its own — there are too few records carrying large
# gains for it to upweight.
#
# Following the SPI synthetic-household precedent in income.py, we stack a
# small set of donor households for each HMRC size-of-gain band from
# £500,000 up, each carrying that band's published mean gain. Calibration
# then adjusts how much weight each band's donors receive, pulled by the
# per-band HMRC targets in targets/sources/hmrc_cgt.py.
#
# The bands below £500,000 need no donors: the twins give each of them
# hundreds or thousands of records. Donors there duplicated the twins. In
# the 1.58.0 build, which stacked donors from £12,300 up, the donors of the
# four bands from £12,300 to £250,000 ended with under 1% of the weight they
# entered with.
#
# Donors enter at the weight that reproduces the published band —
# band taxpayer count / DONORS_PER_BAND — NOT at zero. Zero-weight rows are
# given an equal share of prior mass by calibrate.initialize_weight_priors
# (the SPI mechanism), which for donors carrying multi-million-pound gains
# injected ~£2.3trn of initial gains in a CI build and left calibration
# 18x over the band targets. Whatever the donors and twins of a band weigh
# together, calibration starts with the band's taxpayer count on its target
# (calibrate.rake_priors_to_targets).
#
# Donor households are sampled with probability proportional to
# percent_with_gains at their first adult's total income (the same Advani &
# Summers table used above), so large gains land on the income profiles
# most likely to realise gains rather than on uniformly random households.

# Few, heavy donors rather than many light ones: the first full build with
# 300 donors per band left the £2m+ bands 94-97% under target, and failed
# the built-dataset tests.
DONORS_PER_BAND = 30
_DONOR_SEED = 1
_MIN_DONOR_BAND_LOWER = 500_000


def load_hmrc_size_bands() -> pd.DataFrame:
    """HMRC Table 2.1a size-of-gain bands, with each band's mean gain."""
    bands = load_size_bands()
    bands["taxpayers"] = bands.taxpayers_thousands * 1e3
    bands["gains"] = bands.gains_gbp_millions * 1e6
    bands["mean_gain"] = bands.gains / bands.taxpayers
    return bands


def load_donor_bands() -> pd.DataFrame:
    """The bands whose records are donors."""
    bands = load_hmrc_size_bands()
    return bands[bands.lower_limit >= _MIN_DONOR_BAND_LOWER].reset_index(drop=True)


def stack_cgt_band_donors(dataset: UKSingleYearDataset) -> UKSingleYearDataset:
    """Stack donor households carrying HMRC band mean gains at band-exact
    initial weights."""
    dataset = dataset.copy()
    dataset.household["household_is_cgt_band_donor"] = False
    if "capital_gains" not in dataset.person.columns:
        dataset.person["capital_gains"] = 0.0

    bands = load_donor_bands()
    rng = np.random.default_rng(_DONOR_SEED)

    total_income, first_adult = _first_adult_incomes(dataset)
    # A has-gains twin is the same household as its original, so only the
    # original is a candidate.
    if "household_is_capital_gains_clone" in dataset.household.columns:
        twins = dataset.household.household_id[
            dataset.household.household_is_capital_gains_clone.astype(bool)
        ]
        first_adult &= ~dataset.person.person_household_id.isin(twins).values

    fa_household_id = dataset.person.person_household_id.values[first_adult]
    propensity = capital_gains.percent_with_gains.values[
        np.clip(_income_band(total_income[first_adult]), 0, len(capital_gains) - 1)
    ]

    n_donors = DONORS_PER_BAND * len(bands)
    selected = rng.choice(
        fa_household_id,
        size=min(n_donors, len(fa_household_id)),
        replace=False,
        p=propensity / propensity.sum(),
    )
    band_of_household = dict(
        zip(
            selected,
            np.repeat(bands.mean_gain.values, DONORS_PER_BAND)[: len(selected)],
        )
    )
    # Initial weight per donor reproduces the band's published taxpayer
    # count exactly (see module comment above).
    weight_of_household = dict(
        zip(
            selected,
            np.repeat(bands.taxpayers.values / DONORS_PER_BAND, DONORS_PER_BAND)[
                : len(selected)
            ],
        )
    )

    donor = _households(dataset, selected)
    donor.household["household_weight"] = (
        donor.household.household_id.map(weight_of_household).fillna(0.0).values
    )
    donor.household["household_is_cgt_band_donor"] = True
    if "household_calibration_prior" in dataset.household:
        # Donors add their published band mass, not the copied source prior.
        donor.household["household_calibration_prior"] = donor.household[
            "household_weight"
        ]

    # The band mean gain goes to each donor household's first adult;
    # everyone else in the household carries no gain, mirroring the
    # one-gainer-per-household convention of the imputation above.
    donor_first_adult = first_adult[
        dataset.person.person_household_id.isin(selected).values
    ]
    donor_gains = (
        donor.person.person_household_id.map(band_of_household).fillna(0.0).values
    )
    donor.person["capital_gains"] = np.where(donor_first_adult, donor_gains, 0.0)

    data = stack_datasets(dataset, donor)
    data.validate()
    return data
