import pandas as pd
import numpy as np
from policyengine_uk_data.utils.stack import stack_datasets

# Fit a spline to each income band's percentiles
from scipy.interpolate import UnivariateSpline

from policyengine_uk_data.storage import STORAGE_FOLDER

import torch
from torch.optim import Adam
from policyengine_uk.data import UKSingleYearDataset
import logging

capital_gains = pd.read_csv(
    STORAGE_FOLDER / "capital_gains_distribution_advani_summers.csv.gz"
)
capital_gains["maximum_total_income"] = capital_gains.minimum_total_income.shift(
    -1
).fillna(np.inf)

# Silence verbose logging
logging.getLogger("root").setLevel(logging.WARNING)


def impute_cg_to_doubled_dataset(
    dataset: UKSingleYearDataset,
) -> tuple[np.ndarray, np.ndarray]:
    """Assumes that the capital gains distribution is the same for all years."""

    from policyengine_uk import Microsimulation

    sim = Microsimulation(dataset=dataset)
    ti = sim.calculate("total_income").values
    household_weight = sim.calculate("household_weight").values
    first_half = (
        np.concatenate(
            [
                np.ones(len(household_weight) // 2),
                np.zeros(len(household_weight) // 2),
            ]
        )
        > 0
    )
    # Give capital gains to one adult aged 15+ in each household
    adult_index = sim.calculate("adult_index").values == 1
    in_person_second_half = np.zeros(len(ti)) > 0
    in_person_second_half[len(ti) // 2 :] = True
    has_cg = np.zeros(len(ti)) > 0
    has_cg[adult_index & in_person_second_half] = True
    blend_factor = torch.tensor(
        np.zeros(first_half.sum()), requires_grad=True, dtype=torch.float32
    )
    household_weight = torch.tensor(household_weight, dtype=torch.float32)
    sigmoid = torch.nn.Sigmoid()

    def loss(blend_factor):
        loss = 0

        blended_household_weight = household_weight.clone()
        adjusted_blend_factor = sigmoid(blend_factor)
        blended_household_weight[first_half] = (
            adjusted_blend_factor * blended_household_weight[first_half]
        )
        blended_household_weight[~first_half] = (
            1 - adjusted_blend_factor
        ) * blended_household_weight[first_half]
        for i in range(len(capital_gains)):
            lower = capital_gains.minimum_total_income.iloc[i]
            upper = capital_gains.maximum_total_income.iloc[i]
            true_pct_with_gains = capital_gains.percent_with_gains.iloc[i]

            ti_in_range = (ti >= lower) * (ti < upper)
            cg_in_income_range = has_cg * ti_in_range
            household_ti_in_range_count = torch.tensor(
                sim.map_result(ti_in_range, "person", "household", how="sum")
            )
            household_cg_in_income_range_count = torch.tensor(
                sim.map_result(cg_in_income_range, "person", "household", how="sum")
            )
            pred_ti_in_range = (
                blended_household_weight * household_ti_in_range_count
            ).sum()
            pred_cg_in_income_range = (
                blended_household_weight * household_cg_in_income_range_count
            ).sum()
            pred_pct_with_gains = pred_cg_in_income_range / torch.clip(
                pred_ti_in_range, 1
            )
            loss += (pred_pct_with_gains - true_pct_with_gains) ** 2

        return loss

    optimiser = Adam([blend_factor], lr=1e-1)
    progress = range(100)
    logging.info("Splitting household weights into has-gains and no-gains")
    for i in progress:
        optimiser.zero_grad()
        loss_value = loss(blend_factor)
        loss_value.backward()
        optimiser.step()
        if loss_value.item() < 1e-3:
            break

    new_household_weight = household_weight.detach().numpy()
    original_household_weight = new_household_weight.copy()
    blend_factor = sigmoid(blend_factor).detach().numpy()
    new_household_weight[first_half] = (
        blend_factor * original_household_weight[first_half]
    )
    new_household_weight[~first_half] = (1 - blend_factor) * original_household_weight[
        first_half
    ]

    # Impute actual capital gains amounts given gains
    new_cg = np.zeros(len(ti))

    logging.info("Imputing capital gains among those with gains")

    # Draw imputation quantiles from a seeded generator so the build is
    # reproducible: an unseeded global np.random made capital gains (and hence
    # CGT revenue) differ between otherwise identical builds.
    cg_rng = np.random.default_rng(0)

    for i in range(len(capital_gains)):
        row = capital_gains.iloc[i]
        spline = UnivariateSpline(
            [0.05, 0.1, 0.25, 0.5, 0.75, 0.90, 0.95],
            [row.p05, row.p10, row.p25, row.p50, row.p75, row.p90, row.p95],
            k=1,
        )
        lower = row.minimum_total_income
        upper = row.maximum_total_income
        ti_in_range = (ti >= lower) * (ti < upper)
        in_target_range = has_cg * ti_in_range > 0
        quantiles = cg_rng.random(int(in_target_range.sum()))
        pred_capital_gains = spline(quantiles)
        new_cg[in_target_range] = pred_capital_gains

    return new_cg, new_household_weight


def impute_capital_gains(dataset: UKSingleYearDataset) -> UKSingleYearDataset:
    dataset = dataset.copy()
    dataset.household["household_is_capital_gains_clone"] = False
    zero_weight_copy = dataset.copy()
    zero_weight_copy.household.household_weight = 1
    zero_weight_copy.household["household_is_capital_gains_clone"] = True
    data = stack_datasets(
        dataset,
        zero_weight_copy,
    )

    pred_cg, household_weight = impute_cg_to_doubled_dataset(data)

    data.person["capital_gains"] = pred_cg
    data.household["household_weight"] = household_weight

    data.validate()
    return data


# --- HMRC size-band donor stack ------------------------------------------
#
# The Advani & Summers spline stops at each income band's p95, so the
# imputation above cannot produce the large gains HMRC observes: in the
# built data no individual gain exceeds ~£2m, while HMRC attributes ~61% of
# all chargeable gains to gains of £1m or more (Capital Gains Tax
# statistics, Table 2.1a). Calibration cannot fix that on its own — there
# are no records carrying large gains for it to upweight.
#
# Following the SPI synthetic-household precedent in income.py, we stack a
# small set of zero-weight donor households, one group per HMRC size-of-gain
# band, each carrying that band's published mean gain. Calibration then
# decides how much weight each band's donors receive, pulled by the
# per-band HMRC targets in targets/sources/hmrc_cgt.py. At zero initial
# weight the donors change nothing unless the calibration needs them.
#
# Donor households are sampled with probability proportional to
# percent_with_gains at their first adult's total income (the same Advani &
# Summers table used above), so large gains land on the income profiles
# most likely to realise gains rather than on uniformly random households.

HMRC_SIZE_BANDS_FILE = "capital_gains_size_distribution_hmrc.csv"
DONORS_PER_BAND = 300
_DONOR_SEED = 1
# HMRC's rows below £12,300 mix AEA regimes and are definitionally
# incomplete (see the CSV header note); the spline body already covers
# gains of that size, so donors start at the £12,300 band.
_MIN_DONOR_BAND_LOWER = 12_300


def load_hmrc_size_bands() -> pd.DataFrame:
    """HMRC Table 2.1a size-of-gain bands used for donors and targets."""
    bands = pd.read_csv(STORAGE_FOLDER / HMRC_SIZE_BANDS_FILE, comment="#")
    bands = bands[bands.lower_limit >= _MIN_DONOR_BAND_LOWER].reset_index(drop=True)
    bands["upper_limit"] = bands.lower_limit.shift(-1).fillna(np.inf)
    bands["taxpayers"] = bands.taxpayers_thousands * 1e3
    bands["gains"] = bands.gains_gbp_millions * 1e6
    bands["mean_gain"] = bands.gains / bands.taxpayers
    return bands


def stack_cgt_band_donors(dataset: UKSingleYearDataset) -> UKSingleYearDataset:
    """Stack zero-weight donor households carrying HMRC band mean gains."""
    from policyengine_uk import Microsimulation

    dataset = dataset.copy()
    dataset.household["household_is_cgt_band_donor"] = False
    if "capital_gains" not in dataset.person.columns:
        dataset.person["capital_gains"] = 0.0

    bands = load_hmrc_size_bands()
    rng = np.random.default_rng(_DONOR_SEED)

    sim = Microsimulation(dataset=dataset)
    ti = sim.calculate("total_income").values
    first_adult = sim.calculate("adult_index").values == 1

    fa_household_id = dataset.person.person_household_id.values[first_adult]
    fa_income = ti[first_adult]
    income_band = (
        np.searchsorted(
            capital_gains.minimum_total_income.values, fa_income, side="right"
        )
        - 1
    )
    propensity = capital_gains.percent_with_gains.values[
        np.clip(income_band, 0, len(capital_gains) - 1)
    ]

    n_donors = DONORS_PER_BAND * len(bands)
    selected = rng.choice(
        fa_household_id,
        size=min(n_donors, len(fa_household_id)),
        replace=False,
        p=propensity / propensity.sum(),
    )
    band_of_household = dict(
        zip(selected, np.repeat(bands.mean_gain.values, DONORS_PER_BAND)[: len(selected)])
    )

    person_filter = dataset.person.person_household_id.isin(selected)
    donor_person = dataset.person[person_filter].reset_index(drop=True).copy()
    donor_benunit = (
        dataset.benunit[dataset.benunit.benunit_id.isin(donor_person.person_benunit_id)]
        .reset_index(drop=True)
        .copy()
    )
    donor_household = (
        dataset.household[dataset.household.household_id.isin(selected)]
        .reset_index(drop=True)
        .copy()
    )

    donor_household["household_weight"] = 0.0
    donor_household["household_is_cgt_band_donor"] = True

    # The band mean gain goes to each donor household's first adult;
    # everyone else in the household carries no gain, mirroring the
    # one-gainer-per-household convention of the imputation above.
    donor_first_adult = first_adult[person_filter.values]
    donor_person["capital_gains"] = 0.0
    donor_gains = donor_person.person_household_id.map(band_of_household).fillna(0.0).values
    donor_person.loc[donor_first_adult, "capital_gains"] = donor_gains[donor_first_adult]

    donor = UKSingleYearDataset(
        person=donor_person,
        benunit=donor_benunit,
        household=donor_household,
        fiscal_year=dataset.time_period,
    )

    data = stack_datasets(dataset, donor)
    data.validate()
    return data
