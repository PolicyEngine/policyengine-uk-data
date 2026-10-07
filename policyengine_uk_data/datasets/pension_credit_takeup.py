"""Pension Credit take-up solved over entitled benefit units.

The FRS build first draws ``would_claim_pc`` against an unweighted count of
every benefit unit, mostly units with no entitlement. Entitled non-reporters
then claim at about the take-up rate on top of every reporter, so take-up
among the entitled ends up well above it. This step redraws the flag once
the imputations are in; entitlement depends on imputed capital, so it has
to wait for them.

For the calibration year it:
- finds the benefit units with positive Pension Credit entitlement;
- keeps reporters of Pension Credit as claimants, except on SPI-synthetic
  households, whose reports the second-stage imputation (frs_only.py)
  predicts from their SPI incomes rather than observes;
- solves the probability that makes weighted take-up among entitled units
  in Great Britain equal DWP's caseload take-up rate, which covers Great
  Britain only;
- applies that probability to every entitled non-reporter, in Northern
  Ireland too;
- draws every unit with no entitlement in the calibration year at DWP's
  take-up for Savings Credit-only awards (37% in FYE 2024), the rate at
  which a unit claims once a reform or a later year entitles it. This step
  finds no entitlement for them in the calibration year, so their rate only
  matters under a reform, in a later year, or where the saved dataset's
  calibration-year entitlement differs from this step's (#479).

Weights are the survey grossing weights, before calibration.
"""

import numpy as np

from policyengine_uk_data.parameters import load_take_up_rate
from policyengine_uk_data.utils.takeup import solve_fill_probability

# Separate from the FRS build's take-up generator, so redrawing here leaves
# every other stochastic input unchanged.
PENSION_CREDIT_TAKEUP_SEED = 1_792


def spi_synthetic_benunits(dataset) -> np.ndarray:
    """Benefit units in households flagged ``household_is_spi_synthetic``."""
    household = dataset.household
    if "household_is_spi_synthetic" not in household.columns:
        return np.zeros(len(dataset.benunit), dtype=bool)
    synthetic = household.loc[
        household["household_is_spi_synthetic"].astype(bool), "household_id"
    ]
    person = dataset.person
    benunits = person.loc[
        person["person_household_id"].isin(synthetic), "person_benunit_id"
    ]
    return dataset.benunit["benunit_id"].isin(benunits).to_numpy()


def pension_credit_takeup_flags(
    draws, rate, weights, entitled, reported, great_britain, newly_entitled_rate
):
    """Claim flags and the fill probability solved over GB entitled units.

    Only Great Britain enters the solve, because DWP's take-up rate covers
    Great Britain. Reporters claim. Every other entitled unit, wherever it
    lives, claims when its draw is below the probability; every unit that
    isn't entitled claims when its draw is below ``newly_entitled_rate``.
    """
    entitled = np.asarray(entitled, dtype=bool)
    reported = np.asarray(reported, dtype=bool)
    eligible = entitled & np.asarray(great_britain, dtype=bool)
    probability = solve_fill_probability(rate, weights, eligible, reported)
    claim_probability = np.where(entitled, probability, newly_entitled_rate)
    claims = reported | (np.asarray(draws, dtype=np.float64) < claim_probability)
    return claims, probability


def assign_pension_credit_takeup(dataset, year: int):
    """Return a copy of ``dataset`` with ``would_claim_pc`` solved for
    ``year``, and a summary of weighted aggregates.

    The probability is zero when entitled GB reporters already make up the
    take-up rate of entitled GB units. The summary's GB reporter baseline
    (reporters only, before any fill) is for comparison with DWP's absolute
    caseload and spending, which calibration targets later and separately.
    """
    from policyengine_uk import Microsimulation

    sim = Microsimulation(dataset=dataset)
    entitlement = sim.calculate("pension_credit_entitlement", year).values
    entitled = entitlement > 0
    reported_amount = sim.calculate(
        "pension_credit_reported", year, map_to="benunit"
    ).values
    spi_synthetic = spi_synthetic_benunits(dataset)
    reported = (reported_amount > 0) & ~spi_synthetic
    weights = sim.calculate("benunit_weight", year).values
    country = sim.calculate("country", year).values
    in_ni = (
        sim.map_result(
            sim.map_result(
                (country == "NORTHERN_IRELAND").astype(float), "household", "person"
            ),
            "person",
            "benunit",
        )
        > 0
    )
    gb = ~in_ni
    rate = load_take_up_rate("pension_credit", year)
    newly_entitled_rate = load_take_up_rate("pension_credit_newly_entitled", year)
    draws = np.random.default_rng(PENSION_CREDIT_TAKEUP_SEED).random(len(entitled))
    claims, probability = pension_credit_takeup_flags(
        draws, rate, weights, entitled, reported, gb, newly_entitled_rate
    )

    dataset = dataset.copy()
    dataset.benunit["would_claim_pc"] = claims

    def total(values, mask, scale):
        return round(float((weights * values)[mask].sum()) / scale, 3)

    summary = {
        "year": year,
        "rate": rate,
        "fill_probability": round(probability, 4),
        "newly_entitled_rate": newly_entitled_rate,
        "entitled_k": total(1, entitled, 1e3),
        "gb_entitled_k": total(1, gb & entitled, 1e3),
        "reporters_k": total(1, reported, 1e3),
        "spi_synthetic_units_not_anchored": int(
            (spi_synthetic & (reported_amount > 0)).sum()
        ),
        "entitled_reporters_k": total(1, entitled & reported, 1e3),
        "gb_reporters_k": total(1, gb & reported, 1e3),
        "gb_entitled_reporters_k": total(1, gb & entitled & reported, 1e3),
        "gb_reported_pension_credit_bn": total(reported_amount, gb, 1e9),
        "gb_reporters_modelled_pension_credit_bn": total(
            entitlement, gb & reported, 1e9
        ),
        "gb_claims_after_fill_k": total(1, gb & entitled & claims, 1e3),
        "gb_pension_credit_after_fill_bn": total(entitlement, gb & claims, 1e9),
        "gb_not_entitled_k": total(1, gb & ~entitled, 1e3),
        "gb_not_entitled_flagged_share": round(
            float(
                weights[gb & ~entitled & claims].sum()
                / max(weights[gb & ~entitled].sum(), 1)
            ),
            4,
        ),
    }
    return dataset, summary
