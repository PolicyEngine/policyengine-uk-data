"""Pension Credit take-up solved over entitled benefit units.

The FRS build first draws ``would_claim_pc`` against an unweighted count of
every benefit unit, mostly units with no entitlement. Entitled non-reporters
then claim at about the take-up rate on top of every reporter, so take-up
among the entitled ends up well above it. This step redraws the flag once
the imputations are in; entitlement depends on imputed capital, so it has
to wait for them.

For the calibration year it:
- finds the benefit units with positive Pension Credit entitlement;
- keeps reporters of Pension Credit as claimants;
- solves the probability that makes weighted take-up among entitled units
  equal DWP's caseload take-up rate;
- applies that probability to every non-reporter, entitled or not, so a
  unit that a reform makes entitled claims at the same rate.

Weights are the survey grossing weights, before calibration.
"""

import numpy as np

from policyengine_uk_data.parameters import load_take_up_rate
from policyengine_uk_data.utils.takeup import (
    assign_takeup_over_eligible,
    solve_fill_probability,
)

# Separate from the FRS build's take-up generator, so redrawing here leaves
# every other stochastic input unchanged.
PENSION_CREDIT_TAKEUP_SEED = 1_792


def assign_pension_credit_takeup(dataset, year: int):
    """Return a copy of ``dataset`` with ``would_claim_pc`` solved for
    ``year``, and a summary of weighted aggregates.

    The summary's Great Britain reporter baseline (reporters only, before
    any fill) is what DWP's GB targets are compared with: a fill can only
    add claimants, so if reporters alone exceed DWP's caseload the fill is
    zero and only calibration can bring the total down.
    """
    from policyengine_uk import Microsimulation

    sim = Microsimulation(dataset=dataset)
    entitlement = sim.calculate("pension_credit_entitlement", year).values
    entitled = entitlement > 0
    reported_amount = sim.calculate(
        "pension_credit_reported", year, map_to="benunit"
    ).values
    reported = reported_amount > 0
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
    rate = load_take_up_rate("pension_credit", year)
    probability = solve_fill_probability(rate, weights, entitled, reported)
    draws = np.random.default_rng(PENSION_CREDIT_TAKEUP_SEED).random(len(entitled))

    dataset = dataset.copy()
    claims = assign_takeup_over_eligible(draws, rate, weights, entitled, reported)
    dataset.benunit["would_claim_pc"] = claims

    gb = ~in_ni

    def total(values, mask, scale):
        return round(float((weights * values)[mask].sum()) / scale, 3)

    summary = {
        "year": year,
        "rate": rate,
        "fill_probability": round(probability, 4),
        "entitled_k": total(1, entitled, 1e3),
        "reporters_k": total(1, reported, 1e3),
        "entitled_reporters_k": total(1, entitled & reported, 1e3),
        "gb_reporters_k": total(1, gb & reported, 1e3),
        "gb_entitled_reporters_k": total(1, gb & entitled & reported, 1e3),
        "gb_reported_pension_credit_bn": total(reported_amount, gb, 1e9),
        "gb_reporters_modelled_pension_credit_bn": total(
            entitlement, gb & reported, 1e9
        ),
        "gb_claims_after_fill_k": total(1, gb & entitled & claims, 1e3),
        "gb_pension_credit_after_fill_bn": total(entitlement, gb & claims, 1e9),
    }
    return dataset, summary
