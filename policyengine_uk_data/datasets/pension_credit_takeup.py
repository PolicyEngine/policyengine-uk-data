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
from policyengine_uk_data.utils.takeup import assign_takeup_over_eligible

# Separate from the FRS build's take-up generator, so redrawing here leaves
# every other stochastic input unchanged.
PENSION_CREDIT_TAKEUP_SEED = 1_792


def assign_pension_credit_takeup(dataset, year: int):
    """Return a copy of ``dataset`` with ``would_claim_pc`` solved for ``year``."""
    from policyengine_uk import Microsimulation

    sim = Microsimulation(dataset=dataset)
    entitled = sim.calculate("pension_credit_entitlement", year).values > 0
    reported = (
        sim.calculate("pension_credit_reported", year, map_to="benunit").values > 0
    )
    weights = sim.calculate("benunit_weight", year).values
    draws = np.random.default_rng(PENSION_CREDIT_TAKEUP_SEED).random(len(entitled))

    dataset = dataset.copy()
    dataset.benunit["would_claim_pc"] = assign_takeup_over_eligible(
        draws,
        load_take_up_rate("pension_credit", year),
        weights,
        entitled,
        reported,
    )
    return dataset
