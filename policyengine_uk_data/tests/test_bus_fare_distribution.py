"""Post-build validation of the bus fare spending distribution.

These tests run against the *built* Enhanced FRS (via the ``baseline``
fixture, which skips when the dataset has not been rebuilt), so they
validate the dataset after a rebuild rather than the calibration code in
isolation. They lock in the anchors applied by
``calibrate_bus_fare_spending``:

1. National total: DfT BUS05aii passenger fare receipts, England → UK.
2. Within-England regional split: DfT BUS05ai London / outside-London
   receipts (London GBP 1,347m of England's GBP 3,417m). Without this,
   the LCFS imputation under-captures London and overstates
   outside-London fare exposure by ~47%.
3. Income gradient: England quintile fare shares anchored to NTS0705a
   local-bus trip rates (Q1 47.6 ... Q5 13.0 trips/person/year) — the
   lowest quintile makes ~3.7x the trips of the highest, so fare
   spending must decline with income, not rise with it as the raw LCFS
   spend gradient does.
"""

from __future__ import annotations

import numpy as np

from policyengine_uk_data.datasets.frs_release import CURRENT_FRS_RELEASE
from policyengine_uk_data.datasets.imputations.consumption import (
    BUS_FARE_LONDON_SHARE_OF_ENGLAND,
    BUS_FARE_TARGETS,
    ENGLAND_REGIONS_OUTSIDE_LONDON,
    NTS_BUS_TRIPS_BY_INCOME_QUINTILE,
)

PERIOD = CURRENT_FRS_RELEASE.calibration_year
TOTAL_TOLERANCE = 0.05
SHARE_TOLERANCE = 0.03  # absolute share points on the London/England split


def _household_arrays(baseline):
    fares = baseline.calculate("bus_fare_spending", PERIOD, map_to="household")
    values = np.array(fares.values) * np.array(fares.weights)
    region = np.array(
        baseline.calculate("region", PERIOD, map_to="household").values
    ).astype(str)
    decile = np.array(
        baseline.calculate(
            "household_income_decile", PERIOD, map_to="household"
        ).values,
        dtype=float,
    )
    return values, region, decile


def test_bus_fare_total_matches_dft_target(baseline):
    """Weighted bus fare spending matches the DfT BUS05aii UK target."""
    target = BUS_FARE_TARGETS[PERIOD]
    total = baseline.calculate("bus_fare_spending", PERIOD).sum()
    assert abs(total / target - 1) < TOTAL_TOLERANCE, (
        f"bus_fare_spending total {total / 1e9:.2f}bn is >{TOTAL_TOLERANCE:.0%} "
        f"from the DfT target {target / 1e9:.2f}bn."
    )


def test_london_fare_share_matches_bus05ai(baseline):
    """London's share of England fares matches DfT BUS05ai receipts."""
    values, region, _ = _household_arrays(baseline)
    london = values[region == "LONDON"].sum()
    outside = values[np.isin(region, list(ENGLAND_REGIONS_OUTSIDE_LONDON))].sum()
    share = london / (london + outside)
    assert abs(share - BUS_FARE_LONDON_SHARE_OF_ENGLAND) < SHARE_TOLERANCE, (
        f"London fare share {share:.3f} is >{SHARE_TOLERANCE} from the "
        f"BUS05ai share {BUS_FARE_LONDON_SHARE_OF_ENGLAND:.3f}."
    )


def test_fare_spending_declines_with_income(baseline):
    """England quintile fare totals follow the NTS trip gradient direction."""
    values, region, decile = _household_arrays(baseline)
    in_england = np.isin(region, list(ENGLAND_REGIONS_OUTSIDE_LONDON) + ["LONDON"])
    quintile = np.where(
        decile >= 1, np.clip(((decile - 1) // 2 + 1).astype(int), 1, 5), 0
    )
    totals = {
        q: values[in_england & (quintile == q)].sum()
        for q in NTS_BUS_TRIPS_BY_INCOME_QUINTILE
    }
    assert totals[1] > totals[5], (
        f"Q1 fare spending ({totals[1] / 1e9:.2f}bn) should exceed Q5 "
        f"({totals[5] / 1e9:.2f}bn) per the NTS0705a trip gradient."
    )
    # The full anchored gradient: per-person trip rates decline monotonically,
    # so quintile totals must not increase from Q3 upwards.
    assert totals[3] >= totals[4] >= totals[5], (
        "Upper-quintile fare spending should decline: "
        f"Q3 {totals[3] / 1e9:.2f}bn, Q4 {totals[4] / 1e9:.2f}bn, "
        f"Q5 {totals[5] / 1e9:.2f}bn."
    )
