"""Pension Credit take-up solved over entitled units, and DWP targets."""

import numpy as np
import openpyxl
from hypothesis import given, settings
from hypothesis import strategies as st

from policyengine_uk_data.parameters import load_take_up_rate
from policyengine_uk_data.targets import get_all_targets
from policyengine_uk_data.targets.schema import GREAT_BRITAIN, Unit
from policyengine_uk_data.targets.sources import dwp_pension_credit, obr
from policyengine_uk_data.utils.takeup import (
    assign_takeup_over_eligible,
    solve_fill_probability,
)

_units = st.lists(
    st.tuples(st.floats(0, 5_000), st.booleans(), st.booleans()),
    min_size=1,
    max_size=60,
)


@settings(max_examples=300, deadline=None)
@given(_units, st.floats(0, 1))
def test_fill_probability_meets_the_rate_or_hits_a_bound(units, rate):
    weights = np.array([u[0] for u in units])
    eligible = np.array([u[1] for u in units])
    reported = np.array([u[2] for u in units])
    p = solve_fill_probability(rate, weights, eligible, reported)
    assert 0 <= p <= 1
    target = rate * weights[eligible].sum()
    reporting = weights[eligible & reported].sum()
    remaining = weights[eligible & ~reported].sum()
    expected = reporting + p * remaining
    if remaining <= 0:
        assert p == 0
    elif 0 < p < 1:
        assert np.isclose(expected, target, rtol=1e-9, atol=1e-6)
    elif p == 0:
        assert reporting >= target - 1e-6
    else:
        assert reporting + remaining <= target + 1e-6


@settings(max_examples=200, deadline=None)
@given(_units, st.floats(0, 1), st.integers(0, 2**32 - 1))
def test_reporters_claim_and_every_non_reporter_shares_one_probability(
    units, rate, seed
):
    weights = np.array([u[0] for u in units])
    eligible = np.array([u[1] for u in units])
    reported = np.array([u[2] for u in units])
    draws = np.random.default_rng(seed).random(len(units))
    result = assign_takeup_over_eligible(draws, rate, weights, eligible, reported)
    p = solve_fill_probability(rate, weights, eligible, reported)
    assert result[reported].all()
    # Eligibility does not change a non-reporter's flag, so a unit a reform
    # makes entitled claims at the same rate.
    np.testing.assert_array_equal(result[~reported], (draws < p)[~reported])


def test_weighted_take_up_among_eligible_matches_rate():
    rng = np.random.default_rng(0)
    n = 200_000
    weights = rng.uniform(100, 3_000, n)
    eligible = rng.random(n) < 0.2
    reported = eligible & (rng.random(n) < 0.4)
    result = assign_takeup_over_eligible(
        rng.random(n), 0.62, weights, eligible, reported
    )
    take_up = weights[eligible & result].sum() / weights[eligible].sum()
    assert abs(take_up - 0.62) < 0.005
    # Non-entitled units are drawn at the same probability as entitled ones.
    p = solve_fill_probability(0.62, weights, eligible, reported)
    assert abs(result[~eligible].mean() - p) < 0.01


def test_rate_is_dwp_fye_2024_caseload_take_up():
    assert load_take_up_rate("pension_credit", 2025) == 0.62


def test_targets_reconcile_with_dwp_components():
    """Guarantee + savings credit spending and the three claim types sum to
    the transcribed totals (DWP Pension Credit sheet, separate rows)."""
    guarantee = {2022: 4_594.32, 2025: 5_787.44, 2030: 5_543.38}
    savings = {2022: 340.99, 2025: 357.03, 2030: 278.38}
    claims = {2022: (727, 448, 199), 2025: (807, 399, 176), 2030: (734, 285, 114)}
    targets = {t.name: t for t in dwp_pension_credit.get_targets()}
    spending = targets["dwp/pension_credit"]
    caseload = targets["dwp/pension_credit_claims"]
    for year in guarantee:
        assert abs(spending.values[year] / 1e6 - guarantee[year] - savings[year]) < 0.02
        assert abs(caseload.values[year] / 1e3 - sum(claims[year])) <= 1
    for target in targets.values():
        assert target.countries == GREAT_BRITAIN
        assert target.unit == (Unit.COUNT if target.is_count else Unit.GBP)


def test_pension_credit_is_targeted_only_by_dwp():
    names = {t.name for t in get_all_targets() if t.variable == "pension_credit"}
    assert names == {"dwp/pension_credit", "dwp/pension_credit_claims"}


def test_obr_pension_credit_row_is_not_parsed():
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "4.9"
    ws["B2"] = "Pension credit"
    for col in "CDEFGHI":
        ws[f"{col}2"] = 6.0
    assert not obr._parse_welfare(wb)


def test_built_dataset_reporters_claim_and_caseload_is_near_dwp(baseline):
    from policyengine_uk_data.targets.build_loss_matrix import (
        _SimContext,
        restrict_to_countries,
    )

    year = 2025
    reported = (
        baseline.calculate("pension_credit_reported", year, map_to="benunit").values > 0
    )
    would_claim = baseline.calculate("would_claim_pc", year).values.astype(bool)
    assert would_claim[reported].all()

    baseline.default_calculation_period = str(year)
    ctx = _SimContext(baseline, str(year), None, None)
    weight = baseline.calculate("household_weight", year).values
    for target in dwp_pension_credit.get_targets():
        column = restrict_to_countries(
            target.custom_compute(ctx, target, year), ctx.country, target.countries
        )
        ratio = (column * weight).sum() / target.values[year]
        assert 0.5 < ratio < 2, (target.name, ratio)
