"""Tests for the HMRC size-of-gain band donors and their targets.

The spline-based capital gains imputation cannot produce gains above its
last income band's p95, so the build stacks zero-weight donor households
carrying each HMRC Table 2.1a band's mean gain, and calibration decides
their weight via per-band targets. These tests pin: the band table's
integrity, the per-band target computations (including the AEA gate), and
the donor stacking itself.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from policyengine_uk_data.datasets.imputations.capital_gains import (
    DONORS_PER_BAND,
    load_hmrc_size_bands,
    stack_cgt_band_donors,
)
from policyengine_uk_data.targets.sources.hmrc_cgt import (
    _CGT_BASE_YEAR,
    get_targets,
)


def _dummy_ctx(capital_gains, household_of_person, aea=3_000):
    """Minimal _SimContext stand-in (see test_hmrc_cgt_targets)."""
    n_households = max(household_of_person) + 1

    class _Params:
        def __init__(self, value):
            self.gov = SimpleNamespace(
                hmrc=SimpleNamespace(cgt=SimpleNamespace(annual_exempt_amount=value))
            )

    ctx = SimpleNamespace()
    ctx.sim = SimpleNamespace(
        tax_benefit_system=SimpleNamespace(parameters=lambda instant: _Params(aea))
    )
    ctx.pe_person = lambda variable: np.array(capital_gains, dtype=float)

    def household_from_person(values):
        result = np.zeros(n_households)
        for person_index, household_index in enumerate(household_of_person):
            result[household_index] += values[person_index]
        return result

    ctx.household_from_person = household_from_person
    return ctx


def test_band_table_shape():
    bands = load_hmrc_size_bands()
    assert bands.lower_limit.min() == 12_300
    assert np.isinf(bands.upper_limit.iloc[-1])
    assert (bands.lower_limit.values[1:] == bands.upper_limit.values[:-1]).all()
    # Band means must sit inside their band (the top band's mean is
    # unbounded above by construction).
    assert (bands.mean_gain.values >= bands.lower_limit.values).all()
    assert (bands.mean_gain.values[:-1] < bands.upper_limit.values[:-1]).all()
    assert bands.mean_gain.is_monotonic_increasing


def test_band_targets_match_published_table():
    bands = load_hmrc_size_bands()
    targets = {t.name: t for t in get_targets()}
    for row in bands.itertuples():
        label = f"{int(row.lower_limit)}"
        count = targets[f"hmrc/cgt_taxpayers_band_{label}"]
        gains = targets[f"hmrc/capital_gains_band_{label}"]
        assert count.values[_CGT_BASE_YEAR] == pytest.approx(
            row.taxpayers_thousands * 1e3
        )
        assert gains.values[_CGT_BASE_YEAR] == pytest.approx(
            row.gains_gbp_millions * 1e6
        )
        assert count.is_count and not gains.is_count


def test_band_compute_gates_on_band_and_aea():
    # Persons: below AEA, in first band, at a band edge, in top band.
    gains = [2_000, 15_000, 25_000, 10_000_000]
    ctx = _dummy_ctx(gains, household_of_person=[0, 0, 1, 2])
    targets = {t.name: t for t in get_targets()}

    first_count = targets["hmrc/cgt_taxpayers_band_12300"].custom_compute(
        ctx, targets["hmrc/cgt_taxpayers_band_12300"], _CGT_BASE_YEAR
    )
    # Only the £15k gain is in [12,300, 25,000); the £25k gain has moved to
    # the next band and the £2k gain is under the AEA.
    assert first_count.tolist() == [1.0, 0.0, 0.0]

    top_gains = targets["hmrc/capital_gains_band_5000000"].custom_compute(
        ctx, targets["hmrc/capital_gains_band_5000000"], _CGT_BASE_YEAR
    )
    assert top_gains.tolist() == [0.0, 0.0, 10_000_000.0]


def test_stack_cgt_band_donors(frs):
    bands = load_hmrc_size_bands()
    n_households = len(frs.household)
    out = stack_cgt_band_donors(frs)

    donors = out.household[out.household.household_is_cgt_band_donor]
    assert len(donors) == DONORS_PER_BAND * len(bands)
    assert len(out.household) == n_households + len(donors)
    assert (donors.household_weight == 0).all()
    # Non-donor households keep their weights.
    originals = out.household[~out.household.household_is_cgt_band_donor]
    assert originals.household_weight.sum() == pytest.approx(
        frs.household.household_weight.sum()
    )

    donor_people = out.person[out.person.person_household_id.isin(donors.household_id)]
    gainers = donor_people[donor_people.capital_gains > 0]
    # One gainer per donor household, at exactly a band mean.
    assert gainers.person_household_id.is_unique
    assert len(gainers) == len(donors)
    assert set(np.round(gainers.capital_gains, 6)) == set(
        np.round(bands.mean_gain, 6)
    )
    # Every band is populated with its full donor allocation.
    counts = gainers.capital_gains.round(6).value_counts()
    assert (counts == DONORS_PER_BAND).all()
