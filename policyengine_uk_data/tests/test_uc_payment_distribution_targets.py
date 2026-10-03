"""Tests for the DWP UC payment distribution targets.

The Stat-Xplore extract (storage/uc_national_payment_dist.xlsx, May 2025)
counts households on UC by monthly award band and family type. Its top band,
'£2500.01 or over', is open-ended; it used to parse to NaN bounds, so its four
targets (1.4k to 83k households) had a column of zeros and could never be met.
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from policyengine_uk_data.storage import STORAGE_FOLDER
from policyengine_uk_data.targets.compute.benefits import compute_uc_payment_dist
from policyengine_uk_data.targets.sources.dwp import _uc_payment_distribution_targets
from policyengine_uk_data.utils.uc_data import (
    _check_bands_disjoint,
    parse_monthly_award_band,
    uc_national_payment_dist,
)

FAMILY_TYPES = {
    "Single, no children": "SINGLE",
    "Single, with children": "LONE_PARENT",
    "Couple, no children": "COUPLE_NO_CHILDREN",
    "Couple, with children": "COUPLE_WITH_CHILDREN",
}


def _raw_extract() -> pd.DataFrame:
    """Household counts indexed by award band, one column per family type."""
    raw = pd.read_excel(STORAGE_FOLDER / "uc_national_payment_dist.xlsx", header=None)
    counts = raw.iloc[9:, 3:7]
    counts.index = raw.iloc[9:, 1]
    counts.columns = raw.iloc[7, 3:7]
    return counts


def test_parse_monthly_award_band():
    assert parse_monthly_award_band("£0.01 to £100.00") == (0, 1_200)
    assert parse_monthly_award_band("£100.01 to £200.00") == (1_200, 2_400)
    assert parse_monthly_award_band("£1000.01 to £1100.00") == (12_000, 13_200)
    assert parse_monthly_award_band("£2,500.01 or over") == (30_000, np.inf)
    with pytest.raises(ValueError):
        parse_monthly_award_band("No payment")


def test_top_band_targets_are_reachable():
    targets = {t.name: t for t in _uc_payment_distribution_targets()}
    top = _raw_extract().loc["£2500.01 or over"]
    for label, family_type in FAMILY_TYPES.items():
        target = targets[
            f"dwp/uc_payment_dist/{family_type}_annual_payment_30_000_to_inf"
        ]
        assert target.lower_bound == 30_000
        assert target.upper_bound == np.inf
        assert target.values[2025] == top[label]
    for target in targets.values():
        assert not np.isnan(target.lower_bound)
        assert not np.isnan(target.upper_bound)
        assert "nan" not in target.name


def test_band_counts_sum_to_households_with_a_payment():
    """Conservation: the bands hold every household with a payment, once.

    Stat-Xplore perturbs each cell for disclosure control, so a total can
    differ from the sum of its cells by a few households.
    """
    raw = _raw_extract()
    for label, family_type in FAMILY_TYPES.items():
        with_payment = raw.loc["Total", label] - raw.loc["No payment", label]
        parsed = uc_national_payment_dist[
            uc_national_payment_dist.family_type == family_type
        ]
        assert abs(parsed.household_count.sum() - with_payment) <= 10, label


def test_bands_tile_the_positive_awards():
    for family_type, bands in uc_national_payment_dist.groupby("family_type"):
        bands = bands.sort_values("uc_annual_payment_min")
        lower = bands.uc_annual_payment_min.to_numpy()
        upper = bands.uc_annual_payment_max.to_numpy()
        assert lower[0] == 0, family_type
        assert np.array_equal(lower[1:], upper[:-1]), family_type
        assert upper[-1] == np.inf, family_type


def test_overlapping_summary_band_is_rejected():
    bands = pd.DataFrame(
        {
            "family_type": ["SINGLE"] * 3,
            "uc_annual_payment_min": [18_000.0, 19_200.0, 18_000.0],
            "uc_annual_payment_max": [19_200.0, 20_400.0, np.inf],
        }
    )
    with pytest.raises(ValueError):
        _check_bands_disjoint(bands)


def _fake_ctx(uc, family_type, household):
    def calculate(variable, map_to=None):
        values = {"universal_credit": uc, "family_type": family_type}[variable]
        return SimpleNamespace(values=values)

    n_households = household.max() + 1
    return SimpleNamespace(
        sim=SimpleNamespace(calculate=calculate),
        household_from_family=lambda values: np.bincount(
            household, weights=np.asarray(values, float), minlength=n_households
        ),
    )


_TARGETS = _uc_payment_distribution_targets()
_EDGES = sorted({t.upper_bound for t in _TARGETS if np.isfinite(t.upper_bound)})
_AWARDS = st.one_of(
    st.just(0.0),  # no payment
    st.floats(0, 1e6, allow_nan=False),  # anywhere, including the open top band
    st.sampled_from(_EDGES),  # exactly on a band edge
    st.sampled_from(_EDGES).map(lambda edge: np.nextafter(edge, np.inf)),
    st.sampled_from(_EDGES).map(lambda edge: np.nextafter(edge, -np.inf)),
)


@settings(max_examples=200, deadline=None)
@given(
    st.lists(
        st.tuples(
            _AWARDS, st.sampled_from(list(FAMILY_TYPES.values())), st.integers(0, 9)
        ),
        min_size=1,
        max_size=60,
    )
)
def test_every_award_lands_in_exactly_one_band(benefit_units):
    """Property: summed over bands, each household's column counts its benefit
    units with a positive award, and never one without."""
    uc, family_type, household = (np.array(column) for column in zip(*benefit_units))
    ctx = _fake_ctx(uc.astype(float), family_type, household)

    total = sum(compute_uc_payment_dist(t, ctx) for t in _TARGETS)
    expected = np.bincount(
        household, weights=(uc > 0).astype(float), minlength=len(total)
    )
    np.testing.assert_array_equal(total, expected)
