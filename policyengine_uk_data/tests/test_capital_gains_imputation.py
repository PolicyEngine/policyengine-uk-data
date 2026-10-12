"""Tests for the has-gains twins of the capital gains imputation.

The imputation stacks a second copy of each household whose first adult
draws a gain HMRC's statistics describe, and splits the household's mass
between the two records. These tests state what must hold for every input:
mass is conserved, no record carries a loss or a gain under the lowest HMRC
band, zero-weight rows are split like survey rows, and the calibration
starts with every HMRC band's taxpayer count on its target.
"""

import importlib
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from policyengine_uk.data import UKSingleYearDataset

from policyengine_uk_data.targets.sources import hmrc_cgt
from policyengine_uk_data.utils.calibrate import (
    initialize_weight_priors,
    rake_priors_to_targets,
)

# The package re-exports the Advani & Summers table under the module's own
# name, so the module has to be imported by its full path.
cg = importlib.import_module("policyengine_uk_data.datasets.imputations.capital_gains")

TABLE = cg.capital_gains
LOWEST_BAND = 3_000
TOP_INCOME_BAND = len(TABLE) - 1


def _dataset(weights, sizes, income_bands) -> UKSingleYearDataset:
    """Households of one benefit unit whose members share a total income;
    the first member is the first adult."""
    weights = np.asarray(weights, dtype=float)
    household_id = np.arange(1, len(weights) + 1)
    person_household_id = np.repeat(household_id, sizes)
    income_bands = np.broadcast_to(np.asarray(income_bands), weights.shape)
    incomes = TABLE.minimum_total_income.values[income_bands] + 1.0
    person = pd.DataFrame(
        {
            "person_id": np.arange(1, len(person_household_id) + 1),
            "person_household_id": person_household_id,
            "person_benunit_id": person_household_id,
            "total_income": np.repeat(incomes, sizes),
            "first_adult": np.r_[True, np.diff(person_household_id) != 0],
        }
    )
    return UKSingleYearDataset(
        person=person,
        benunit=pd.DataFrame({"benunit_id": household_id}),
        household=pd.DataFrame(
            {"household_id": household_id, "household_weight": weights}
        ),
        fiscal_year=2024,
    )


def _incomes_from_columns(dataset):
    """Stands in for the microsimulation that reads incomes in a build."""
    return (
        dataset.person.total_income.to_numpy(dtype=float),
        dataset.person.first_adult.to_numpy(dtype=bool),
    )


def _impute(dataset, donors: bool = False):
    with mock.patch.object(cg, "_first_adult_incomes", _incomes_from_columns):
        out = cg.impute_capital_gains(dataset)
        return cg.stack_cgt_band_donors(out) if donors else out


households = st.lists(
    st.tuples(
        st.one_of(st.just(0.0), st.floats(0.1, 5_000.0)),
        st.integers(1, 5),
        st.integers(0, TOP_INCOME_BAND),
    ),
    min_size=1,
    max_size=40,
)


@given(households)
@settings(max_examples=200, deadline=None)
def test_shares_put_the_table_share_of_each_band_in_gains(rows):
    weights, sizes, bands = (np.array(column) for column in zip(*rows))
    dataset = _dataset(weights, sizes, bands)
    income_band = cg._income_band(dataset.person.total_income.values)
    first_adult = dataset.person.first_adult.values
    person_weight = np.repeat(weights, sizes)

    share = cg.has_gains_shares(income_band, first_adult, person_weight)

    people = np.bincount(income_band, weights=person_weight, minlength=len(TABLE))
    first_adults = np.bincount(
        income_band[first_adult],
        weights=person_weight[first_adult],
        minlength=len(TABLE),
    )
    populated = first_adults > 0
    assert ((share >= 0) & (share < 1)).all()
    assert (share[~populated] == 0).all()
    np.testing.assert_allclose(
        (share * first_adults)[populated],
        (TABLE.percent_with_gains.values * people)[populated],
    )


def test_a_band_with_too_few_first_adults_fails_loudly():
    """One light first adult cannot hold the gains of a band of heavy
    non-first adults; the build must say so, not cap the share silently."""
    income_band = np.full(11, TOP_INCOME_BAND)
    first_adult = np.r_[True, np.zeros(10, dtype=bool)]
    person_weight = np.r_[1.0, np.full(10, 100.0)]
    with pytest.raises(ValueError, match="too few first adults"):
        cg.has_gains_shares(income_band, first_adult, person_weight)


@given(households)
@settings(max_examples=100, deadline=None)
def test_twins_conserve_mass_and_carry_only_gains_hmrc_describes(rows):
    weights, sizes, bands = (np.array(column) for column in zip(*rows))
    n = len(weights)
    out = _impute(_dataset(weights, sizes, bands))
    household, person = out.household, out.person

    original = household.iloc[:n]
    twin = household.iloc[n:]
    # Originals keep their order; a twin's id is its original's, offset.
    assert original.household_id.tolist() == list(range(1, n + 1))
    assert not original.household_is_capital_gains_clone.any()
    assert twin.household_is_capital_gains_clone.all()
    twin_of = (twin.household_id - (n + 1)).to_numpy()
    assert set(twin_of) <= set(original.household_id)

    # Mass: a survey row and its twin share the row's weight; a zero-weight
    # row stays at zero; no twin is itself a zero-weight row.
    twin_weight = np.zeros(n)
    twin_weight[twin_of - 1] = twin.household_weight
    assert (twin.household_weight > 0).all()
    survey = weights > 0
    np.testing.assert_allclose(
        (original.household_weight.to_numpy() + twin_weight)[survey], weights[survey]
    )
    assert (original.household_weight[~survey] == 0).all()
    assert (original.household_weight[survey] > 0).all()

    # Gains: none on originals; on a twin, the first adult's alone, above
    # the lowest HMRC band limit.
    gains = person.capital_gains.to_numpy()
    in_twin = person.person_household_id.isin(twin.household_id).to_numpy()
    assert (gains[~in_twin] == 0).all()
    assert ((gains == 0) | (gains > LOWEST_BAND)).all()
    gainers = person[gains > 0]
    assert gainers.first_adult.all()
    assert sorted(gainers.person_household_id) == sorted(twin.household_id)

    # A twin exists exactly where the first adult's draw is above the limit
    # and their income band has survey first adults to set its share.
    source = _dataset(weights, sizes, bands).person
    person_band = cg._income_band(source.total_income.values)
    first_adult = source.first_adult.values
    share = cg.has_gains_shares(person_band, first_adult, np.repeat(weights, sizes))
    draws = cg.draw_gains(person_band, first_adult)[first_adult]
    expected = (draws > LOWEST_BAND) & (share[bands] > 0)
    assert sorted(twin_of) == sorted(np.flatnonzero(expected) + 1)


def test_imputation_is_deterministic():
    rng = np.random.default_rng(3)
    dataset = _dataset(
        rng.uniform(1, 2_000, 300),
        rng.integers(1, 5, 300),
        rng.integers(0, TOP_INCOME_BAND + 1, 300),
    )
    first, second = _impute(dataset, donors=True), _impute(dataset, donors=True)
    pd.testing.assert_frame_equal(first.household, second.household)
    pd.testing.assert_frame_equal(first.person, second.person)


def test_zero_weight_rows_split_their_prior_like_survey_rows():
    """The defect behind uk-data#570: a zero-weight row's twin kept a zero
    weight, so the calibration gave it a full zero-weight prior, the same as
    its no-gains original. Half of the zero-weight rows' prior mass then
    started with gains, whatever the table's share."""
    n_survey, n_zero = 400, 200
    weights = np.r_[np.full(n_survey, 1_000.0), np.zeros(n_zero)]
    out = _impute(_dataset(weights, np.ones(len(weights), dtype=int), TOP_INCOME_BAND))
    household = out.household
    n = n_survey + n_zero
    twin = household.iloc[n:]
    twin_of = (twin.household_id - (n + 1)).to_numpy() - 1
    of_zero = twin_of >= n_survey

    # One person per household, so the has-gains share is the table's.
    share = TABLE.percent_with_gains.iloc[TOP_INCOME_BAND]
    priors = initialize_weight_priors(household.household_weight.to_numpy())
    survey_twins = priors[n:][~of_zero]
    np.testing.assert_allclose(
        survey_twins / (survey_twins + priors[twin_of[~of_zero]]), share
    )
    # The same split for zero-weight rows, up to the mass their twins add.
    added = twin.household_weight[of_zero].sum() / weights.sum()
    np.testing.assert_allclose(
        priors[n:][of_zero] / priors[twin_of[of_zero]], share / (1 + added)
    )
    # The only zero-weight rows left are the zero-weight originals.
    assert (household.household_weight == 0).sum() == n_zero


def _band_matrix(dataset, year: int) -> tuple[pd.DataFrame, pd.Series]:
    """The HMRC columns of the calibration's national matrix, built by the
    target functions themselves."""
    gains = dataset.person.capital_gains.to_numpy(dtype=float)
    household = pd.Index(dataset.household.household_id).get_indexer(
        dataset.person.person_household_id
    )
    parameters = SimpleNamespace(
        gov=SimpleNamespace(
            hmrc=SimpleNamespace(cgt=SimpleNamespace(annual_exempt_amount=3_000))
        )
    )
    ctx = SimpleNamespace(
        pe_person=lambda variable: gains,
        household_from_person=lambda values: np.bincount(
            household, weights=values, minlength=len(dataset.household)
        ),
        sim=SimpleNamespace(
            tax_benefit_system=SimpleNamespace(parameters=lambda instant: parameters)
        ),
    )
    targets = hmrc_cgt.get_targets()
    return (
        pd.DataFrame({t.name: t.custom_compute(ctx, t, year) for t in targets}),
        pd.Series({t.name: t.values[year] for t in targets}),
    )


def _stacked_population():
    """Survey and zero-weight households over every income band, with enough
    top-income households for the twins to reach every band below £2m."""
    rng = np.random.default_rng(7)
    n = 6_000
    bands = np.where(
        rng.random(n) < 0.5, TOP_INCOME_BAND, rng.integers(0, TOP_INCOME_BAND, n)
    )
    weights = np.where(rng.random(n) < 0.4, 0.0, rng.uniform(100, 3_000, n))
    return _impute(_dataset(weights, rng.integers(1, 4, n), bands), donors=True)


def test_donors_fill_the_top_bands_and_are_never_twins():
    out = _stacked_population()
    bands = cg.load_donor_bands()
    donors = out.household[out.household.household_is_cgt_band_donor]
    assert len(donors) == cg.DONORS_PER_BAND * len(bands)
    assert not donors.household_is_capital_gains_clone.any()

    donor_people = out.person[out.person.person_household_id.isin(donors.household_id)]
    gainers = donor_people[donor_people.capital_gains > 0]
    assert gainers.first_adult.all()
    assert gainers.person_household_id.is_unique and len(gainers) == len(donors)
    weight = gainers.person_household_id.map(
        donors.set_index("household_id").household_weight
    )
    for band in bands.itertuples():
        in_band = np.isclose(gainers.capital_gains, band.mean_gain)
        assert in_band.sum() == cg.DONORS_PER_BAND
        assert weight[in_band].sum() == pytest.approx(band.taxpayers)


def test_calibration_starts_with_every_band_count_on_target():
    """End to end: twins, donors, the calibration's priors and the rake.

    Before uk-data#570 the same quantities started 17 times over on the
    taxpayer count and 5 times over on total gains.
    """
    out = _stacked_population()
    year = hmrc_cgt._CGT_BASE_YEAR + 1
    matrix, targets = _band_matrix(out, year)
    names = hmrc_cgt.band_count_target_names()
    assert len(names) == 12

    priors = initialize_weight_priors(out.household.household_weight.to_numpy())
    start = rake_priors_to_targets(priors, matrix, targets, names)

    estimate = start @ matrix
    # Every band has records, so every band count starts on its target, and
    # with them their sum, the aggregate taxpayer count.
    assert (priors @ matrix[names] > 0).all()
    np.testing.assert_allclose(estimate[names], targets[names])
    assert estimate["hmrc/cgt_taxpayers"] == pytest.approx(
        targets["hmrc/cgt_taxpayers"]
    )
    # Households outside the bands keep their prior.
    outside = (matrix[names] == 0).all(axis=1).to_numpy()
    np.testing.assert_array_equal(start[outside], priors[outside])
