"""SPI draws are rebased from the SPI year to the dataset's year.

The income QRF is trained on SPI 2022-23 amounts but draws into an FRS
2024-25 dataset. ``rebase_spi_draws`` multiplies each draw by its variable's
uprating-index ratio, the index ``uprate_dataset`` applies, before the draws
are written, conditioned on (second stage) or stacked.

Invariants (property-tested below, for every input):
- equal years change nothing, bit for bit;
- within a column the map is weakly increasing (monotone) and keeps signs;
- zero stays zero and only zero becomes zero;
- penny amounts keep their ranks, ties included;
- missing draws stay missing;
- the factor is the table's index ratio, so rebasing composes and inverts;
- Gift Aid and qualifying-investment gifts, which have no index, are untouched;
- it agrees with ``uprate_dataset`` on a dataset in the SPI year (differential).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from hypothesis import given, settings, strategies as st
from scipy.stats import rankdata

from policyengine_uk_data.datasets.frs_release import CURRENT_FRS_RELEASE
from policyengine_uk_data.datasets.imputations import income as income_module
from policyengine_uk_data.datasets.imputations.income import (
    IMPUTATIONS,
    SPI_NOMINAL_IMPUTATIONS,
    rebase_spi_draws,
)
from policyengine_uk_data.datasets.spi import SPI_FISCAL_YEAR
from policyengine_uk_data.storage import STORAGE_FOLDER
from policyengine_uk_data.utils.uprating import END_YEAR, START_YEAR

UPRATING = pd.read_csv(STORAGE_FOLDER / "uprating_factors.csv").set_index("Variable")
INDEXED = [column for column in IMPUTATIONS if column not in SPI_NOMINAL_IMPUTATIONS]
FRS_YEAR = CURRENT_FRS_RELEASE.survey_year

years = st.integers(START_YEAR, END_YEAR)
money = st.floats(-1e10, 1e10, allow_nan=False, allow_infinity=False)
pennies = st.integers(-(10**11), 10**11).map(lambda p: p / 100)


def _factor(column: str, start: int, end: int) -> float:
    return UPRATING.loc[column, str(end)] / UPRATING.loc[column, str(start)]


def _draws(values) -> pd.DataFrame:
    """The same values in every imputed column."""
    values = np.asarray(values, dtype=float)
    return pd.DataFrame({column: values for column in IMPUTATIONS})


def test_every_spi_draw_has_exactly_one_rebasing_rule():
    for column in IMPUTATIONS:
        assert (column in UPRATING.index) != (column in SPI_NOMINAL_IMPUTATIONS), column


def test_nominal_draws_have_no_policyengine_uk_index():
    """If policyengine-uk gives these an index, they must be rebased too."""
    import policyengine_uk
    from policyengine_uk.system import system

    load_indices = yaml.safe_load(
        (
            Path(policyengine_uk.__file__).parent / "data" / "uprating_indices.yaml"
        ).read_text()
    )
    indexed_at_load = {v for variables in load_indices.values() for v in variables}
    for column in SPI_NOMINAL_IMPUTATIONS:
        assert system.variables[column].uprating is None, column
        assert column not in indexed_at_load, column


def test_frs_release_is_later_than_the_spi_year():
    """The case this module exists for: draws do move."""
    assert FRS_YEAR > SPI_FISCAL_YEAR
    assert all(_factor(column, SPI_FISCAL_YEAR, FRS_YEAR) > 1 for column in INDEXED)


@settings(deadline=None)
@given(st.lists(money, min_size=1, max_size=40), years)
def test_equal_years_are_the_identity(values, year):
    draws = _draws(values)
    pd.testing.assert_frame_equal(rebase_spi_draws(draws, year, spi_year=year), draws)


@settings(deadline=None)
@given(st.lists(money, min_size=2, max_size=40), years, years)
def test_rebasing_is_monotone_and_keeps_signs(values, spi_year, year):
    draws = _draws(values)
    rebased = rebase_spi_draws(draws, year, spi_year=spi_year)
    order = np.argsort(draws[IMPUTATIONS[0]].to_numpy(), kind="stable")
    for column in IMPUTATIONS:
        assert (np.diff(rebased[column].to_numpy()[order]) >= 0).all(), column
        np.testing.assert_array_equal(np.sign(rebased[column]), np.sign(draws[column]))


@settings(deadline=None)
@given(st.lists(pennies, min_size=1, max_size=40), years, years)
def test_rebasing_keeps_zeros_and_penny_ranks(values, spi_year, year):
    draws = _draws(values)
    rebased = rebase_spi_draws(draws, year, spi_year=spi_year)
    for column in IMPUTATIONS:
        np.testing.assert_array_equal(rebased[column] == 0, draws[column] == 0)
        np.testing.assert_array_equal(
            rankdata(rebased[column]), rankdata(draws[column])
        )


@settings(deadline=None)
@given(st.lists(st.one_of(money, st.just(np.nan)), min_size=1, max_size=40), years)
def test_missing_draws_stay_missing(values, year):
    draws = _draws(values)
    rebased = rebase_spi_draws(draws, year)
    for column in IMPUTATIONS:
        np.testing.assert_array_equal(rebased[column].isna(), draws[column].isna())


@settings(deadline=None)
@given(st.lists(money, min_size=1, max_size=40), years, years, years)
def test_factor_is_the_index_ratio_so_rebasing_composes(values, a, b, c):
    draws = _draws(values)
    direct = rebase_spi_draws(draws, c, spi_year=a)
    via_b = rebase_spi_draws(rebase_spi_draws(draws, b, spi_year=a), c, spi_year=b)
    there_and_back = rebase_spi_draws(rebase_spi_draws(draws, b, spi_year=a), a, b)
    for column in INDEXED:
        np.testing.assert_allclose(
            direct[column], draws[column] * _factor(column, a, c), rtol=1e-12
        )
        np.testing.assert_allclose(via_b[column], direct[column], rtol=1e-12)
        np.testing.assert_allclose(there_and_back[column], draws[column], rtol=1e-12)
    for column in SPI_NOMINAL_IMPUTATIONS:
        pd.testing.assert_series_equal(direct[column], draws[column])


@settings(deadline=None, max_examples=25)
@given(st.lists(st.floats(0, 1e8, allow_nan=False), min_size=1, max_size=20), years)
def test_agrees_with_uprate_dataset_on_an_spi_year_dataset(values, year):
    """Differential: the build's own whole-dataset uprating, started in the SPI year."""
    from policyengine_uk.data import UKSingleYearDataset
    from policyengine_uk_data.utils.uprating import uprate_dataset

    n = len(values)
    ids = np.arange(1, n + 1)
    draws = _draws(values)
    person = draws.assign(person_id=ids, person_benunit_id=ids, person_household_id=ids)
    dataset = UKSingleYearDataset(
        person=person,
        benunit=pd.DataFrame({"benunit_id": ids}),
        household=pd.DataFrame({"household_id": ids}),
        fiscal_year=SPI_FISCAL_YEAR,
    )
    uprated = uprate_dataset(dataset, year).person
    rebased = rebase_spi_draws(draws, year)
    for column in IMPUTATIONS:
        np.testing.assert_allclose(rebased[column], uprated[column], rtol=1e-12)


def test_a_column_without_an_index_or_a_nominal_rule_raises():
    with pytest.raises(KeyError):
        rebase_spi_draws(pd.DataFrame({"not_an_spi_income": [1.0]}), FRS_YEAR)


class _FixedDraws:
    """A model whose draws are known: column k of row i is (i + 1) * 1000 * (k + 1)."""

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        rows = np.arange(1, len(X) + 1)[:, None] * 1_000.0
        return pd.DataFrame(
            rows * np.arange(1, len(IMPUTATIONS) + 1), columns=IMPUTATIONS
        )


class _FakeSimulation:
    def __init__(self, dataset):
        self.n = len(dataset.person)

    def calculate_dataframe(self, columns):
        return pd.DataFrame(
            {
                "age": [40] * self.n,
                "gender": ["MALE"] * self.n,
                "region": ["LONDON"] * self.n,
            }
        )


class _Dataset:
    def __init__(self, person, time_period):
        self.person, self.time_period = person, time_period

    def copy(self):
        return _Dataset(self.person.copy(), self.time_period)


@pytest.mark.parametrize("time_period", [FRS_YEAR, str(FRS_YEAR), SPI_FISCAL_YEAR])
def test_impute_over_incomes_writes_draws_in_the_datasets_year(
    monkeypatch, time_period
):
    monkeypatch.setattr(income_module, "Microsimulation", _FakeSimulation)
    # Working-age employees, so an earnings-group draw (#529) draws them too.
    person = pd.DataFrame(
        {
            "employment_status": "FT_EMPLOYED",
            **{c: np.full(3, -1.0) for c in IMPUTATIONS},
        }
    )
    expected = _FixedDraws().predict(person)

    result = income_module.impute_over_incomes(
        _Dataset(person, time_period), _FixedDraws(), IMPUTATIONS
    )

    for column in IMPUTATIONS:
        factor = (
            1.0
            if column in SPI_NOMINAL_IMPUTATIONS
            else _factor(column, SPI_FISCAL_YEAR, int(time_period))
        )
        np.testing.assert_allclose(
            result.person[column], expected[column] * factor, rtol=1e-12
        )


def test_second_stage_and_frs_dividends_see_rebased_draws(monkeypatch):
    """Only the draws are rebased, and before the second stage sees them.

    The SPI-synthetic copy reaches the FRS-only QRF with rebased draws; the
    FRS respondents that QRF trains on, the FRS rows' undrawn incomes and
    every undrawn money column keep their survey-year values. The FRS half's
    dividend draw is rebased too. (If #498 lands, FRS dividends are no longer
    drawn and that last assertion goes.)"""
    from policyengine_uk_data.datasets import disability_benefits
    from policyengine_uk_data.datasets.imputations import frs_only

    person = pd.DataFrame(
        {
            "person_id": [1, 2],
            "person_household_id": [1, 2],
            "person_benunit_id": [1, 2],
            "employment_status": ["FT_EMPLOYED", "FT_EMPLOYED"],
            **{column: [1_234.0, 56_789.0] for column in IMPUTATIONS},
            # Undrawn and indexed in the uprating table: must not move.
            "employee_pension_contributions": [500.0, 2_500.0],
        }
    )
    household = pd.DataFrame({"household_id": [1, 2], "household_weight": [1.0, 1.0]})

    class _FullDataset(_Dataset):
        def __init__(self, person, household, time_period):
            super().__init__(person, time_period)
            self.household = household

        def copy(self):
            return _FullDataset(
                self.person.copy(), self.household.copy(), self.time_period
            )

        def validate(self):
            return None

    seen = {}

    def _capture_stage_two(train_dataset, target_dataset):
        seen["train"] = train_dataset.person.copy()
        seen["target"] = target_dataset.person.copy()
        return target_dataset

    monkeypatch.setattr(income_module, "Microsimulation", _FakeSimulation)
    monkeypatch.setattr(income_module, "create_income_model", _FixedDraws)
    monkeypatch.setattr(
        income_module, "subsample_dataset", lambda dataset, _size: dataset.copy()
    )
    monkeypatch.setattr(frs_only, "impute_frs_only_variables", _capture_stage_two)
    monkeypatch.setattr(
        disability_benefits,
        "strip_internal_disability_reported_amounts",
        lambda dataset: dataset,
    )
    monkeypatch.setattr(income_module, "stack_datasets", lambda frs, spi: (frs, spi))

    frs, spi = income_module.impute_income(_FullDataset(person, household, FRS_YEAR))

    raw = _FixedDraws().predict(person)
    rebased = {
        column: raw[column]
        * (
            1.0
            if column in SPI_NOMINAL_IMPUTATIONS
            else _factor(column, SPI_FISCAL_YEAR, FRS_YEAR)
        )
        for column in IMPUTATIONS
    }
    for column in IMPUTATIONS:
        np.testing.assert_allclose(seen["target"][column], rebased[column], rtol=1e-12)
        np.testing.assert_allclose(spi.person[column], rebased[column], rtol=1e-12)
        np.testing.assert_array_equal(seen["train"][column], person[column])
    for column in INDEXED:
        if column != "dividend_income":
            np.testing.assert_array_equal(frs.person[column], person[column])
    np.testing.assert_allclose(
        frs.person["dividend_income"], rebased["dividend_income"], rtol=1e-12
    )
    for half in (frs, spi, seen["train"], seen["target"]):
        table = half if isinstance(half, pd.DataFrame) else half.person
        np.testing.assert_array_equal(
            table["employee_pension_contributions"],
            person["employee_pension_contributions"],
        )
