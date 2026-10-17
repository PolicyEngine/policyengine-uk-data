"""Tests that the inputs VAT is computed from cannot be negative.

policyengine-uk charges VAT on consumption times the full-rate expenditure
rate, with no floor. On the 1.58.0 enhanced FRS, 44 households had negative
total consumption and 32 a negative rate, so 76 had negative VAT
(uk-data#565). Both came from survey training rows that a QRF reproduced.
"""

from unittest import mock

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from policyengine_uk_data.datasets.frs import WEEKS_IN_YEAR
from policyengine_uk_data.datasets.frs_release import CURRENT_FRS_RELEASE
from policyengine_uk_data.datasets.imputations import consumption, vat

CATEGORY_CODES = [f"p{code}" for code in range(601, 613)]
CATEGORIES = [consumption.CONSUMPTION_VARIABLE_RENAMES[code] for code in CATEGORY_CODES]


def _lcfs_household(category_amounts: np.ndarray) -> pd.DataFrame:
    """LCFS households whose only varying fields are the twelve categories."""
    n = len(category_amounts)
    constants = {
        "g018": 2,
        "g019": 1,
        "gorx": 7,
        "p389p": 1_000.0,
        "p344p": 1_500.0,
        "weighta": 0.5,
        "a121": 2,
        "a122": 5,
        "b226": 10.0,
        "b489": 0.0,
        "b490": 0.0,
        "p537": 20.0,
        "c72211": 5.0,
        "c72212": 6.0,
        "c73212": 2.0,
        "c73213": 3.0,
        "c73214": 1.0,
    }
    household = pd.DataFrame({"case": np.arange(1, n + 1)})
    for column, value in constants.items():
        household[column] = value
    household[CATEGORY_CODES] = category_amounts
    return household


def _lcfs_table(category_amounts: np.ndarray) -> pd.DataFrame:
    household = _lcfs_household(category_amounts)
    person = pd.DataFrame(
        {
            "case": household.case,
            "b303p": 100.0,
            "b3262p": 10.0,
            "b3381": 0.0,
            "p049p": 5.0,
        }
    )
    with mock.patch.object(
        consumption,
        "impute_has_fuel_to_lcfs",
        lambda table: table.assign(has_fuel_consumption=1.0),
    ):
        return consumption.generate_lcfs_table(person, household)


weekly_amounts = st.lists(
    st.lists(st.floats(-500.0, 2_000.0), min_size=12, max_size=12),
    min_size=1,
    max_size=20,
).map(np.array)


@given(weekly_amounts)
@settings(max_examples=50, deadline=None)
def test_consumption_training_table_clamps_negative_amounts_only(amounts):
    table = _lcfs_table(amounts)

    assert len(table) == len(amounts)
    assert (table[consumption.IMPUTATIONS] >= 0).all().all()
    # A category is its LCFS amount, annualised, unless that was negative.
    np.testing.assert_allclose(
        table[CATEGORIES].to_numpy(), np.clip(amounts, 0, None) * WEEKS_IN_YEAR
    )


def test_negative_housing_amount_no_longer_makes_consumption_negative():
    """The case in the survey: a negative housing amount that outweighs the
    household's other eleven categories."""
    amounts = np.full((1, 12), 10.0)
    amounts[0, CATEGORY_CODES.index("p604")] = -264.85
    table = _lcfs_table(amounts)
    assert table["housing_water_and_electricity_consumption"].iloc[0] == 0
    assert table[CATEGORIES].sum(axis=1).iloc[0] == pytest.approx(110 * WEEKS_IN_YEAR)


etb_rows = st.lists(
    st.tuples(st.floats(0.0, 500.0), st.floats(-300.0, 3_000.0)),
    min_size=1,
    max_size=30,
)


@given(etb_rows)
@settings(max_examples=200, deadline=None)
def test_vat_training_rate_is_a_share_of_expenditure(rows):
    totvat, expdis = (np.array(column) for column in zip(*rows))
    year = vat.DEFAULT_ETB_YEAR
    etb = pd.DataFrame(
        {
            "year": year,
            "adults": 2,
            "childs": 1,
            "noretd": 0,
            "disinc": 600.0,
            "totvat": totvat,
            "expdis": expdis,
        }
    )
    table = vat.generate_etb_table(etb, year=year)
    rate = table.full_rate_vat_expenditure_rate

    # A row is kept exactly when it has expenditure left after VAT.
    defined = expdis - totvat > 0
    assert table.index.tolist() == np.flatnonzero(defined).tolist()
    assert rate.between(0, 1).all()
    # A rate that was already a share is unchanged; one above one is one.
    standard_rate, reduced_rate_share = vat._get_vat_parameters(year)
    with np.errstate(divide="ignore", invalid="ignore"):
        raw = (totvat * (1 - reduced_rate_share) / standard_rate) / (expdis - totvat)
    np.testing.assert_allclose(rate.to_numpy(), np.minimum(raw[defined], 1))


def test_guard_names_the_negative_column():
    household = pd.DataFrame({"a_consumption": [1.0, -2.0, -3.0], "b_spending": 0.0})
    consumption._require_non_negative(household, ["b_spending"])
    with pytest.raises(ValueError, match="a_consumption is negative for 2"):
        consumption._require_non_negative(household, ["b_spending", "a_consumption"])


@pytest.mark.parametrize("bad_rate", [-0.01, 1.01])
def test_impute_vat_fails_on_a_rate_outside_the_unit_interval(monkeypatch, bad_rate):
    class _Model:
        input_columns = ("is_adult",)

        def predict(self, frame):
            return pd.DataFrame({"full_rate_vat_expenditure_rate": [0.5, bad_rate]})

    class _Dataset:
        household = pd.DataFrame({"household_id": [1, 2]})

        def copy(self):
            return self

        def validate(self):
            return None

    class _Simulation:
        def __init__(self, dataset):
            pass

        def calculate_dataframe(self, columns, map_to):
            return pd.DataFrame({"is_adult": [1, 2]})

    monkeypatch.setattr(vat, "create_vat_model", lambda: _Model())
    monkeypatch.setattr(vat, "Microsimulation", _Simulation)
    with pytest.raises(ValueError, match="outside \\[0, 1\\] for 1 households"):
        vat.impute_vat(_Dataset())


def test_models_trained_before_the_fix_are_retrained():
    """A cached model is reused only when its metadata matches, so the
    metadata must tell a model trained on the uncleaned tables apart."""
    consumption_metadata = consumption.get_consumption_model_metadata()
    stale = type("Model", (), {})()
    stale.metadata = {
        key: value
        for key, value in consumption_metadata.items()
        if key != "impute_variables_minimum"
    }
    stale.model = type("Inner", (), {"imputed_variables": consumption.IMPUTATIONS})()
    assert not consumption._qrf_model_matches_current_metadata(
        stale, consumption_metadata, consumption.IMPUTATIONS
    )
    stale.metadata = consumption_metadata
    assert consumption._qrf_model_matches_current_metadata(
        stale, consumption_metadata, consumption.IMPUTATIONS
    )

    vat_metadata = vat.get_vat_model_metadata()
    stale.metadata = {k: v for k, v in vat_metadata.items() if k != "rate_bounds"}
    stale.model = type("Inner", (), {"imputed_variables": vat.IMPUTATIONS})()
    assert not vat._vat_model_matches_current_release(stale)
    stale.metadata = vat_metadata
    assert vat._vat_model_matches_current_release(stale)


@pytest.mark.slow
def test_built_vat_inputs_are_non_negative(enhanced_frs):
    household = enhanced_frs.household
    for column in consumption.IMPUTATIONS:
        negative = int((household[column] < 0).sum())
        assert not negative, f"{column} is negative for {negative} households."
    rate = household["full_rate_vat_expenditure_rate"]
    assert rate.between(0, 1).all(), (
        f"The full-rate VAT expenditure rate is outside [0, 1] for "
        f"{int((~rate.between(0, 1)).sum())} households."
    )


@pytest.mark.slow
@pytest.mark.parametrize("year", [CURRENT_FRS_RELEASE.calibration_year, 2030])
def test_built_vat_is_non_negative_for_every_household(baseline, year):
    """Holds for a reduced build as for a full one: it is a property of the
    inputs, not of the calibration."""
    vat_amount = baseline.calculate("vat", year).values
    assert (vat_amount >= 0).all(), (
        f"{int((vat_amount < 0).sum())} households have negative VAT in {year}."
    )
