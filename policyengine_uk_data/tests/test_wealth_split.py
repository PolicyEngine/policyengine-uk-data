"""Invariants of the WAS wealth split: pensions out of corporate_wealth, the
share-like components, buy-to-let property and secured debts."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st

_WEALTH_PATH = (
    Path(__file__).resolve().parents[1] / "datasets" / "imputations" / "wealth.py"
)
_WEALTH_SPEC = importlib.util.spec_from_file_location(
    "wealth_split_module", _WEALTH_PATH
)
wealth = importlib.util.module_from_spec(_WEALTH_SPEC)
_WEALTH_SPEC.loader.exec_module(wealth)

COMPONENTS = list(wealth.CORPORATE_WEALTH_COMPONENTS)
DEBTS = list(wealth.SECURED_DEBT_ASSETS)

amount = st.floats(min_value=0, max_value=1e8, allow_nan=False)
signed_amount = st.floats(min_value=-1e7, max_value=1e8, allow_nan=False)

WAS_SOURCE_COLUMNS = {
    "pensions": "totalpenr8_aggr",
    "db_pensions": "dvvaldbt_scaper8_aggr",
    "uk_shares": "DVFShUKVR8_aggr",
    "emp_shares": "DVFESHARESR8_aggr",
    "isa": "DVIISAVR8_aggr",
    "cash_isa": "DVCISAVR8_aggr",
    "trusts": "DVFCollVR8_aggr",
    "second_homes": "DVHseValR8_sum",
    "second_homes_debt": "DVHseDebtR8_sum",
    "btl": "DVBltValR8_sum",
    "btl_debt": "DVBLtDebtR8_sum",
    "buildings_debt": "DVBldDebtR8_sum",
    "land_debt": "DVLUKDebtR8_sum",
}


def _was_row(**values):
    row = {column: 0 for column in wealth.WAS_RENAMES}
    row.update({"R8xshhwgt": 1, "GORR8": 11, "DVPriRntR8": 1})
    for key, value in values.items():
        row[WAS_SOURCE_COLUMNS[key]] = value
    return row


@st.composite
def was_rows(draw):
    pensions = draw(amount)
    return _was_row(
        pensions=pensions,
        db_pensions=draw(st.floats(min_value=0, max_value=pensions)),
        **{
            key: draw(amount)
            for key in WAS_SOURCE_COLUMNS
            if key not in ("pensions", "db_pensions")
        },
    )


def test_every_imputed_target_is_built_from_the_survey():
    was = wealth.generate_was_table(pd.DataFrame([_was_row()]))
    missing = set(wealth.IMPUTE_VARIABLES) - set(was.columns)
    assert not missing


def test_corporate_wealth_is_derived_not_imputed():
    assert "corporate_wealth" not in wealth.IMPUTE_VARIABLES
    assert set(COMPONENTS) <= set(wealth.IMPUTE_VARIABLES)
    assert "private_pension_wealth" in wealth.IMPUTE_VARIABLES


def test_each_secured_debt_is_imputed_after_its_asset():
    order = wealth.IMPUTE_VARIABLES
    for debt, asset in wealth.SECURED_DEBT_ASSETS.items():
        assert order.index(asset) < order.index(debt)


def test_definitions_are_part_of_the_model_metadata():
    metadata = wealth.get_wealth_model_metadata()
    assert dict(metadata["derived_targets"]) == wealth.DERIVED_TARGETS


def test_generate_was_table_maps_each_component():
    was = wealth.generate_was_table(
        pd.DataFrame(
            [
                _was_row(
                    pensions=500,
                    db_pensions=200,
                    uk_shares=10,
                    emp_shares=20,
                    isa=40,
                    cash_isa=7,
                    trusts=30,
                    second_homes=1_000,
                    second_homes_debt=100,
                    btl=2_000,
                    btl_debt=900,
                    buildings_debt=50,
                    land_debt=5,
                )
            ]
        )
    ).iloc[0]
    assert was.private_pension_wealth == 300
    assert was.directly_held_shares == 30
    assert was.unit_and_investment_trusts == 30
    assert was.stocks_and_shares_isa == 40
    assert was.cash_isa == 7
    assert was.corporate_wealth == 100
    assert was.other_residential_property_value == 3_000
    assert was.other_residential_property_secured_debt == 1_000
    assert was.non_residential_property_secured_debt == 50
    assert was.owned_land_secured_debt == 5


@settings(max_examples=200, deadline=None)
@given(was_rows())
def test_split_conserves_the_old_corporate_wealth(row):
    """Old corporate_wealth = new corporate_wealth + private_pension_wealth."""
    was = wealth.generate_was_table(pd.DataFrame([row])).iloc[0]
    old_corporate_wealth = (
        row["totalpenr8_aggr"]
        - row["dvvaldbt_scaper8_aggr"]
        + row["DVFESHARESR8_aggr"]
        + row["DVFShUKVR8_aggr"]
        + row["DVIISAVR8_aggr"]
        + row["DVFCollVR8_aggr"]
    )
    assert np.isclose(
        was.corporate_wealth + was.private_pension_wealth,
        old_corporate_wealth,
        rtol=1e-12,
        atol=1e-6,
    )
    assert np.isclose(
        was.corporate_wealth, sum(was[c] for c in COMPONENTS), rtol=1e-12, atol=1e-6
    )
    assert was.private_pension_wealth >= 0
    assert (
        was.other_residential_property_value
        == row["DVHseValR8_sum"] + row["DVBltValR8_sum"]
    )


@st.composite
def draws(draw):
    n = draw(st.integers(min_value=1, max_value=20))
    columns = {
        column: draw(st.lists(signed_amount, min_size=n, max_size=n))
        for column in (*COMPONENTS, *DEBTS, *wealth.SECURED_DEBT_ASSETS.values())
    }
    # Exact zero assets must be reachable, since that is where debt is cleared.
    for asset in wealth.SECURED_DEBT_ASSETS.values():
        zeros = draw(st.lists(st.booleans(), min_size=n, max_size=n))
        columns[asset] = [0.0 if z else v for z, v in zip(zeros, columns[asset])]
    return pd.DataFrame(columns)


@settings(max_examples=300, deadline=None)
@given(draws())
def test_derived_outputs_hold_their_identities(output):
    derived = wealth.derive_wealth_outputs(output)
    assert (derived[COMPONENTS] >= 0).all().all()
    assert (derived[DEBTS] >= 0).all().all()
    assert np.allclose(
        derived.corporate_wealth, derived[COMPONENTS].sum(axis=1), rtol=0, atol=0
    )
    for debt, asset in wealth.SECURED_DEBT_ASSETS.items():
        no_asset = derived[asset] <= 0
        assert (derived.loc[no_asset, debt] == 0).all()
        held = ~no_asset
        assert np.array_equal(
            derived.loc[held, debt], output.loc[held, debt].clip(lower=0)
        )
    # Assets themselves are untouched, and deriving twice changes nothing.
    for asset in wealth.SECURED_DEBT_ASSETS.values():
        assert derived[asset].equals(output[asset])
    pd.testing.assert_frame_equal(wealth.derive_wealth_outputs(derived), derived)


def test_each_target_draws_its_quantile_independently():
    """microimpute seeds every target alike; the wealth model must not."""
    from types import SimpleNamespace

    models = {v: SimpleNamespace(seed=42) for v in wealth.IMPUTE_VARIABLES}
    model = SimpleNamespace(model=SimpleNamespace(models=models))
    wealth.use_independent_quantile_draws(model)
    seeds = [models[v].seed for v in wealth.IMPUTE_VARIABLES]
    assert len(set(seeds)) == len(seeds)
    assert dict(wealth.get_wealth_model_metadata()["quantile_draw_seeds"]) == {
        v: models[v].seed for v in wealth.IMPUTE_VARIABLES
    }
