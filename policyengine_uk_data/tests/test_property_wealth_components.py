"""Preserve the seven WAS gross-property categories without a saved total.

The local Reform is the companion policyengine-uk contract specified in
docs/property-wealth-model-changes.md. It is used explicitly: passing these
synthetic tests does not mean the installed production model has that change.

The synthetic tests need no private data. The built-dataset tests run after
`make data` and are skipped when the enhanced FRS is absent. Failure messages
report counts and maxima only, never individual records.
"""

import builtins
import importlib
import io
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import yaml
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from policyengine_uk import Microsimulation
from policyengine_uk.data import UKSingleYearDataset
from policyengine_uk.model_api import GBP, STOCK, YEAR, Household, Reform, Variable
from policyengine_uk.utils.scenario import Scenario

from policyengine_uk_data.datasets.frs_release import CURRENT_FRS_RELEASE
from policyengine_uk_data.datasets.imputations.regional_property_uprating import (
    _load_regional_house_prices,
    uprate_property_by_region,
)
from policyengine_uk_data.datasets.imputations.wealth import (
    CONDITIONING_ONLY_VARIABLES,
    DERIVED_TARGETS,
    IMPUTE_VARIABLES,
    PROPERTY_COMPONENTS,
    QUANTILE_DRAW_SEEDS,
    WAS_RENAMES,
    derive_wealth_outputs,
    generate_was_table,
    require_property_component_model,
    store_wealth_predictions,
)
from policyengine_uk_data.storage import STORAGE_FOLDER
from policyengine_uk_data.utils.uprating import uprate_dataset

COMPONENTS = (
    "main_residence_value",
    "other_residential_property_value",
    "non_residential_property_value",
    "owned_land",
    "overseas_property_value",
    "other_property_value",
)
# Explicit specification, independent of the implementation's rename table.
SOURCE_COMPONENTS = (
    "DVhvalueR8",
    "DVHseValR8_sum",
    "DVBltValR8_sum",
    "DVBlDValR8_sum",
    "DVLUKValR8_sum",
    "DVLOSValR8_sum",
    "DVOprValR8_sum",
)
# Seven independent source terms become six model inputs: #501 combines the
# two kinds of residential property other than the main residence.
SOURCE_INPUTS = (
    "main_residence_value",
    "other_residential_property_value",
    "other_residential_property_value",
    "non_residential_property_value",
    "owned_land",
    "overseas_property_value",
    "other_property_value",
)
NEW_COMPONENTS = (
    "overseas_property_value",
    "other_property_value",
)
BASE_YEAR = CURRENT_FRS_RELEASE.base_year
# Two years past the base year: 2026 for the 2024-25 FRS.
PROJECTION_YEAR = BASE_YEAR + 2
YEARS = sorted({BASE_YEAR, CURRENT_FRS_RELEASE.calibration_year, PROJECTION_YEAR})
RTOL = 1e-5


class PropertyComponentsContract(Reform):
    """Executable contract for the model change required before release."""

    def apply(self):
        for name in NEW_COMPONENTS:
            variable = type(
                name,
                (Variable,),
                {
                    "label": name.replace("_", " ").capitalize(),
                    "value_type": float,
                    "entity": Household,
                    "definition_period": YEAR,
                    "unit": GBP,
                    "quantity_type": STOCK,
                    "uprating": "gov.economic_assumptions.indices.obr.per_capita.gdp",
                },
            )
            if name in self.variables:
                self.update_variable(variable)
            else:
                self.add_variable(variable)

        class residential_property_value(Variable):
            value_type = float
            entity = Household
            definition_period = YEAR
            unit = GBP
            quantity_type = STOCK
            adds = [
                "main_residence_value",
                "other_residential_property_value",
            ]

        class property_wealth(Variable):
            value_type = float
            entity = Household
            definition_period = YEAR
            unit = GBP
            quantity_type = STOCK
            adds = [
                "residential_property_value",
                "non_residential_property_value",
                "owned_land",
                "overseas_property_value",
                "other_property_value",
            ]

        class household_land_value(Variable):
            value_type = float
            entity = Household
            definition_period = YEAR
            unit = GBP
            quantity_type = STOCK

            def formula(household, period, parameters):
                region = household("region", period)
                regional = parameters(
                    period
                ).household.wealth.land.intensity.property_wealth_by_region
                regions = tuple(_load_regional_house_prices()) + ("NORTHERN_IRELAND",)
                intensity = np.select(
                    [
                        region == getattr(region.possible_values, name)
                        for name in regions
                    ],
                    [getattr(regional, name) for name in regions],
                    default=regional.UNKNOWN,
                )
                land = household("owned_land", period)
                overseas = household("overseas_property_value", period)
                domestic_property = (
                    household("property_wealth", period) - land - overseas
                )
                return domestic_property * intensity + land

        self.update_variable(residential_property_value)
        self.update_variable(property_wealth)
        self.update_variable(household_land_value)


@contextmanager
def _contract_uprating_spec(omit=()):
    # The model extends datasets before setting inputs. That extension reads
    # this YAML, so Variable.uprating alone does not uprate new dataset columns.
    # Exercise the companion YAML entries in memory, without changing the
    # installed model or treating the production dependency as completed.
    economic_assumptions = importlib.import_module(
        "policyengine_uk.data.economic_assumptions"
    )
    config_path = Path(economic_assumptions.__file__).with_name("uprating_indices.yaml")
    indices = yaml.safe_load(config_path.read_text())
    for names in indices.values():
        names[:] = [name for name in names if name not in omit]
    gdp_variables = indices["gov.economic_assumptions.yoy_growth.obr.per_capita.gdp"]
    gdp_variables.extend(
        name
        for name in NEW_COMPONENTS
        if name not in gdp_variables and name not in omit
    )
    contents = yaml.safe_dump(indices)

    def contract_yaml(filename, *args, **kwargs):
        if Path(filename).resolve() == config_path.resolve():
            return io.StringIO(contents)
        return builtins.open(filename, *args, **kwargs)

    wealth = importlib.import_module("policyengine_uk_data.datasets.imputations.wealth")
    with (
        patch.object(economic_assumptions, "open", contract_yaml, create=True),
        patch.object(wealth, "open", contract_yaml, create=True),
    ):
        yield


def _contract_simulation(dataset):
    # Apply the variable definitions before dataset loading can skip them.
    scenario = Scenario(
        applied_before_data_load=True,
        simulation_modifier=lambda sim: setattr(
            sim,
            "tax_benefit_system",
            PropertyComponentsContract(sim.tax_benefit_system),
        ),
    )
    simulation_module = importlib.import_module("policyengine_uk.simulation")
    model = importlib.import_module("policyengine_uk.system").system
    # Clone the real, already loaded model rather than reading its thousands
    # of source/parameter files again for each Hypothesis example.
    with (
        _contract_uprating_spec(),
        patch.object(simulation_module, "CountryTaxBenefitSystem", model.clone),
    ):
        return Microsimulation(dataset=dataset, scenario=scenario)


def _source_table(component_values):
    raw = pd.DataFrame(
        0.0,
        index=range(len(component_values)),
        columns=list(dict.fromkeys((*WAS_RENAMES, *SOURCE_COMPONENTS, "DVPropertyR8"))),
    )
    raw["R8xshhwgt"] = 1.0
    raw["GORR8"] = 8
    raw.loc[:, list(SOURCE_COMPONENTS)] = component_values
    raw["DVPropertyR8"] = np.asarray(component_values).sum(axis=1)
    return raw


def test_source_mapping_preserves_each_of_seven_categories():
    """Each row owns only one category, including each formerly omitted one."""
    amounts = np.arange(1, 8) * 10_000.0
    mapped = generate_was_table(_source_table(np.diag(amounts)))
    expected = np.zeros((7, 6))
    expected[np.arange(7), [COMPONENTS.index(name) for name in SOURCE_INPUTS]] = amounts
    assert PROPERTY_COMPONENTS == COMPONENTS
    np.testing.assert_array_equal(mapped[list(COMPONENTS)].to_numpy(), expected)
    np.testing.assert_array_equal(
        mapped["property_wealth"], mapped[list(COMPONENTS)].sum(axis=1)
    )


@settings(max_examples=30, deadline=None)
@given(
    components=st.lists(
        st.integers(min_value=0, max_value=5_000_000), min_size=7, max_size=7
    )
)
def test_source_total_reconciles_before_independent_imputation(components):
    mapped = generate_was_table(_source_table([components]))
    assert mapped["property_wealth"].iloc[0] == sum(components)
    assert mapped[list(COMPONENTS)].sum(axis=1).iloc[0] == sum(components)


@pytest.mark.parametrize("source", SOURCE_COMPONENTS)
def test_source_mapping_requires_every_property_category(source):
    raw = _source_table([[0.0] * 7]).drop(columns=source)
    with pytest.raises(ValueError, match="Could not find column"):
        generate_was_table(raw)


@pytest.mark.parametrize("missing", [np.nan, -8.0, -9.0])
def test_source_mapping_normalises_property_missing_codes(missing):
    raw = _source_table([[0.0] * 7])
    raw.loc[0, list(SOURCE_COMPONENTS) + ["DVPropertyR8"]] = missing
    mapped = generate_was_table(raw)
    assert (mapped[list(COMPONENTS) + ["property_wealth"]] == 0).all().all()


@pytest.mark.parametrize("source", ["DVHseValR8_sum", "DVBltValR8_sum"])
@pytest.mark.parametrize("missing", [np.nan, -8.0, -9.0])
def test_combined_residential_sources_normalise_missing_codes(source, missing):
    """A missing source does not reduce the positive value of its sibling."""
    raw = _source_table([[0.0, 100.0, 200.0, 0.0, 0.0, 0.0, 0.0]])
    raw.loc[0, source] = missing
    expected = 200.0 if source == "DVHseValR8_sum" else 100.0
    raw.loc[0, "DVPropertyR8"] = expected
    mapped = generate_was_table(raw)
    assert mapped["other_residential_property_value"].iloc[0] == expected
    assert mapped[list(COMPONENTS)].sum(axis=1).iloc[0] == expected


@pytest.mark.parametrize("source", (*SOURCE_COMPONENTS, "DVPropertyR8"))
def test_source_mapping_rejects_unrecognised_negative_gross_value(source):
    raw = _source_table([[0.0] * 7])
    raw.loc[0, source] = -1.0
    with pytest.raises(ValueError, match="negative"):
        generate_was_table(raw)


@pytest.mark.parametrize("source", ["DVHseValR8_sum", "DVBltValR8_sum"])
def test_combined_residential_source_cannot_mask_a_negative_gross_value(source):
    """Validate raw sources before a positive sibling can conceal a negative."""
    raw = _source_table([[0.0, 100.0, 200.0, 0.0, 0.0, 0.0, 0.0]])
    raw.loc[0, source] = -1.0
    with pytest.raises(ValueError, match="negative"):
        generate_was_table(raw)


def test_build_requires_companion_model_change(monkeypatch):
    from policyengine_uk.system import system

    monkeypatch.delitem(system.variables, "overseas_property_value", raising=False)
    with pytest.raises(ValueError, match="companion policyengine-uk change"):
        require_property_component_model()


def test_companion_model_contract_satisfies_build_guard(monkeypatch):
    system_module = importlib.import_module("policyengine_uk.system")
    contract = PropertyComponentsContract(system_module.system)
    monkeypatch.setattr(system_module, "system", contract)
    with _contract_uprating_spec():
        require_property_component_model()


def test_build_guard_rejects_obsolete_buy_to_let_addition(monkeypatch):
    system_module = importlib.import_module("policyengine_uk.system")
    contract = PropertyComponentsContract(system_module.system)
    monkeypatch.setattr(
        contract.variables["residential_property_value"],
        "adds",
        [
            "main_residence_value",
            "other_residential_property_value",
            "buy_to_let_property_value",
        ],
    )
    monkeypatch.setattr(system_module, "system", contract)
    with _contract_uprating_spec():
        with pytest.raises(ValueError, match="property aggregates"):
            require_property_component_model()


def test_build_guard_rejects_old_land_formula(monkeypatch):
    system_module = importlib.import_module("policyengine_uk.system")
    contract = PropertyComponentsContract(system_module.system)

    def old_london_formula(household, period, parameters):
        regional = parameters(
            period
        ).household.wealth.land.intensity.property_wealth_by_region
        return household("property_wealth", period) * regional.LONDON + household(
            "owned_land", period
        )

    monkeypatch.setattr(
        contract.variables["household_land_value"],
        "get_formula",
        lambda period: old_london_formula,
    )
    monkeypatch.setattr(system_module, "system", contract)
    with _contract_uprating_spec():
        with pytest.raises(ValueError, match="land"):
            require_property_component_model()


def test_build_guard_rejects_missing_dataset_uprating_registration(monkeypatch):
    system_module = importlib.import_module("policyengine_uk.system")
    contract = PropertyComponentsContract(system_module.system)
    monkeypatch.setattr(system_module, "system", contract)
    with _contract_uprating_spec(omit=("overseas_property_value",)):
        with pytest.raises(ValueError, match="uprating registration"):
            require_property_component_model()


def test_build_guard_rejects_a_different_non_null_uprating_index(monkeypatch):
    system_module = importlib.import_module("policyengine_uk.system")
    contract = PropertyComponentsContract(system_module.system)
    monkeypatch.setattr(system_module, "system", contract)
    monkeypatch.setattr(
        contract.variables["overseas_property_value"],
        "uprating",
        "gov.economic_assumptions.indices.ons.population",
    )
    with _contract_uprating_spec():
        with pytest.raises(ValueError, match="model uprating index"):
            require_property_component_model()


def test_build_guard_rejects_formula_discarding_domestic_property(monkeypatch):
    system_module = importlib.import_module("policyengine_uk.system")
    contract = PropertyComponentsContract(system_module.system)
    monkeypatch.setattr(system_module, "system", contract)
    monkeypatch.setattr(
        contract.variables["household_land_value"],
        "get_formula",
        lambda period: (
            lambda household, period, parameters: household("owned_land", period)
        ),
    )
    with _contract_uprating_spec():
        with pytest.raises(ValueError, match="land formula"):
            require_property_component_model()


@pytest.mark.parametrize(
    "filename", ["uprating_factors.csv", "uprating_growth_factors.csv"]
)
def test_new_components_have_materialisation_uprating_indices(filename):
    factors = pd.read_csv(STORAGE_FOLDER / filename, index_col="Variable")
    for component in NEW_COMPONENTS:
        np.testing.assert_array_equal(
            factors.loc[component].to_numpy(),
            factors.loc["non_residential_property_value"].to_numpy(),
        )


def _calculate(sim, variable, year) -> np.ndarray:
    return np.asarray(sim.calculate(variable, year).values, dtype=float)


def _property_arrays(sim, years) -> dict:
    out = {}
    for year in years:
        main = _calculate(sim, "main_residence_value", year)
        components = sum(_calculate(sim, name, year) for name in COMPONENTS)
        out[year] = {
            "property_wealth": _calculate(sim, "property_wealth", year),
            "main_residence_value": main,
            "components": components,
        }
    return out


def _n_not_close(actual, expected) -> tuple[int, float]:
    gap = np.abs(actual - expected)
    bad = gap > RTOL * np.abs(expected) + 1.0
    return int(bad.sum()), float(gap[bad].sum())


def _synthetic_dataset(household: pd.DataFrame) -> UKSingleYearDataset:
    n = len(household)
    ids = np.arange(n)
    household = household.copy()
    for name, default in {
        "household_weight": 1.0,
        "region": "LONDON",
        "council_tax": 0.0,
        "rent": 0.0,
        "tenure_type": "OWNED_OUTRIGHT",
    }.items():
        if name not in household:
            household[name] = default
    return UKSingleYearDataset(
        person=pd.DataFrame(
            {
                "person_id": ids,
                "person_benunit_id": ids,
                "person_household_id": ids,
                "age": np.full(n, 45),
            }
        ),
        benunit=pd.DataFrame({"benunit_id": ids}),
        household=household.assign(household_id=ids),
        fiscal_year=BASE_YEAR,
    )


@pytest.mark.parametrize("component", COMPONENTS)
def test_household_owning_only_one_category_keeps_its_value_and_growth(component):
    household = pd.DataFrame(
        {
            "region": ["LONDON"],
            **{name: [100_000.0 if name == component else 0.0] for name in COMPONENTS},
        }
    )
    simulation = _contract_simulation(_synthetic_dataset(household))
    np.testing.assert_allclose(
        _calculate(simulation, "property_wealth", BASE_YEAR), [100_000.0]
    )
    projected = _calculate(simulation, component, PROJECTION_YEAR)
    assert projected[0] > 100_000.0
    np.testing.assert_allclose(
        _calculate(simulation, "property_wealth", PROJECTION_YEAR), projected
    )


def test_property_wealth_is_imputed_before_its_components():
    """Non-land components follow the total; the existing land draw stays first."""
    assert set(CONDITIONING_ONLY_VARIABLES) <= set(IMPUTE_VARIABLES)
    position = IMPUTE_VARIABLES.index("property_wealth")
    assert "property_wealth" in CONDITIONING_ONLY_VARIABLES
    assert IMPUTE_VARIABLES.index("owned_land") < position
    for component in set(COMPONENTS) - {"owned_land"}:
        assert IMPUTE_VARIABLES.index(component) > position


def test_wealth_split_chain_and_definitions_are_preserved():
    """Appending #543 targets preserves #501's chain, draws and definitions."""
    wealth_split_targets = (
        "owned_land",
        "property_wealth",
        "private_pension_wealth",
        "directly_held_shares",
        "unit_and_investment_trusts",
        "stocks_and_shares_isa",
        "gross_financial_wealth",
        "net_financial_wealth",
        "main_residence_value",
        "other_residential_property_value",
        "non_residential_property_value",
        "savings",
        "num_vehicles",
        "student_loan_balance",
        "cash_isa",
        "other_residential_property_secured_debt",
        "non_residential_property_secured_debt",
        "owned_land_secured_debt",
    )
    assert tuple(IMPUTE_VARIABLES) == (*wealth_split_targets, *NEW_COMPONENTS)
    assert {name: QUANTILE_DRAW_SEEDS[name] for name in wealth_split_targets} == {
        name: 1_000 + index for index, name in enumerate(wealth_split_targets)
    }
    assert DERIVED_TARGETS["other_residential_property_value"] == (
        "DVHseValR8_sum",
        "DVBltValR8_sum",
    )
    assert DERIVED_TARGETS["other_residential_property_secured_debt"] == (
        "DVHseDebtR8_sum",
        "DVBLtDebtR8_sum",
    )


def test_impute_wealth_does_not_save_property_wealth(monkeypatch):
    """The imputation step drops the WAS total and keeps the components."""
    from policyengine_uk_data.datasets.imputations import wealth

    predicted = {
        **{name: [0.0, 0.0] for name in IMPUTE_VARIABLES},
        "property_wealth": [250_000.0, 0.0],
        **{name: [10_000.0 * (i + 1), 0.0] for i, name in enumerate(COMPONENTS)},
    }

    class DummyModel:
        input_columns = ["region"]

        @staticmethod
        def predict(input_df):
            return pd.DataFrame(predicted, index=input_df.index)

    class DummyMicrosimulation:
        def __init__(self, dataset):
            self.regions = dataset.household["region"]

        def calculate_dataframe(self, predictors, map_to):
            return pd.DataFrame({"region": self.regions.values})

    monkeypatch.setattr(wealth, "create_wealth_model", lambda: DummyModel())
    monkeypatch.setattr(wealth, "Microsimulation", DummyMicrosimulation)
    monkeypatch.setattr(wealth, "require_property_component_model", lambda: None)
    household = pd.DataFrame(
        {"household_weight": [1.0, 1.0], "region": ["LONDON", "WALES"]}
    )
    imputed = wealth.impute_wealth(_synthetic_dataset(household))

    assert "property_wealth" not in imputed.household.columns
    for component in COMPONENTS:
        assert imputed.household[component].tolist() == predicted[component]


def test_storage_removes_an_inherited_property_total():
    dataset = _synthetic_dataset(pd.DataFrame({"property_wealth": [999_999.0]}))
    predictions = pd.DataFrame(
        {**{name: [100.0] for name in COMPONENTS}, "property_wealth": [7.0]}
    )
    stored = store_wealth_predictions(dataset, predictions)
    assert "property_wealth" not in stored.household
    assert set(COMPONENTS) <= set(stored.household)


@settings(max_examples=20, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    second_homes=st.integers(min_value=0, max_value=5_000_000),
    buy_to_let=st.integers(min_value=1, max_value=5_000_000),
    second_homes_debt=st.integers(min_value=0, max_value=5_000_000),
    buy_to_let_debt=st.integers(min_value=0, max_value=5_000_000),
)
def test_buy_to_let_is_counted_exactly_once_with_secured_debt_separate(
    second_homes, buy_to_let, second_homes_debt, buy_to_let_debt
):
    """For every draw, gross values merge once and debt remains a separate input.

    Neither residential source is exported separately. Stored and projected
    residential_property_value and property_wealth contain the merged gross
    value exactly once, including where debt exceeds the value of the asset.
    """
    raw = _source_table([[0, second_homes, buy_to_let, 0, 0, 0, 0]])
    raw.loc[0, "DVHseDebtR8_sum"] = second_homes_debt
    raw.loc[0, "DVBLtDebtR8_sum"] = buy_to_let_debt
    mapped = generate_was_table(raw)
    gross = second_homes + buy_to_let
    debt = second_homes_debt + buy_to_let_debt
    assert mapped["other_residential_property_value"].iloc[0] == gross
    assert mapped["other_residential_property_secured_debt"].iloc[0] == debt
    assert mapped[list(COMPONENTS)].sum(axis=1).iloc[0] == gross

    intermediates = {
        "second_homes_value",
        "buy_to_let_value",
        "buy_to_let_property_value",
    }
    assert not intermediates.intersection(IMPUTE_VARIABLES)
    assert not intermediates.intersection(PROPERTY_COMPONENTS)
    predictions = derive_wealth_outputs(mapped[IMPUTE_VARIABLES])
    stored = store_wealth_predictions(
        _synthetic_dataset(pd.DataFrame({"region": ["LONDON"]})), predictions
    )
    assert not intermediates.intersection(stored.household.columns)
    assert "property_wealth" not in stored.household.columns
    assert stored.household["other_residential_property_value"].iloc[0] == gross
    assert stored.household["other_residential_property_secured_debt"].iloc[0] == debt

    simulation = _contract_simulation(stored)
    np.testing.assert_allclose(
        _calculate(simulation, "property_wealth", BASE_YEAR),
        [gross],
        rtol=RTOL,
        atol=1.0,
    )
    for year in YEARS:
        residential = _calculate(simulation, "other_residential_property_value", year)
        np.testing.assert_allclose(
            _calculate(simulation, "residential_property_value", year),
            residential,
            rtol=RTOL,
            atol=1.0,
        )
        np.testing.assert_allclose(
            _calculate(simulation, "property_wealth", year),
            residential,
            rtol=RTOL,
            atol=1.0,
        )
        np.testing.assert_allclose(
            _calculate(simulation, "total_wealth", year),
            residential,
            rtol=RTOL,
            atol=1.0,
        )
    assert (
        _calculate(simulation, "other_residential_property_value", PROJECTION_YEAR)[0]
        > gross
    )


value = st.one_of(
    st.just(0.0),
    st.floats(min_value=1_000, max_value=5_000_000, allow_nan=False),
)
household_rows = st.lists(
    st.tuples(
        st.sampled_from(sorted(_load_regional_house_prices())),
        st.tuples(*([value] * len(COMPONENTS))),
        value,  # WAS property total, imputed independently of the components
    ),
    min_size=1,
    max_size=8,
)


@settings(
    max_examples=25,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(rows=household_rows)
def test_saved_property_wealth_follows_its_components(rows):
    """Invariants for any non-negative imputed components and any WAS total:

    - no `property_wealth` column is saved;
    - property_wealth equals the sum of its components in every year,
      including after uk-data materialises the calibration year;
    - property_wealth >= main_residence_value (P4);
    - property_wealth grows exactly as its components do.
    """
    regions, components, was_total = map(list, zip(*rows))
    n = len(rows)
    household = pd.DataFrame(
        {
            "household_weight": np.ones(n),
            "region": regions,
            "council_tax": np.zeros(n),
            "rent": np.zeros(n),
            "tenure_type": ["OWNED_OUTRIGHT"] * n,
        }
    )
    predictions = pd.DataFrame(
        {
            "property_wealth": was_total,
            **dict(zip(COMPONENTS, zip(*components))),
        }
    )
    dataset = store_wealth_predictions(_synthetic_dataset(household), predictions)
    dataset = uprate_property_by_region(dataset)
    assert "property_wealth" not in dataset.household.columns

    simulation = _contract_simulation(dataset)
    arrays = _property_arrays(simulation, YEARS)
    calibration_year = CURRENT_FRS_RELEASE.calibration_year
    materialised = uprate_dataset(dataset, calibration_year)
    # uprate_dataset stores an int year; policyengine-uk expects a string.
    materialised.time_period = str(materialised.time_period)
    arrays_materialised = _property_arrays(
        _contract_simulation(materialised), [calibration_year]
    )

    for a in [*arrays.values(), *arrays_materialised.values()]:
        np.testing.assert_allclose(
            a["property_wealth"], a["components"], rtol=RTOL, atol=1.0
        )
        assert np.all(a["property_wealth"] >= a["main_residence_value"])

    base, later = arrays[BASE_YEAR], arrays[PROJECTION_YEAR]
    owns = base["components"] > 0
    np.testing.assert_allclose(
        later["property_wealth"][owns] / base["property_wealth"][owns],
        later["components"][owns] / base["components"][owns],
        rtol=RTOL,
    )
    for component in COMPONENTS:
        base_component = _calculate(simulation, component, BASE_YEAR)
        projected_component = _calculate(simulation, component, PROJECTION_YEAR)
        owners = base_component > 0
        assert np.all(projected_component[owners] > base_component[owners])


@settings(max_examples=15, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(amount=st.floats(min_value=1_000, max_value=5_000_000, allow_nan=False))
def test_uk_land_counted_once_and_overseas_excluded_from_uk_land(amount):
    household = pd.DataFrame(
        {
            "region": ["LONDON", "LONDON"],
            "owned_land": [amount, 0.0],
            "overseas_property_value": [0.0, amount],
            **{
                name: [0.0, 0.0]
                for name in set(COMPONENTS) - {"owned_land", "overseas_property_value"}
            },
        }
    )
    sim = _contract_simulation(_synthetic_dataset(household))
    for year in YEARS:
        land = _calculate(sim, "owned_land", year)
        overseas = _calculate(sim, "overseas_property_value", year)
        np.testing.assert_allclose(
            _calculate(sim, "property_wealth", year), land + overseas
        )
        np.testing.assert_allclose(
            _calculate(sim, "total_wealth", year), land + overseas
        )
        np.testing.assert_allclose(
            _calculate(sim, "household_land_value", year), land, rtol=RTOL, atol=1.0
        )


def test_mixed_property_preserves_worldwide_wealth_and_domestic_land_accounting():
    source_values = [
        100_000.0,
        20_000.0,
        60_000.0,
        40_000.0,
        10_000.0,
        30_000.0,
        70_000.0,
    ]
    mapped = generate_was_table(_source_table([source_values]))
    dataset = store_wealth_predictions(
        _synthetic_dataset(pd.DataFrame({"region": ["LONDON"]})),
        mapped[IMPUTE_VARIABLES],
    )
    sim = _contract_simulation(dataset)
    np.testing.assert_allclose(
        _calculate(sim, "property_wealth", BASE_YEAR), mapped["property_wealth"]
    )
    for year in YEARS:
        values = {name: _calculate(sim, name, year) for name in COMPONENTS}
        domestic = sum(
            values[name]
            for name in COMPONENTS
            if name not in {"owned_land", "overseas_property_value"}
        )
        worldwide = sum(values.values())
        intensity = sim.tax_benefit_system.parameters(
            str(year)
        ).household.wealth.land.intensity.property_wealth_by_region.LONDON
        np.testing.assert_allclose(
            _calculate(sim, "property_wealth", year), worldwide, rtol=RTOL
        )
        np.testing.assert_allclose(
            _calculate(sim, "total_wealth", year), worldwide, rtol=RTOL
        )
        np.testing.assert_allclose(
            _calculate(sim, "household_land_value", year),
            domestic * intensity + values["owned_land"],
            rtol=RTOL,
        )


@pytest.fixture(scope="module")
def built_property_arrays():
    try:
        dataset = UKSingleYearDataset(
            STORAGE_FOLDER / CURRENT_FRS_RELEASE.enhanced_dataset_file
        )
    except FileNotFoundError:
        pytest.skip("Enhanced FRS dataset not available")
    stored = "property_wealth" in dataset.household.columns
    return stored, _property_arrays(Microsimulation(dataset=dataset), YEARS)


def test_built_dataset_does_not_save_property_wealth(built_property_arrays):
    stored, _ = built_property_arrays
    assert not stored, "the enhanced FRS saves a property_wealth column"


@pytest.mark.parametrize("year", YEARS, ids=map(str, YEARS))
def test_built_property_wealth_is_the_sum_of_its_components(
    built_property_arrays, year
):
    _, arrays = built_property_arrays
    a = arrays[year]
    n_bad, total_gap = _n_not_close(a["property_wealth"], a["components"])
    assert n_bad == 0, (
        f"{year}: property_wealth differs from the sum of its components for "
        f"{n_bad} of {len(a['components'])} households (unweighted absolute "
        f"gap £{total_gap / 1e9:,.1f}bn)"
    )


@pytest.mark.parametrize("year", YEARS, ids=map(str, YEARS))
def test_built_property_wealth_at_least_main_residence(built_property_arrays, year):
    """P4: property_wealth >= main_residence_value for every household."""
    _, arrays = built_property_arrays
    a = arrays[year]
    below = a["property_wealth"] < a["main_residence_value"]
    owners = a["main_residence_value"] > 0
    assert not below.any(), (
        f"{year}: property_wealth < main_residence_value for {int(below.sum())} "
        f"households ({int((below & owners).sum())} of {int(owners.sum())} owners)"
    )


def test_built_property_wealth_grows_with_its_components(built_property_arrays):
    """Per household, property_wealth(projection) / property_wealth(base)
    equals the same ratio for the sum of its components."""
    _, arrays = built_property_arrays
    base, later = arrays[BASE_YEAR], arrays[PROJECTION_YEAR]
    owns = base["components"] > 0
    assert owns.any()
    with np.errstate(divide="ignore", invalid="ignore"):
        growth = later["property_wealth"][owns] / base["property_wealth"][owns]
    component_growth = later["components"][owns] / base["components"][owns]
    bad = ~(np.abs(growth - component_growth) <= RTOL * component_growth)
    assert not bad.any(), (
        f"property_wealth growth {BASE_YEAR}-{PROJECTION_YEAR} differs from "
        f"component growth for {int(bad.sum())} of {int(owns.sum())} households "
        f"with property (mean property_wealth growth "
        f"{np.nanmean(growth):.4f}, mean component growth "
        f"{np.mean(component_growth):.4f})"
    )
