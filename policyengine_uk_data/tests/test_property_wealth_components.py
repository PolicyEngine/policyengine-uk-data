"""property_wealth must come from its uprated components, not a saved column.

policyengine-uk defines `property_wealth` as main_residence_value +
other_residential_property_value + non_residential_property_value. A saved
`property_wealth` column overrides that sum and, having no uprating index,
stays at its dataset-year value while the components are uprated.

The synthetic tests need no private data. The built-dataset tests run after
`make data` and are skipped when the enhanced FRS is absent. Failure messages
report counts and maxima only, never individual records.
"""

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from policyengine_uk import Microsimulation
from policyengine_uk.data import UKSingleYearDataset

from policyengine_uk_data.datasets.frs_release import CURRENT_FRS_RELEASE
from policyengine_uk_data.datasets.imputations.regional_property_uprating import (
    _load_regional_house_prices,
    uprate_property_by_region,
)
from policyengine_uk_data.datasets.imputations.wealth import (
    CONDITIONING_ONLY_VARIABLES,
    IMPUTE_VARIABLES,
    store_wealth_predictions,
)
from policyengine_uk_data.storage import STORAGE_FOLDER
from policyengine_uk_data.utils.uprating import uprate_dataset

COMPONENTS = (
    "main_residence_value",
    "other_residential_property_value",
    "non_residential_property_value",
)
BASE_YEAR = CURRENT_FRS_RELEASE.base_year
# Two years past the base year: 2026 for the 2024-25 FRS.
PROJECTION_YEAR = BASE_YEAR + 2
YEARS = sorted({BASE_YEAR, CURRENT_FRS_RELEASE.calibration_year, PROJECTION_YEAR})
RTOL = 1e-5


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


def test_property_wealth_is_imputed_before_its_components():
    """The components are predicted conditional on the WAS property total."""
    assert set(CONDITIONING_ONLY_VARIABLES) <= set(IMPUTE_VARIABLES)
    position = IMPUTE_VARIABLES.index("property_wealth")
    assert "property_wealth" in CONDITIONING_ONLY_VARIABLES
    for component in COMPONENTS:
        assert IMPUTE_VARIABLES.index(component) > position


def test_impute_wealth_does_not_save_property_wealth(monkeypatch):
    """The imputation step drops the WAS total and keeps the components."""
    from policyengine_uk_data.datasets.imputations import wealth

    predicted = {
        "property_wealth": [250_000.0, 0.0],
        "main_residence_value": [200_000.0, 0.0],
        "other_residential_property_value": [90_000.0, 0.0],
        "non_residential_property_value": [0.0, 0.0],
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
    household = pd.DataFrame(
        {"household_weight": [1.0, 1.0], "region": ["LONDON", "WALES"]}
    )
    imputed = wealth.impute_wealth(_synthetic_dataset(household))

    assert "property_wealth" not in imputed.household.columns
    for component in COMPONENTS:
        assert imputed.household[component].tolist() == predicted[component]


value = st.one_of(
    st.just(0.0),
    st.floats(min_value=1_000, max_value=5_000_000, allow_nan=False),
)
household_rows = st.lists(
    st.tuples(
        st.sampled_from(sorted(_load_regional_house_prices())),
        value,  # main residence
        value,  # other residential property
        value,  # non-residential property
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
    regions, main, other, non_res, was_total = map(list, zip(*rows))
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
            "main_residence_value": main,
            "other_residential_property_value": other,
            "non_residential_property_value": non_res,
        }
    )
    dataset = store_wealth_predictions(_synthetic_dataset(household), predictions)
    dataset = uprate_property_by_region(dataset)
    assert "property_wealth" not in dataset.household.columns

    arrays = _property_arrays(Microsimulation(dataset=dataset), YEARS)
    calibration_year = CURRENT_FRS_RELEASE.calibration_year
    materialised = uprate_dataset(dataset, calibration_year)
    # uprate_dataset stores an int year; policyengine-uk expects a string.
    materialised.time_period = str(materialised.time_period)
    arrays_materialised = _property_arrays(
        Microsimulation(dataset=materialised), [calibration_year]
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
