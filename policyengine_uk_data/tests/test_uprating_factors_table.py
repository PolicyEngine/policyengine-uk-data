"""uprating_factors.csv follows policyengine-uk's load-time uprating.

``uprate_dataset`` moves the build between the FRS base year and the
calibration year with this table, and policyengine-uk moves the saved file
to later years with ``extend_single_year_dataset``. Calibrated values are the
values the model runs on only where the two agree.

Invariants, checked for every row and every year (or pair of years):
- the committed table is what the generator builds from the locked
  policyengine-uk, and its rows are exactly the variables policyengine-uk
  uprates at load;
- outside ``OVERRIDDEN_VARIABLES`` each row grows each year by one plus the
  growth parameter policyengine-uk applies to it, and agrees with
  policyengine-uk's own ``extend_single_year_dataset`` (differential), as
  does ``uprate_dataset`` from the FRS base year;
- the fuel rows divide by the household-weight override, so fuel spending
  times household weight stays on policyengine-uk's path;
- every row is a level index (1 in ``START_YEAR``, finite, above 0.5), so
  ``uprate_dataset`` there and back between any two years is the identity;
- every variable a target projection uprates has a row.

test_income_projection checks that incomes_projection.csv is the committed
SPI table projected with this table.
"""

import numpy as np
import pandas as pd
import pytest
from policyengine_uk.data import UKSingleYearDataset

from policyengine_uk_data.datasets.frs_release import CURRENT_FRS_RELEASE
from policyengine_uk_data.storage import STORAGE_FOLDER
from policyengine_uk_data.utils import uprating
from policyengine_uk_data.utils.uprating import (
    END_YEAR,
    OVERRIDDEN_VARIABLES,
    START_YEAR,
    UPRATING_TABLE_DECIMALS,
    VOLUME_OVERRIDDEN_VARIABLES,
    build_uprating_factors_table,
    policyengine_uk_load_time_index,
    policyengine_uk_uprating_indices,
    uprate_dataset,
)

TABLE = pd.read_csv(STORAGE_FOLDER / "uprating_factors.csv", index_col="Variable")
TABLE.columns = TABLE.columns.astype(int)
YEARS = list(range(START_YEAR, END_YEAR + 1))
ENGINE_ROWS = TABLE.index.difference(OVERRIDDEN_VARIABLES)
# Each stored level is within half a unit in the last stored place of the
# unrounded level, and every level exceeds 0.5, so a ratio of two stored
# levels is within 2 units in that place, relative, of the unrounded ratio.
ROUNDING = 2 * 10**-UPRATING_TABLE_DECIMALS
# The engine's own uprating moves council tax by country and rent by region
# and tenure. Neither is a single index, so neither is in the table and
# `uprate_dataset` leaves both unchanged.
NOT_SINGLE_INDICES = ("council_tax", "rent")
# Dataset columns that declare an `uprating` attribute policyengine-uk does not
# apply at load (policyengine-uk#1862): both paths carry them unchanged.
CARRIED_UNCHANGED = (
    "housing_service_charges",
    "pension_contributions_via_salary_sacrifice",
    "rail_usage",
    "water_and_sewerage_charges",
)


def _unit_dataset(variables, year: int) -> UKSingleYearDataset:
    """One person, benefit unit and household with every variable at 1."""
    from policyengine_uk.system import system

    tables = {
        "person": {
            "person_id": [1],
            "person_benunit_id": [1],
            "person_household_id": [1],
        },
        "benunit": {"benunit_id": [1]},
        "household": {
            "household_id": [1],
            "region": ["LONDON"],
            "tenure_type": ["RENT_PRIVATELY"],
            **{variable: [1.0] for variable in NOT_SINGLE_INDICES},
        },
    }
    for variable in variables:
        tables[system.variables[variable].entity.key][variable] = [1.0]
    return UKSingleYearDataset(
        **{name: pd.DataFrame(columns) for name, columns in tables.items()},
        fiscal_year=year,
    )


def _value(dataset: UKSingleYearDataset, variable: str) -> float:
    (column,) = [table[variable] for table in dataset.tables if variable in table]
    return float(column.iloc[0])


def _project_with_policyengine_uk(dataset, end_year: int) -> dict:
    from policyengine_uk.data.economic_assumptions import extend_single_year_dataset
    from policyengine_uk.system import system

    return extend_single_year_dataset(
        dataset, system.parameters, end_year=end_year
    ).datasets


def test_committed_table_is_what_the_generator_builds():
    """Fails when policyengine-uk's growth parameters or indices change.

    Regenerate with ``python -m policyengine_uk_data.utils.uprating``, then
    ``python -m policyengine_uk_data.utils.incomes_projection``.
    """
    pd.testing.assert_frame_equal(
        TABLE, build_uprating_factors_table(), check_names=False, rtol=0, atol=1e-12
    )


def test_rows_are_the_variables_policyengine_uk_uprates_at_load():
    engine = policyengine_uk_load_time_index()
    assert set(TABLE.index) == set(engine.index)
    assert set(OVERRIDDEN_VARIABLES) <= set(TABLE.index)


def test_each_row_grows_by_its_policyengine_uk_growth_parameter():
    from policyengine_uk.system import system

    for index_name, variables in policyengine_uk_uprating_indices().items():
        growth = system.parameters.get_child(index_name)
        for variable in set(variables).difference(OVERRIDDEN_VARIABLES):
            for year in YEARS[1:]:
                factor = TABLE.loc[variable, year] / TABLE.loc[variable, year - 1]
                assert factor == pytest.approx(1 + growth(str(year)), rel=ROUNDING), (
                    variable,
                    year,
                )


def test_table_agrees_with_policyengine_uks_own_load_time_uprating():
    """Differential: policyengine-uk projects a dataset of ones from START_YEAR."""
    projected = _project_with_policyengine_uk(
        _unit_dataset(TABLE.index, START_YEAR), END_YEAR
    )
    for year in YEARS:
        engine = projected[year]
        for variable in ENGINE_ROWS:
            assert TABLE.loc[variable, year] == pytest.approx(
                _value(engine, variable), rel=ROUNDING
            ), (variable, year)
        engine_weight = _value(engine, "household_weight")
        for variable in VOLUME_OVERRIDDEN_VARIABLES:
            weighted = TABLE.loc[variable, year] * TABLE.loc["household_weight", year]
            assert weighted == pytest.approx(
                _value(engine, variable) * engine_weight, rel=ROUNDING
            ), (variable, year)


@pytest.mark.parametrize("year", range(CURRENT_FRS_RELEASE.base_year, 2031))
def test_uprate_dataset_from_the_base_year_matches_policyengine_uk(year):
    """What calibration sees in ``year`` is what policyengine-uk runs on.

    policyengine-uk projects a dataset to 2030 at load, so later years follow
    each variable's own ``uprating`` attribute and are not compared.
    """
    base_year = CURRENT_FRS_RELEASE.base_year
    variables = list(TABLE.index) + list(CARRIED_UNCHANGED)
    dataset = _unit_dataset(variables, base_year)
    engine = _project_with_policyengine_uk(dataset, year)[year]
    calibration = uprate_dataset(dataset, year)
    for variable in ENGINE_ROWS.union(CARRIED_UNCHANGED):
        assert _value(calibration, variable) == pytest.approx(
            _value(engine, variable), rel=ROUNDING
        ), variable
    for variable in CARRIED_UNCHANGED:
        assert _value(calibration, variable) == 1.0, variable
    for variable in VOLUME_OVERRIDDEN_VARIABLES:
        assert _value(calibration, variable) * _value(
            calibration, "household_weight"
        ) == pytest.approx(
            _value(engine, variable) * _value(engine, "household_weight"),
            rel=ROUNDING,
        ), variable


def test_every_row_is_a_level_index():
    levels = TABLE.to_numpy()
    assert (TABLE[START_YEAR] == 1).all()
    assert np.isfinite(levels).all()
    assert (levels > 0.5).all()


@pytest.mark.parametrize("start", YEARS)
def test_uprate_dataset_there_and_back_is_the_identity(start):
    dataset = _unit_dataset(TABLE.index, start)
    for end in YEARS:
        there = uprate_dataset(dataset, end)
        back = uprate_dataset(there, start)
        for variable in TABLE.index:
            assert _value(there, variable) == pytest.approx(
                TABLE.loc[variable, end] / TABLE.loc[variable, start], rel=1e-15
            )
            assert _value(back, variable) == pytest.approx(1.0, rel=1e-12)


def test_every_variable_a_target_projection_uprates_has_a_row():
    """A missing row loses targets: the CGT projection stops at the first year
    it cannot uprate, behind a log warning, and the SPI income projection
    cannot be regenerated."""
    from policyengine_uk_data.utils.incomes_projection import ALL_INCOME_VARIABLES

    needed = set(ALL_INCOME_VARIABLES) | {"household_weight", "capital_gains"}
    assert needed <= set(TABLE.index), needed - set(TABLE.index)


def test_a_variable_listed_under_two_indices_raises(monkeypatch):
    monkeypatch.setattr(
        uprating,
        "policyengine_uk_uprating_indices",
        lambda: {
            "gov.economic_assumptions.yoy_growth.obr.consumer_price_index": ["x"],
            "gov.economic_assumptions.yoy_growth.obr.average_earnings": ["x"],
        },
    )
    with pytest.raises(ValueError, match="listed under two indices"):
        policyengine_uk_load_time_index()
