"""The OBR SDLT target leaves out the corporate SDLT policyengine-uk adds (#564).

The OBR's SDLT line covers every buyer. policyengine-uk adds SDLT paid by
non-natural persons as `corporate_sdlt` (its corporate statistic times each
household's shareholding), so household `stamp_duty_land_tax` is targeted at
the OBR line less that statistic.

Invariants, for every year the OBR forecasts:
- target + corporate statistic = OBR SDLT;
- 0 <= target <= OBR SDLT whenever 0 <= statistic <= OBR SDLT, and a statistic
  above the OBR figure is an error rather than a negative target;
- the statistic subtracted is the one `corporate_sdlt` adds.
"""

import logging
from unittest.mock import patch

import pytest
from hypothesis import given
from hypothesis import strategies as st

from policyengine_uk_data.targets.sources import obr
from policyengine_uk_data.targets.sources._common import load_config

YEARS = sorted(obr._FY_COL_TO_YEAR.values())


@pytest.fixture(scope="module")
def receipts_workbook():
    """The committed EFO receipts workbook for the vintage in sources.yaml."""
    return obr._fallback_workbook(load_config()["obr"]["efo_receipts"])


def _sheet(wb, title: str):
    (name,) = [n for n in wb.sheetnames if title in str(wb[n]["B2"].value)]
    return wb[name]


def _row(ws, label: str, columns: str) -> dict[int, float]:
    row = obr._find_row(ws, label)
    return {year: ws[f"{col}{row}"].value * 1e9 for col, year in zip(columns, YEARS)}


def _stub_statistic(corporate: dict[int, float]):
    return patch.object(
        obr,
        "corporate_sdlt_receipts",
        lambda years=None: {year: corporate[year] for year in years},
    )


def test_target_is_obr_sdlt_less_corporate_statistic(receipts_workbook):
    obr_sdlt = _row(
        obr._find_receipts_sheet(receipts_workbook), "Stamp duty land tax", "DEFGHIJ"
    )
    target = {t.name: t for t in obr._parse_receipts(receipts_workbook)}["obr/sdlt"]
    corporate = obr.corporate_sdlt_receipts(YEARS)

    assert target.variable == "stamp_duty_land_tax"
    assert sorted(target.values) == YEARS
    for year in YEARS:
        assert target.values[year] == pytest.approx(
            obr_sdlt[year] - corporate[year], rel=1e-12
        )


def test_receipts_sdlt_line_is_the_property_type_total(receipts_workbook):
    """The line the statistic is subtracted from is the total of the property
    types policyengine-uk projects corporate SDLT with (EFO Table 3.7)."""
    by_sector = _sheet(receipts_workbook, "Property transactions taxes")
    assert _row(by_sector, "Total SDLT", "CDEFGHI") == pytest.approx(
        _row(
            obr._find_receipts_sheet(receipts_workbook),
            "Stamp duty land tax",
            "DEFGHIJ",
        )
    )


def test_statistic_is_what_corporate_sdlt_adds():
    """A household holding the whole corporate sector pays the statistic."""
    from policyengine_uk import Simulation

    simulation = Simulation(
        situation={
            "people": {"adult": {"age": dict.fromkeys(YEARS, 40)}},
            "benunits": {"benunit": {"members": ["adult"]}},
            "households": {
                "household": {
                    "members": ["adult"],
                    "shareholding": dict.fromkeys(YEARS, 1.0),
                }
            },
        }
    )
    statistic = obr.corporate_sdlt_receipts(YEARS)
    for year in YEARS:
        assert float(simulation.calculate("corporate_sdlt", year)[0]) == (
            pytest.approx(statistic[year], rel=1e-6)
        )


_amounts = st.floats(min_value=0, max_value=1e12, allow_nan=False)


@given(st.dictionaries(st.integers(2000, 2040), st.tuples(_amounts, _amounts)))
def test_household_sdlt_is_obr_sdlt_less_statistic(pairs):
    obr_sdlt = {year: max(pair) for year, pair in pairs.items()}
    corporate = {year: min(pair) for year, pair in pairs.items()}
    with _stub_statistic(corporate):
        household = obr.household_sdlt(obr_sdlt)

    assert household.keys() == obr_sdlt.keys()
    for year, total in obr_sdlt.items():
        assert 0 <= household[year] <= total
        assert household[year] + corporate[year] == pytest.approx(total, abs=1e-3)


@given(
    st.dictionaries(st.integers(2000, 2040), _amounts, min_size=1),
    st.floats(min_value=1, max_value=1e9),
    st.data(),
)
def test_statistic_above_obr_sdlt_is_an_error(obr_sdlt, excess, data):
    year = data.draw(st.sampled_from(sorted(obr_sdlt)))
    corporate = {**dict.fromkeys(obr_sdlt, 0.0), year: obr_sdlt[year] + excess}
    with _stub_statistic(corporate), pytest.raises(ValueError, match=str(year)):
        obr.household_sdlt(obr_sdlt)


def test_flat_statistic_warns(caplog):
    flat = dict.fromkeys(YEARS, 3.13e9)
    with _stub_statistic(flat), caplog.at_level(logging.WARNING):
        obr.household_sdlt(dict.fromkeys(YEARS, 20e9))
    assert "PolicyEngine/policyengine-uk#2322" in caplog.text


def test_projected_statistic_does_not_warn(caplog):
    projected = {year: 4e9 + 1e8 * (year - YEARS[0]) for year in YEARS}
    with _stub_statistic(projected), caplog.at_level(logging.WARNING):
        obr.household_sdlt(dict.fromkeys(YEARS, 20e9))
    assert caplog.text == ""
