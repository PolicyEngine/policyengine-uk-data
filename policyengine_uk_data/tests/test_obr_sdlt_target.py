"""The OBR SDLT target leaves out the corporate SDLT policyengine-uk adds (#564).

The OBR's SDLT line covers every buyer. policyengine-uk adds SDLT paid by
non-natural persons as `corporate_sdlt` (its corporate statistic times each
household's shareholding), so household `stamp_duty_land_tax` is targeted at
the OBR line less that statistic.

Invariants, for every year the OBR forecasts:
- target + corporate statistic = OBR SDLT;
- 0 <= target <= OBR SDLT whenever 0 <= statistic <= OBR SDLT, and a statistic
  above the OBR figure is an error rather than a negative target;
- the statistic subtracted is the one `corporate_sdlt` adds;
- the projection check reads each statistic's uprating index, and on the
  installed policyengine-uk finds both uprated from 2.152.0 (#2322) and
  neither before.
"""

import importlib.metadata
import itertools
import logging
from contextlib import ExitStack, contextmanager
from unittest.mock import patch

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from packaging.version import Version

from policyengine_uk_data.targets.obr_sources import (
    SourceError,
    active_manifest,
    load_workbook,
    read_row,
)
from policyengine_uk_data.targets.sources import obr

YEARS = active_manifest().years


@pytest.fixture(scope="module")
def receipts_workbook():
    """The verified EFO receipts workbook the active manifest names."""
    return load_workbook("receipts")


def _sheet(wb, title: str):
    (name,) = [n for n in wb.sheetnames if title in str(wb[n]["B2"].value)]
    return wb[name]


def _row(ws, label: str) -> dict[int, float]:
    """A row's values by year, with years from the table's own header."""
    return read_row(ws, obr._find_row(ws, label), active_manifest())


@contextmanager
def _stub_statistic(corporate: dict[int, float], unprojected: tuple = ()):
    """Stand in for policyengine-uk's statistic and its projection check."""
    with (
        patch.object(
            obr,
            "corporate_sdlt_receipts",
            lambda years=None: {year: corporate[year] for year in years},
        ),
        patch.object(obr, "unprojected_corporate_sdlt", lambda: list(unprojected)),
    ):
        yield


def test_target_is_obr_sdlt_less_corporate_statistic(receipts_workbook):
    obr_sdlt = _row(obr._find_receipts_sheet(receipts_workbook), "Stamp duty land tax")
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
    total = _row(by_sector, "Total SDLT")
    assert sorted(total) == YEARS
    assert total == pytest.approx(
        _row(obr._find_receipts_sheet(receipts_workbook), "Stamp duty land tax")
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
# Generation can pass Hypothesis's time limit after policyengine-uk's imports
# on a loaded machine; the properties do not depend on timing.
relaxed = settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])


@relaxed
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


@relaxed
@given(
    st.dictionaries(st.integers(2000, 2040), _amounts, min_size=1),
    st.floats(min_value=1, max_value=1e9),
    st.data(),
)
def test_statistic_above_obr_sdlt_is_an_error(obr_sdlt, excess, data):
    year = data.draw(st.sampled_from(sorted(obr_sdlt)))
    corporate = {**dict.fromkeys(obr_sdlt, 0.0), year: obr_sdlt[year] + excess}
    with _stub_statistic(corporate), pytest.raises(SourceError, match=str(year)):
        obr.household_sdlt(obr_sdlt)


def test_statistic_above_obr_sdlt_stops_the_receipts_parse(receipts_workbook):
    """The error is not caught as a missing row and the target dropped."""
    with (
        _stub_statistic(dict.fromkeys(YEARS, 1e12)),
        pytest.raises(SourceError, match="exceeds the OBR's SDLT"),
    ):
        obr._parse_receipts(receipts_workbook)


def test_unprojected_statistic_warns(caplog):
    flat = dict.fromkeys(YEARS, 3.13e9)
    unprojected = ("gov.hmrc.stamp_duty.statistics.residential.corporate.revenue",)
    with _stub_statistic(flat, unprojected), caplog.at_level(logging.WARNING):
        obr.household_sdlt(dict.fromkeys(YEARS, 20e9))
    assert "PolicyEngine/policyengine-uk#2322" in caplog.text
    assert unprojected[0] in caplog.text


def test_projected_statistic_does_not_warn(caplog):
    """Even when equal in every year, as a one-year manifest's always is."""
    with _stub_statistic(dict.fromkeys(YEARS, 4e9)), caplog.at_level(logging.WARNING):
        obr.household_sdlt(dict.fromkeys(YEARS, 20e9))
    assert caplog.text == ""


@pytest.mark.parametrize("uprated", itertools.product([False, True], repeat=2))
def test_projection_check_reads_each_statistics_uprating_index(uprated):
    """Whatever the installed version, a statistic counts as projected exactly
    when its parameter names an uprating index."""
    from policyengine_uk.system import system

    statistics = system.parameters.gov.hmrc.stamp_duty.statistics
    expected = []
    with ExitStack() as stack:
        for kind, on in zip(("residential", "non_residential"), uprated):
            metadata = getattr(statistics, kind).corporate.revenue.metadata
            stack.enter_context(patch.dict(metadata))
            metadata.pop("uprating", None)
            if on:
                metadata["uprating"] = (
                    f"gov.economic_assumptions.indices.obr.stamp_duty_land_tax.{kind}"
                )
            else:
                expected.append(
                    f"gov.hmrc.stamp_duty.statistics.{kind}.corporate.revenue"
                )
        assert obr.unprojected_corporate_sdlt() == expected


def test_projection_check_on_the_installed_policyengine_uk():
    """policyengine-uk 2.152.0 and later (#2322) uprate both statistics. The
    earlier versions uk-data can lock uprate neither and hold the 2020-21
    figures, £3.13bn in all, in every later year."""
    version = Version(importlib.metadata.version("policyengine-uk"))
    unprojected = obr.unprojected_corporate_sdlt()
    if version >= Version("2.152.0"):
        assert unprojected == []
    else:
        assert unprojected == [
            f"gov.hmrc.stamp_duty.statistics.{kind}.corporate.revenue"
            for kind in ("residential", "non_residential")
        ]
        for value in obr.corporate_sdlt_receipts().values():
            assert value == pytest.approx(3.13e9)
