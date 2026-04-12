from types import SimpleNamespace

import numpy as np

from policyengine_uk_data.targets.compute.households import compute_tenure
from policyengine_uk_data.targets.sources.devolved_housing import (
    _COUNTRY_ANNUAL_PRIVATE_RENT,
    _compute_private_rent_average_gap,
    get_targets,
)


class _FakeSim:
    def __init__(self, tenure_type):
        self._tenure_type = np.array(tenure_type)

    def calculate(self, variable, map_to=None):
        if variable != "tenure_type" or map_to != "household":
            raise AssertionError(f"Unexpected calculate call: {variable}, {map_to}")
        return SimpleNamespace(values=self._tenure_type)


class _FakeCtx:
    def __init__(self, tenure_type, country, rent):
        self.sim = _FakeSim(tenure_type)
        self.country = np.array(country)
        self._rent = np.array(rent)

    def pe(self, variable):
        if variable != "rent":
            raise AssertionError(f"Unexpected pe call: {variable}")
        return self._rent


def test_devolved_targets_exist():
    names = {t.name for t in get_targets()}
    assert "gov_scot/tenure_scotland_rented_privately" in names
    assert "gov_wales/tenure_wales_rented_privately" in names
    assert "housing/private_rent_average_gap/scotland" in names
    assert "housing/private_rent_average_gap/wales" in names


def test_devolved_private_rent_gap_targets_zero():
    targets = {t.name: t for t in get_targets()}
    assert targets["housing/private_rent_average_gap/scotland"].values[2025] == 0
    assert targets["housing/private_rent_average_gap/wales"].values[2025] == 0


def test_compute_private_rent_average_gap_filters_to_country():
    ctx = _FakeCtx(
        tenure_type=["RENT_PRIVATELY", "RENT_PRIVATELY", "RENT_FROM_HA"],
        country=["SCOTLAND", "WALES", "SCOTLAND"],
        rent=[12_600.0, 9_600.0, 600.0],
    )
    target = SimpleNamespace(geo_code="S")
    result = _compute_private_rent_average_gap(ctx, target, 2025)
    expected_gap = 12_600.0 - _COUNTRY_ANNUAL_PRIVATE_RENT["S"]
    np.testing.assert_array_equal(result, np.array([expected_gap, 0.0, 0.0]))


def test_compute_tenure_filters_to_country():
    ctx = _FakeCtx(
        tenure_type=["RENT_PRIVATELY", "RENT_PRIVATELY", "OWNED_OUTRIGHT"],
        country=["SCOTLAND", "WALES", "SCOTLAND"],
        rent=[0.0, 0.0, 0.0],
    )
    target = SimpleNamespace(name="gov_scot/tenure_scotland_rented_privately")
    result = compute_tenure(target, ctx)
    np.testing.assert_array_equal(result, np.array([1.0, 0.0, 0.0]))
