from types import SimpleNamespace

import numpy as np

from policyengine_uk_data.targets.compute.households import compute_tenure
from policyengine_uk_data.targets.compute.other import compute_housing
from policyengine_uk_data.targets.sources.devolved_housing import (
    _SCOTLAND_PRIVATE_RENT_TOTAL_2025,
    _WALES_PRIVATE_RENT_TOTAL_2025,
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
    assert "housing/rent_private/scotland" in names
    assert "housing/rent_private/wales" in names


def test_devolved_private_rent_values():
    targets = {t.name: t for t in get_targets()}
    assert (
        targets["housing/rent_private/scotland"].values[2025]
        == _SCOTLAND_PRIVATE_RENT_TOTAL_2025
    )
    assert (
        targets["housing/rent_private/wales"].values[2025]
        == _WALES_PRIVATE_RENT_TOTAL_2025
    )


def test_compute_housing_filters_to_country():
    ctx = _FakeCtx(
        tenure_type=["RENT_PRIVATELY", "RENT_PRIVATELY", "RENT_FROM_HA"],
        country=["SCOTLAND", "WALES", "SCOTLAND"],
        rent=[1000.0, 800.0, 600.0],
    )
    target = SimpleNamespace(name="housing/rent_private/scotland", geo_code="S")
    result = compute_housing(target, ctx)
    np.testing.assert_array_equal(result, np.array([1000.0, 0.0, 0.0]))


def test_compute_tenure_filters_to_country():
    ctx = _FakeCtx(
        tenure_type=["RENT_PRIVATELY", "RENT_PRIVATELY", "OWNED_OUTRIGHT"],
        country=["SCOTLAND", "WALES", "SCOTLAND"],
        rent=[0.0, 0.0, 0.0],
    )
    target = SimpleNamespace(name="gov_scot/tenure_scotland_rented_privately")
    result = compute_tenure(target, ctx)
    np.testing.assert_array_equal(result, np.array([1.0, 0.0, 0.0]))
