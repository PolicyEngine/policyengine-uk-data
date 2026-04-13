import numpy as np
import pandas as pd

from policyengine_uk_data.datasets.local_areas.constituencies.devolved_housing import (
    _PRIVATE_RENT_TARGETS,
    add_private_rent_targets,
)
from policyengine_uk_data.datasets.local_areas.constituencies import (
    loss as constituency_loss,
)


def _age_targets():
    return pd.DataFrame(
        {
            "code": ["W07000041", "W07000042", "S14000001", "S14000002"],
            "name": ["W1", "W2", "S1", "S2"],
            "age/0_10": [100, 300, 200, 200],
            "age/10_20": [100, 300, 200, 200],
        }
    )


def test_add_private_rent_targets_filters_matrix_to_country_private_renters():
    matrix = pd.DataFrame()
    y = pd.DataFrame()

    add_private_rent_targets(
        matrix,
        y,
        _age_targets(),
        country=np.array(["WALES", "WALES", "SCOTLAND", "ENGLAND"]),
        tenure_type=np.array(
            ["RENT_PRIVATELY", "OWNED_OUTRIGHT", "RENT_PRIVATELY", "RENT_PRIVATELY"]
        ),
        rent=np.array([9_600.0, 0.0, 12_000.0, 15_000.0]),
    )

    np.testing.assert_array_equal(
        matrix["housing/wales_private_renter_households"].values,
        np.array([1.0, 0.0, 0.0, 0.0]),
    )
    np.testing.assert_array_equal(
        matrix["housing/scotland_private_renter_households"].values,
        np.array([0.0, 0.0, 1.0, 0.0]),
    )
    np.testing.assert_array_equal(
        matrix["housing/wales_private_rent_amount"].values,
        np.array([9_600.0, 0.0, 0.0, 0.0]),
    )
    np.testing.assert_array_equal(
        matrix["housing/scotland_private_rent_amount"].values,
        np.array([0.0, 0.0, 12_000.0, 0.0]),
    )


def test_add_private_rent_targets_allocate_country_totals_by_population_share():
    matrix = pd.DataFrame()
    y = pd.DataFrame()

    add_private_rent_targets(
        matrix,
        y,
        _age_targets(),
        country=np.array(["WALES", "SCOTLAND"]),
        tenure_type=np.array(["RENT_PRIVATELY", "RENT_PRIVATELY"]),
        rent=np.array([9_600.0, 12_000.0]),
    )

    wales_shares = np.array([0.25, 0.75, 0.0, 0.0])
    scotland_shares = np.array([0.0, 0.0, 0.5, 0.5])

    np.testing.assert_allclose(
        y["housing/wales_private_renter_households"].values,
        wales_shares * _PRIVATE_RENT_TARGETS["WALES"]["private_renter_households"],
    )
    np.testing.assert_allclose(
        y["housing/scotland_private_renter_households"].values,
        scotland_shares
        * _PRIVATE_RENT_TARGETS["SCOTLAND"]["private_renter_households"],
    )
    np.testing.assert_allclose(
        y["housing/wales_private_rent_amount"].values.sum(),
        _PRIVATE_RENT_TARGETS["WALES"]["private_renter_households"]
        * _PRIVATE_RENT_TARGETS["WALES"]["annual_private_rent"],
    )
    np.testing.assert_allclose(
        y["housing/scotland_private_rent_amount"].values.sum(),
        _PRIVATE_RENT_TARGETS["SCOTLAND"]["private_renter_households"]
        * _PRIVATE_RENT_TARGETS["SCOTLAND"]["annual_private_rent"],
    )


class _FakeDataset:
    time_period = 2025


class _FakeSim:
    def __init__(self, *args, **kwargs):
        self.default_calculation_period = 2025

    def calculate(self, variable):
        mapping = {
            "self_employment_income": np.array([0.0, 0.0]),
            "employment_income": np.array([0.0, 0.0]),
            "income_tax": np.array([1.0, 1.0]),
            "age": np.array([35, 35]),
            "universal_credit": np.array([1.0, 1.0]),
            "is_child": np.array([0.0, 0.0]),
            "country": np.array(["WALES", "SCOTLAND"]),
            "tenure_type": np.array(["RENT_PRIVATELY", "RENT_PRIVATELY"]),
            "rent": np.array([9_600.0, 12_000.0]),
        }
        return type("Result", (), {"values": mapping[variable]})()

    def map_result(self, values, source_entity, target_entity):
        return np.asarray(values)


def test_constituency_target_matrix_includes_devolved_housing_targets(monkeypatch):
    age_targets = _age_targets().iloc[[0, 2]].reset_index(drop=True)
    income_targets = pd.DataFrame(
        {
            "self_employment_income_amount": [1.0, 1.0],
            "self_employment_income_count": [1.0, 1.0],
            "employment_income_amount": [1.0, 1.0],
            "employment_income_count": [1.0, 1.0],
        }
    )
    national_income = pd.DataFrame(
        {
            "total_income_lower_bound": [12_570],
            "total_income_upper_bound": [np.inf],
            "self_employment_income_amount": [1.0],
            "employment_income_amount": [1.0],
        }
    )
    uc_by_children = pd.DataFrame(
        {
            "uc_hh_0_children": [1.0, 1.0],
            "uc_hh_1_child": [0.0, 0.0],
            "uc_hh_2_children": [0.0, 0.0],
            "uc_hh_3plus_children": [0.0, 0.0],
        }
    )

    monkeypatch.setattr(constituency_loss, "Microsimulation", _FakeSim)
    monkeypatch.setattr(
        constituency_loss, "get_constituency_income_targets", lambda: income_targets
    )
    monkeypatch.setattr(
        constituency_loss,
        "get_national_income_projections",
        lambda year: national_income,
    )
    monkeypatch.setattr(
        constituency_loss, "get_constituency_age_targets", lambda: age_targets
    )
    monkeypatch.setattr(constituency_loss, "get_uk_total_population", lambda year: 2.0)
    monkeypatch.setattr(
        constituency_loss,
        "get_constituency_uc_targets",
        lambda: pd.Series([1.0, 1.0]),
    )
    monkeypatch.setattr(
        constituency_loss,
        "get_constituency_uc_by_children_targets",
        lambda: uc_by_children,
    )
    monkeypatch.setattr(constituency_loss, "mapping_matrix", np.eye(2))
    monkeypatch.setattr(
        constituency_loss.pd,
        "read_csv",
        lambda path: pd.DataFrame({"code": ["W07000041", "S14000001"]}),
    )

    matrix, y, country_mask = constituency_loss.create_constituency_target_matrix(
        _FakeDataset()
    )

    assert "housing/wales_private_renter_households" in matrix.columns
    assert "housing/scotland_private_rent_amount" in matrix.columns
    np.testing.assert_allclose(
        y["housing/wales_private_renter_households"].values,
        np.array([_PRIVATE_RENT_TARGETS["WALES"]["private_renter_households"], 0.0]),
    )
    np.testing.assert_allclose(
        y["housing/scotland_private_renter_households"].values,
        np.array([0.0, _PRIVATE_RENT_TARGETS["SCOTLAND"]["private_renter_households"]]),
    )
    np.testing.assert_array_equal(
        country_mask,
        np.array([[1.0, 0.0], [0.0, 1.0]]),
    )
