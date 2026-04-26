from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from policyengine_uk import CountryTaxBenefitSystem
from policyengine_uk import Microsimulation

from policyengine_uk_data.datasets import (
    ENHANCED_CPS_SOURCE_FILE,
    create_enhanced_cps,
)
from policyengine_uk_data.datasets.enhanced_cps import _assign_council_tax_bands
from policyengine_uk_data.utils import reweight as reweight_module
from policyengine_uk_data.utils.loss import get_loss_results

ALLOWED_REPORTED_DATA_INPUTS = {
    # PE-UK uses this reported base field to derive basic/additional/new
    # state pension; it carries a formula only for year-to-year uprating.
    "state_pension_reported",
}


def _subset_source(tmp_path: Path, rows: int) -> Path:
    source = pd.read_csv(ENHANCED_CPS_SOURCE_FILE).head(rows).copy()
    subset_path = tmp_path / f"enhanced_cps_source_{rows}.csv"
    source.to_csv(subset_path, index=False)
    return subset_path


def test_policybench_transfer_dataset_validates(tmp_path: Path):
    dataset = create_enhanced_cps(
        source_file_path=_subset_source(tmp_path, 10),
        calibrate=False,
    )

    dataset.validate()

    assert len(dataset.household) == 10
    assert len(dataset.benunit) == 10
    assert len(dataset.person) >= 10
    assert (dataset.household.household_weight >= 0).all()
    assert dataset.household.household_weight.sum() > 0


def test_policybench_transfer_writes_only_valid_leaf_inputs(tmp_path: Path):
    dataset = create_enhanced_cps(
        source_file_path=_subset_source(tmp_path, 10),
        calibrate=False,
    )
    system = CountryTaxBenefitSystem()

    for entity, frame in (
        ("person", dataset.person),
        ("benunit", dataset.benunit),
        ("household", dataset.household),
    ):
        invalid_columns = [
            column
            for column in frame.columns
            if column not in system.variables
            or system.variables[column].entity.key != entity
            or (
                not system.variables[column].is_input_variable()
                and column not in ALLOWED_REPORTED_DATA_INPUTS
            )
        ]
        assert invalid_columns == []

    assert "household_wealth" not in dataset.household.columns
    assert "total_wealth" not in dataset.household.columns
    for column in (
        "savings",
        "main_residence_value",
        "other_residential_property_value",
        "non_residential_property_value",
        "owned_land",
        "corporate_wealth",
    ):
        assert column in dataset.household.columns


def test_policybench_transfer_runs_uk_microsimulation(tmp_path: Path):
    dataset = create_enhanced_cps(
        source_file_path=_subset_source(tmp_path, 10),
        calibrate=False,
    )
    sim = Microsimulation(dataset=dataset)

    for variable in (
        "household_net_income",
        "income_tax",
        "universal_credit",
    ):
        values = sim.calculate(variable, map_to="household").values
        assert len(values) == len(dataset.household)


def test_policybench_transfer_calibration_improves_loss(tmp_path: Path):
    source_file_path = _subset_source(tmp_path, 100)
    dataset = create_enhanced_cps(
        source_file_path=source_file_path,
        calibrate=False,
    )
    uncalibrated_loss = get_loss_results(dataset, "2025")

    calibrated = create_enhanced_cps(
        source_file_path=source_file_path,
        calibrate=True,
    )
    calibrated_loss = get_loss_results(calibrated, "2025")

    assert (calibrated.household.household_weight > 0).all()
    assert calibrated_loss.abs_rel_error.mean() < uncalibrated_loss.abs_rel_error.mean()


def test_policybench_transfer_calibration_uses_iterative_solver(monkeypatch):
    calls = {}

    def fake_create_target_matrix(dataset, time_period, reform=None):
        return pd.DataFrame(
            [[1.0, 0.0], [0.0, 1.0]],
        ), pd.Series([1.0, 2.0])

    def fake_lsq_linear(*args, **kwargs):
        calls.update(kwargs)
        return SimpleNamespace(success=True, x=np.array([1.0, 2.0]))

    monkeypatch.setattr(
        reweight_module,
        "create_target_matrix",
        fake_create_target_matrix,
    )
    monkeypatch.setattr(reweight_module, "lsq_linear", fake_lsq_linear)

    dataset = SimpleNamespace(household=pd.DataFrame({"household_id": [1, 2]}))
    weights, diagnostics = reweight_module.calibrate_household_weights(
        dataset,
        "2025",
        compute_diagnostics=False,
    )

    assert calls["lsq_solver"] == "lsmr"
    assert calls["lsmr_tol"] == "auto"
    assert calls["lsmr_maxiter"] == 2000
    assert diagnostics is None
    assert weights.tolist() == [1.0, 2.0]


def test_policybench_transfer_family_structure_matches_person_membership(
    tmp_path: Path,
):
    dataset = create_enhanced_cps(
        source_file_path=_subset_source(tmp_path, 20),
        calibrate=False,
    )
    sim = Microsimulation(dataset=dataset)

    benunit_ids = sim.calculate("benunit_id", map_to="benunit").values
    person_benunit_ids = sim.calculate("person_benunit_id", map_to="person").values
    is_adult = sim.calculate("is_adult", map_to="person").values
    is_child = sim.calculate("is_child", map_to="person").values
    is_married = sim.calculate("is_married", map_to="benunit").values
    family_type = sim.calculate("family_type", map_to="benunit").values

    for benunit_id, married, family in zip(benunit_ids, is_married, family_type):
        member_mask = person_benunit_ids == benunit_id
        adults = int(is_adult[member_mask].sum())
        children = int(is_child[member_mask].sum())

        assert bool(married) == (adults == 2)

        if adults == 2 and children > 0:
            expected = "COUPLE_WITH_CHILDREN"
        elif adults == 2:
            expected = "COUPLE_NO_CHILDREN"
        elif children > 0:
            expected = "LONE_PARENT"
        else:
            expected = "SINGLE"

        observed = family.name if hasattr(family, "name") else str(family)
        assert observed == expected


def test_assign_council_tax_bands_handles_upper_percentile_edge():
    households = pd.DataFrame(
        {
            "household_id": [1, 2],
            "region": ["NORTH_EAST", "NORTH_EAST"],
            "household_weight": [0.0, 1.0],
            "housing_score": [10.0, 20.0],
        }
    )

    assigned = _assign_council_tax_bands(households)

    assert len(assigned) == 2
    assert assigned["council_tax_band"].isin(list("ABCDEFGHI")).all()
