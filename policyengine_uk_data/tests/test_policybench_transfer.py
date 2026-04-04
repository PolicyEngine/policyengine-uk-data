from pathlib import Path

import pandas as pd
from policyengine_uk import Microsimulation

from policyengine_uk_data.datasets import (
    ENHANCED_CPS_SOURCE_FILE,
    create_enhanced_cps,
)
from policyengine_uk_data.datasets.enhanced_cps import _assign_council_tax_bands
from policyengine_uk_data.utils.loss import get_loss_results


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
    assert (dataset.household.household_weight > 0).all()


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

    assert calibrated_loss.abs_rel_error.mean() < uncalibrated_loss.abs_rel_error.mean()


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
