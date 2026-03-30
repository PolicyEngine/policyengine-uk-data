from policyengine_uk import Microsimulation

from policyengine_uk_data.datasets import create_enhanced_cps
from policyengine_uk_data.utils.loss import get_loss_results


def test_policybench_transfer_dataset_validates():
    dataset = create_enhanced_cps(max_rows=10, calibrate=False)

    dataset.validate()

    assert len(dataset.household) == 10
    assert len(dataset.benunit) == 10
    assert len(dataset.person) >= 10
    assert (dataset.household.household_weight > 0).all()


def test_policybench_transfer_runs_uk_microsimulation():
    dataset = create_enhanced_cps(max_rows=10, calibrate=False)
    sim = Microsimulation(dataset=dataset)

    for variable in (
        "household_net_income",
        "income_tax",
        "universal_credit",
    ):
        values = sim.calculate(variable, map_to="household").values
        assert len(values) == len(dataset.household)


def test_policybench_transfer_calibration_improves_loss():
    dataset = create_enhanced_cps(max_rows=100, calibrate=False)
    uncalibrated_loss = get_loss_results(dataset, "2025")

    calibrated = create_enhanced_cps(max_rows=100, calibrate=True)
    calibrated_loss = get_loss_results(calibrated, "2025")

    assert calibrated_loss.abs_rel_error.mean() < uncalibrated_loss.abs_rel_error.mean()


def test_policybench_transfer_family_structure_matches_person_membership():
    dataset = create_enhanced_cps(max_rows=20, calibrate=False)
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
