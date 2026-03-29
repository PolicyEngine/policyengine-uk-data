from policyengine_uk import Microsimulation

from policyengine_uk_data.datasets import create_policybench_transfer


def test_policybench_transfer_dataset_validates():
    dataset = create_policybench_transfer(max_rows=10)

    dataset.validate()

    assert len(dataset.household) == 10
    assert len(dataset.benunit) == 10
    assert len(dataset.person) >= 10
    assert (dataset.household.household_weight > 0).all()


def test_policybench_transfer_runs_uk_microsimulation():
    dataset = create_policybench_transfer(max_rows=10)
    sim = Microsimulation(dataset=dataset)

    for variable in (
        "household_net_income",
        "income_tax",
        "universal_credit",
    ):
        values = sim.calculate(variable, map_to="household").values
        assert len(values) == len(dataset.household)
