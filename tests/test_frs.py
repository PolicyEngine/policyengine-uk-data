import pytest

def test_frs_builds():
    from policyengine_uk_data_v2.datasets.frs.main import PolicyEngineFRSDataset

    dataset = PolicyEngineFRSDataset()

    dataset.build(year=2022, tab_folder="data/ukda/frs_2022_23")
    dataset.save_to_h5("frs_2022.h5")


def test_frs_no_nan():
    from policyengine_uk_data_v2.datasets.frs.main import PolicyEngineFRSDataset
    dataset = PolicyEngineFRSDataset()
    dataset.load_from_h5("frs_2022.h5", year=2022)

    assert dataset.person.isna().sum().sum() == 0
    assert dataset.benunit.isna().sum().sum() == 0
    assert dataset.household.isna().sum().sum() == 0
    assert dataset.state.isna().sum().sum() == 0

def test_frs_runs():
    from policyengine_uk import Microsimulation
    from policyengine_core.data import Dataset

    sim = Microsimulation(dataset=Dataset.from_file("frs_2022.h5", time_period=2025))
    sim.calculate("household_net_income", 2025)