
from typing import Any

def test_frs_builds() -> None:
    """Test that the FRS dataset can be built and saved to an HDF5 file."""
    from policyengine_uk_data_v2.datasets.frs.main import PolicyEngineFRSDataset

    dataset = PolicyEngineFRSDataset()

    dataset.build(year=2022, tab_folder="data/ukda/frs_2022_23")
    dataset.save_to_h5("frs_2022.h5")


def test_frs_no_nan() -> None:
    """Test that the FRS dataset loaded from an HDF5 file contains no missing values."""
    from policyengine_uk_data_v2.datasets.frs.main import PolicyEngineFRSDataset

    dataset = PolicyEngineFRSDataset()
    dataset.load_from_h5("frs_2022.h5", year=2022)

    assert dataset.person.isna().sum().sum() == 0
    assert dataset.benunit.isna().sum().sum() == 0
    assert dataset.household.isna().sum().sum() == 0
    assert dataset.state.isna().sum().sum() == 0


def test_frs_runs() -> None:
    """Test that the FRS dataset can be used with the PolicyEngine UK microsimulation model."""
    from policyengine_core.data import Dataset
    from policyengine_uk import Microsimulation

    sim = Microsimulation(dataset=Dataset.from_file("frs_2022.h5", time_period=2025))
    sim.calculate("household_net_income", 2025)
