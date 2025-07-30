import pytest
from policyengine_uk.data import UKSingleYearDataset
from policyengine_uk_data.storage import STORAGE_FOLDER


@pytest.fixture
def enhanced_frs():
    dataset = UKSingleYearDataset(STORAGE_FOLDER / "enhanced_frs_2023.h5")
    return dataset


@pytest.fixture
def baseline(enhanced_frs: UKSingleYearDataset):
    from policyengine_uk import Microsimulation

    return Microsimulation(dataset=enhanced_frs)
