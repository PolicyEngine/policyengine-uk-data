import pytest
from policyengine_uk.data import UKSingleYearDataset
from policyengine_uk_data.storage import STORAGE_FOLDER


@pytest.fixture
def frs():
    """FRS dataset for testing."""
    try:
        return UKSingleYearDataset(STORAGE_FOLDER / "frs_2023.h5")
    except FileNotFoundError:
        pytest.skip("FRS dataset not available")


@pytest.fixture
def enhanced_frs():
    """Enhanced FRS dataset for testing."""
    try:
        return UKSingleYearDataset(STORAGE_FOLDER / "enhanced_frs_2023.h5")
    except FileNotFoundError:
        pytest.skip("Enhanced FRS dataset not available")


@pytest.fixture
def baseline(enhanced_frs: UKSingleYearDataset):
    """Baseline microsimulation using enhanced FRS."""
    from policyengine_uk import Microsimulation

    return Microsimulation(dataset=enhanced_frs)
