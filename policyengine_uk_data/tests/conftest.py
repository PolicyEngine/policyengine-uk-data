import pytest
from policyengine_uk.data import UKSingleYearDataset
from policyengine_uk_data.storage import STORAGE_FOLDER


@pytest.fixture
def frs():
    """FRS dataset for testing."""
    try:
        return UKSingleYearDataset(STORAGE_FOLDER / "frs_2023_24.h5")
    except FileNotFoundError:
        pytest.skip("FRS dataset not available")


@pytest.fixture
def enhanced_frs():
    """Enhanced FRS dataset for testing."""
    try:
        return UKSingleYearDataset(STORAGE_FOLDER / "enhanced_frs_2023_24.h5")
    except FileNotFoundError:
        pytest.skip("Enhanced FRS dataset not available")


@pytest.fixture
def baseline(enhanced_frs: UKSingleYearDataset):
    """Baseline microsimulation using enhanced FRS."""
    from policyengine_uk import Microsimulation

    return Microsimulation(dataset=enhanced_frs)


@pytest.fixture
def enhanced_frs_for_year():
    """Factory fixture that loads the per-year panel snapshot.

    Usage in a test::

        def test_2024_snapshot(enhanced_frs_for_year):
            ds = enhanced_frs_for_year(2024)

    Produced by ``create_yearly_snapshots`` (see step 2 of #345). Skips
    the test if the requested year's file is missing, so the same test
    file can run locally (where only one year may exist) and in the full
    panel build. The base 2023-24 year is served from the legacy
    ``enhanced_frs_2023_24.h5`` filename for backwards compatibility; all
    other years use the ``enhanced_frs_<year>.h5`` naming.
    """

    def _load(year: int) -> UKSingleYearDataset:
        year = int(year)
        candidates = [
            STORAGE_FOLDER / f"enhanced_frs_{year}.h5",
        ]
        if year == 2023:
            candidates.append(STORAGE_FOLDER / "enhanced_frs_2023_24.h5")
        for path in candidates:
            if path.exists():
                return UKSingleYearDataset(path)
        pytest.skip(
            f"Enhanced FRS snapshot for {year} not available "
            f"(looked for: {', '.join(p.name for p in candidates)})"
        )

    return _load
