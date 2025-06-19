import pytest
import numpy as np


@pytest.mark.parametrize("year", ["2022_23"])
def test_small_frs_loads(year: int):
    from policyengine_core.data import Dataset
    from policyengine_uk_data.storage import STORAGE_FOLDER
    from policyengine_us import Microsimulation

    sim = Microsimulation(
        dataset=Dataset.from_file(
            STORAGE_FOLDER / f"small_cps_{year}.h5",
        )
    )

    assert not sim.calculate("household_net_income", 2025).isna().any()
