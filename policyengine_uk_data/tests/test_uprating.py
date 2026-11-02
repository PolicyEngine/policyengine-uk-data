"""Dataset uprating preserves the meaning of carried calibration priors."""

import numpy as np
import pandas as pd
from policyengine_uk.data import UKSingleYearDataset

from policyengine_uk_data.utils import uprating


def test_carried_calibration_priors_follow_household_weight_uprating(
    tmp_path, monkeypatch
):
    factors = pd.DataFrame(
        {"2024": [1.2, 1.0], "2025": [1.5, 2.0]},
        index=pd.Index(["household_weight", "capital_gains"], name="Variable"),
    )
    factors.to_csv(tmp_path / "uprating_factors.csv")
    monkeypatch.setattr(uprating, "STORAGE_FOLDER", tmp_path)
    household = pd.DataFrame(
        {
            "household_id": [1, 2, 3, 4],
            "household_weight": [850.0, 150.0, 0.0, 50.0],
            "household_calibration_prior": [425.0, 75.0, 850.0, 150.0],
        }
    )
    dataset = UKSingleYearDataset(
        person=pd.DataFrame(
            {
                "person_id": [1, 2, 3, 4],
                "person_household_id": [1, 2, 3, 4],
                "person_benunit_id": [1, 2, 3, 4],
                "capital_gains": [0.0, 4_000.0, 0.0, 5_000.0],
            }
        ),
        benunit=pd.DataFrame({"benunit_id": [1, 2, 3, 4]}),
        household=household.copy(),
        fiscal_year=2024,
    )

    result = uprating.uprate_dataset(dataset, 2025)

    np.testing.assert_allclose(
        result.household.household_calibration_prior,
        household.household_calibration_prior * 1.25,
    )
    np.testing.assert_allclose(
        result.household.household_weight, household.household_weight * 1.25
    )
    np.testing.assert_allclose(
        result.person.capital_gains, dataset.person.capital_gains * 2
    )
    pd.testing.assert_frame_equal(dataset.household, household)
    restored = uprating.uprate_dataset(result, 2024)
    pd.testing.assert_frame_equal(restored.household, household)
    pd.testing.assert_frame_equal(restored.person, dataset.person)
