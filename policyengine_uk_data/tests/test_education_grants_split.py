import numpy as np
import pandas as pd

from policyengine_uk_data.datasets.frs import (
    allocate_reported_education_grants,
    split_reported_education_grants,
)


def test_allocate_reported_education_grants_splits_by_capacity():
    allocations = allocate_reported_education_grants(
        reported_grants=np.array([50, 300, 1_000, 100]),
        grant_capacities={
            "grant_a": np.array([100, 100, 100, 0]),
            "grant_b": np.array([100, 0, 100, 0]),
        },
    )

    np.testing.assert_allclose(allocations["grant_a"], [25, 100, 100, 0])
    np.testing.assert_allclose(allocations["grant_b"], [25, 0, 100, 0])
    np.testing.assert_allclose(allocations["education_grants"], [0, 200, 800, 100])


class FakeStudentSupportSim:
    def __init__(self, values):
        self.values = values

    def calculate(self, variable, year):
        del year
        return self.values[variable]


def test_split_reported_education_grants_updates_residual_and_dsa_expenses():
    pe_person = pd.DataFrame({"education_grants": [900, 1_200, 100]})
    sim = FakeStudentSupportSim(
        {
            "childcare_grant": np.array([300, 0, 0]),
            "parents_learning_allowance": np.array([600, 400, 0]),
            "adult_dependants_grant": np.array([0, 600, 0]),
            "maintenance_loan_in_england_system": np.array([False, False, True]),
            "disabled_students_allowance_course_eligible": np.array(
                [False, False, True]
            ),
            "disabled_students_allowance_has_qualifying_condition": np.array(
                [False, False, True]
            ),
            "disabled_students_allowance_receives_equivalent_support": np.array(
                [False, False, False]
            ),
        }
    )

    result = split_reported_education_grants(pe_person, sim, 2025, dsa_maximum=500)

    assert "childcare_grant" not in result.columns
    assert "parents_learning_allowance" not in result.columns
    assert "adult_dependants_grant" not in result.columns
    np.testing.assert_allclose(
        result["disabled_students_allowance_eligible_expenses"], [0, 0, 100]
    )
    np.testing.assert_allclose(result["education_grants"], [0, 200, 0])
