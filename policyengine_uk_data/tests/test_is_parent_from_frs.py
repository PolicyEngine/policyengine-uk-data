import numpy as np

from policyengine_uk_data.datasets.frs import derive_is_parent_from_frs_microdata


def test_is_parent_uses_benefit_unit_not_household_rank():
    result = derive_is_parent_from_frs_microdata(
        person_ids=np.array([1_001, 1_002, 1_003, 1_004]),
        person_benunit_ids=np.array([101, 101, 102, 102]),
        adult_person_ids=np.array([1_001, 1_002, 1_003]),
        benunit_ids=np.array([101, 102]),
        dependent_children=np.array([0, 1]),
    )

    assert result.tolist() == [False, False, True, False]


def test_is_parent_marks_both_adults_in_couple_with_children():
    result = derive_is_parent_from_frs_microdata(
        person_ids=np.array([2_001, 2_002, 2_003]),
        person_benunit_ids=np.array([201, 201, 201]),
        adult_person_ids=np.array([2_001, 2_002]),
        benunit_ids=np.array([201]),
        dependent_children=np.array([1]),
    )

    assert result.tolist() == [True, True, False]
