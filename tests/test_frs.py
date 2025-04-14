import pytest

def test_frs_builds():
    from policyengine_uk_data_v2.datasets.frs.main import PolicyEngineFRSDataset

    dataset = PolicyEngineFRSDataset()

    dataset.build(year=2022, tab_folder="data/ukda/frs_2022_23")
    dataset.save_to_h5("frs_2022.h5")


@pytest.mark.depends_on("tests/test_frs_builds.py")
def test_frs_no_nan():
    from policyengine_uk_data_v2.datasets.frs.main import PolicyEngineFRSDataset
    dataset = PolicyEngineFRSDataset()
    dataset.load_from_h5("frs_2022.h5", year=2022)

    assert dataset.person.isna().sum().sum() == 0
    assert dataset.benunit.isna().sum().sum() == 0
    assert dataset.household.isna().sum().sum() == 0
    assert dataset.state.isna().sum().sum() == 0