def test_frs_builds():
    from policyengine_uk_data_v2.frs import PolicyEngineFRSDataset

    dataset = PolicyEngineFRSDataset(year=2022)

    dataset.build()
    dataset.save("frs_2022.h5")
