from policyengine_uk_data.utils.build_environment import get_local_build_issues


def test_get_local_build_issues_requires_python_313():
    issues = get_local_build_issues(
        python_version=(3, 14, 0),
        variable_names={"num_vehicles"},
    )

    assert len(issues) == 1
    assert "Python 3.13" in issues[0]


def test_get_local_build_issues_requires_num_vehicles():
    issues = get_local_build_issues(
        python_version=(3, 13, 7),
        variable_names={"salary", "household_weight"},
    )

    assert len(issues) == 1
    assert "num_vehicles" in issues[0]


def test_get_local_build_issues_accepts_supported_environment():
    assert (
        get_local_build_issues(
            python_version=(3, 13, 7),
            variable_names={"num_vehicles", "salary"},
        )
        == []
    )
