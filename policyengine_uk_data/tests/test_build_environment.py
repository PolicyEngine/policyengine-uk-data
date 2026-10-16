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


def test_get_local_build_issues_requires_projected_corporate_sdlt():
    """policyengine-uk before #2322 held corporate SDLT at £3.13bn."""
    issues = get_local_build_issues(
        python_version=(3, 13, 7),
        variable_names={"num_vehicles"},
        corporate_sdlt=dict.fromkeys(range(2024, 2031), 3.13e9),
    )

    assert len(issues) == 1
    assert "PolicyEngine/policyengine-uk#2322" in issues[0]


def test_get_local_build_issues_accepts_projected_corporate_sdlt():
    assert (
        get_local_build_issues(
            python_version=(3, 13, 7),
            variable_names={"num_vehicles"},
            corporate_sdlt={2024: 4.49e9, 2025: 4.81e9, 2026: 5.98e9},
        )
        == []
    )
