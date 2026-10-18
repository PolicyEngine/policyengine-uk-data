import pandas as pd
import pytest

from policyengine_uk_data.targets.sources.ons_demographics import (
    _aggregate_ages,
    _find_projection_member,
)


def test_aggregate_ages_accepts_string_age_values():
    df = pd.DataFrame(
        {
            "Sex": ["Females", "Females", "Females", "Males"],
            "Age": ["14", "15", "90", "15"],
            2025: [1, 2, 4, 8],
        }
    )

    assert _aggregate_ages(df, "female", 15, 90, [2025]) == {2025: 6.0}


def test_find_projection_member_fails_loudly():
    with pytest.raises(RuntimeError, match="uk_ppp_machine_readable"):
        _find_projection_member(["uk/readme.txt"])
