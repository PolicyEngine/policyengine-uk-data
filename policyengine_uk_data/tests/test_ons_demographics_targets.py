from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st

from policyengine_uk_data.targets.build_loss_matrix import _compute_column
from policyengine_uk_data.targets.schema import Target, Unit
from policyengine_uk_data.targets.sources.ons_demographics import (
    _AGE_BANDS,
    _GENDER_BANDS,
    _aggregate_ages,
    _find_projection_member,
    _parse_regional_from_csv,
    _parse_uk_totals,
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


@pytest.mark.parametrize("bands", [_AGE_BANDS, _GENDER_BANDS])
@given(age=st.integers(min_value=0, max_value=80))
def test_age_bands_partition_frs_ages_and_respect_top_code(bands, age):
    assert bands[-1] == (80, None)
    for low, high in bands:
        assert low <= 80, "No target may start above the FRS top code"
        assert high is None or high < 80, "80+ must be open-ended"
    assert sum(low <= age and (high is None or age <= high) for low, high in bands) == 1


@pytest.mark.parametrize("sex,multiplier", [("female", 1), ("male", 10)])
def test_80_plus_target_sums_every_projection_age(sex, multiplier):
    projection = pd.DataFrame(
        [
            {"Sex": label, "Age": str(age), 2025: (age + 1) * factor}
            for label, factor in [("Females", 1), ("Males", 10)]
            for age in range(101)
        ]
    )
    targets = {target.name: target for target in _parse_uk_totals(projection)}

    assert targets[f"ons/{sex}_80_plus"].values[2025] == sum(
        (age + 1) * multiplier for age in range(80, 101)
    )


@pytest.mark.parametrize("sex,label", [("female", "Females"), ("male", "Males")])
def test_80_plus_matches_published_ons_2025_projection(sex, label):
    projection = pd.read_csv(
        Path(__file__).parent / "fixtures" / "ons_uk_population_2025.csv"
    )
    targets = {target.name: target for target in _parse_uk_totals(projection)}
    # The fixture retains ONS's single ages 0-104 and terminal 105-109/110+.
    older_ages = {str(age) for age in range(80, 105)} | {
        "105 - 109",
        "110 and over",
    }
    expected = projection.loc[
        (projection.Sex == label) & projection.Age.isin(older_ages), "2025"
    ].sum()
    assert targets[f"ons/{sex}_80_plus"].values[2025] == expected


@pytest.mark.parametrize("sex,label", [("female", "Females"), ("male", "Males")])
@pytest.mark.parametrize(
    "terminal_ages", [["89", "90+"], ["105 - 109", "110 and over"]]
)
def test_80_plus_target_includes_terminal_age_groups(sex, label, terminal_ages):
    projection = pd.DataFrame(
        {
            "Sex": [label] * 4,
            "Age": ["79", "80", *terminal_ages],
            2025: [1, 2, 4, 8],
        }
    )
    targets = {target.name: target for target in _parse_uk_totals(projection)}

    assert targets[f"ons/{sex}_80_plus"].values[2025] == 14


def test_emitted_age_target_names_respect_frs_top_code():
    projection = pd.DataFrame(
        {"Sex": ["Females", "Males"], "Age": [80, 80], 2025: [1, 1]}
    )
    targets = _parse_uk_totals(projection) + _parse_regional_from_csv()
    top_bands = []
    for target in targets:
        if "_age_" not in target.name and not target.name.startswith(
            ("ons/female_", "ons/male_")
        ):
            continue
        low, high = target.name.rsplit("_", 2)[-2:]
        assert int(low) <= 80, target.name
        assert high == "plus" or int(high) < 80, target.name
        if int(low) == 80:
            top_bands.append(target.name)
    assert len(top_bands) == 14  # Both sexes and all twelve regions.


@pytest.mark.parametrize(
    "name,expected",
    [
        ("ons/female_80_plus", [1, 1, 1, 0]),
        ("ons/male_80_plus", [0, 1, 1, 1]),
        ("ons/london_age_80_plus", [1, 1, 1, 1]),
    ],
)
def test_80_plus_dispatch_counts_matching_household_members(name, expected):
    household = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    gender = pd.Series(
        ["FEMALE", "FEMALE", "MALE", "FEMALE", "MALE", "FEMALE", "MALE", "MALE"]
    )
    ctx = SimpleNamespace(
        age=np.array([79, 80, 81, 100, 80, 80, 79, 85]),
        region=pd.Series(
            [
                "LONDON",
                "LONDON",
                "LONDON",
                "WALES",
                "WALES",
                "LONDON",
                "LONDON",
                "LONDON",
            ]
        ),
        sim=SimpleNamespace(calculate=lambda variable: gender),
        household_from_person=lambda values: np.bincount(household, weights=values),
    )
    target = Target(
        name=name,
        variable="age",
        source="ons",
        unit=Unit.COUNT,
        values={2025: 1},
        is_count=True,
    )

    np.testing.assert_array_equal(_compute_column(target, ctx, 2025), expected)
