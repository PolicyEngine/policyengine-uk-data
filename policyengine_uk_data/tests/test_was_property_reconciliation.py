"""Aggregate-only source checks do not need licensed WAS records."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings, strategies as st

SCRIPT = Path(__file__).resolve().parents[2] / "tools/reconcile_was_property.py"
SPEC = importlib.util.spec_from_file_location("reconcile_was_property", SCRIPT)
reconciliation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reconciliation)

# Deliberately independent of both the source mapping and measurement script.
SEVEN_SOURCE_FIELDS = (
    "DVhValueR8",
    "DVHseValR8_sum",
    "DVBltValR8_sum",
    "DVBlDValR8_sum",
    "DVLUKValR8_sum",
    "DVLOSValR8_sum",
    "DVOprValR8_sum",
)
EXPECTED_SOURCE_TO_MODEL_INPUT = {
    "DVhValueR8": "main_residence_value",
    "DVHseValR8_sum": "other_residential_property_value",
    "DVBltValR8_sum": "other_residential_property_value",
    "DVBlDValR8_sum": "non_residential_property_value",
    "DVLUKValR8_sum": "owned_land",
    "DVLOSValR8_sum": "overseas_property_value",
    "DVOprValR8_sum": "other_property_value",
}


def source_frame(values, weights=None):
    frame = pd.DataFrame(values, columns=SEVEN_SOURCE_FIELDS)
    frame["DVPropertyR8"] = frame.sum(axis=1)
    frame["R8xshhwgt"] = 1.0 if weights is None else weights
    return frame


def test_all_categories_and_households_owning_only_omitted_categories():
    frame = source_frame(np.eye(7) * 100, weights=np.arange(1, 8))
    result = reconciliation.reconcile_property(frame)
    complete = result["complete_cases"]
    assert complete["source_total"] == {
        "unweighted_sum": 700.0,
        "weighted_sum": 2800.0,
    }
    assert complete["source_total_minus_original_three"] == {
        "unweighted_sum": 400.0,
        "weighted_sum": 2100.0,
    }
    assert complete["source_total_minus_seven_components"]["unweighted_sum"] == 0
    assert complete["households_owning_only_omitted_categories"] == {
        "unweighted_count": 4,
        "weighted_count": 21.0,
    }
    assert complete["stack_501_four_component_sum"] == {
        "unweighted_sum": 400.0,
        "weighted_sum": 1000.0,
    }
    assert complete["source_total_minus_stack_501_four"] == {
        "unweighted_sum": 300.0,
        "weighted_sum": 1800.0,
    }
    assert (
        complete["source_total_minus_stack_501_four"]
        == complete["three_remaining_components"]
    )
    assert complete["households_owning_only_remaining_categories"] == {
        "unweighted_count": 3,
        "weighted_count": 18.0,
    }
    assert complete["model_input_sum"] == complete["source_total"]
    assert complete["model_inputs"]["other_residential_property_value"] == {
        "source_fields": ["DVHseValR8_sum", "DVBltValR8_sum"],
        "unweighted_sum": 200.0,
        "weighted_sum": 500.0,
        "owning_households": {"unweighted_count": 2, "weighted_count": 5.0},
        "households_owning_only_this_input": {
            "unweighted_count": 2,
            "weighted_count": 5.0,
        },
    }
    for category in complete["components"].values():
        assert category["unweighted_sum"] == 100
        assert category["households_owning_only_this_category"]["unweighted_count"] == 1


def test_mapping_metadata_distinguishes_historical_and_stack_501_baselines():
    report = reconciliation.reconcile_property(source_frame(np.zeros((1, 7))))
    assert report["source_to_model_input"] == EXPECTED_SOURCE_TO_MODEL_INPUT
    assert len(set(report["source_to_model_input"].values())) == 6
    historical = report["baselines"]["historical_pre_501"]
    assert "Historical" in historical["description"]
    assert historical["source_fields"] == [
        "DVhValueR8",
        "DVHseValR8_sum",
        "DVBlDValR8_sum",
    ]
    assert report["baselines"]["stack_501"]["source_fields"] == [
        "DVhValueR8",
        "DVHseValR8_sum",
        "DVBltValR8_sum",
        "DVBlDValR8_sum",
    ]


@pytest.mark.parametrize("source_field", SEVEN_SOURCE_FIELDS)
def test_each_source_singleton_reaches_exactly_its_expected_model_input(source_field):
    frame = source_frame(np.zeros((1, 7)), weights=[2])
    frame[source_field] = 125
    frame["DVPropertyR8"] = 125
    complete = reconciliation.reconcile_property(frame)["complete_cases"]
    assert set(complete["model_inputs"]) == set(EXPECTED_SOURCE_TO_MODEL_INPUT.values())
    for model_input, summary in complete["model_inputs"].items():
        expected = (
            125 if model_input == EXPECTED_SOURCE_TO_MODEL_INPUT[source_field] else 0
        )
        assert summary["unweighted_sum"] == expected
        assert summary["weighted_sum"] == expected * 2
        assert summary["owning_households"]["unweighted_count"] == (expected > 0)
    assert complete["model_input_sum"] == complete["source_total"]
    assert complete["source_total_minus_model_inputs"]["unweighted_sum"] == 0


def test_buy_to_let_and_second_homes_are_grouped_and_counted_exactly_once():
    frame = source_frame([[0, 130, 70, 0, 0, 0, 0]], weights=[3])
    complete = reconciliation.reconcile_property(frame)["complete_cases"]
    other_residential = complete["model_inputs"]["other_residential_property_value"]
    assert other_residential["source_fields"] == ["DVHseValR8_sum", "DVBltValR8_sum"]
    assert other_residential["unweighted_sum"] == 200
    assert other_residential["weighted_sum"] == 600
    assert other_residential["owning_households"]["unweighted_count"] == 1
    assert complete["model_input_sum"] == complete["source_total"]
    assert complete["source_total_minus_model_inputs"]["unweighted_sum"] == 0
    assert complete["source_total_minus_stack_501_four"]["unweighted_sum"] == 0


@given(
    st.lists(
        st.lists(st.integers(0, 1_000_000), min_size=7, max_size=7),
        min_size=1,
        max_size=20,
    )
)
@settings(deadline=None)
def test_complete_source_identity_reconciles_before_any_transformation(values):
    result = reconciliation.reconcile_property(source_frame(values))
    complete = result["complete_cases"]
    assert complete["households"] == len(values)
    assert complete["source_total_minus_seven_components"]["unweighted_sum"] == 0
    assert complete["absolute_seven_component_residual"]["weighted_sum"] == 0
    assert (
        complete["source_total_minus_original_three"]
        == complete["four_previously_omitted_components"]
    )
    assert (
        complete["source_total_minus_stack_501_four"]
        == complete["three_remaining_components"]
    )
    assert complete["model_input_sum"] == complete["source_total"]
    assert complete["source_total_minus_model_inputs"]["unweighted_sum"] == 0
    for model_input, summary in complete["model_inputs"].items():
        expected_source_fields = [
            source_field
            for source_field, expected_input in EXPECTED_SOURCE_TO_MODEL_INPUT.items()
            if expected_input == model_input
        ]
        expected = sum(
            row[SEVEN_SOURCE_FIELDS.index(source_field)]
            for row in values
            for source_field in expected_source_fields
        )
        assert summary["source_fields"] == expected_source_fields
        assert summary["unweighted_sum"] == expected
        assert summary["weighted_sum"] == expected


def test_missing_and_negative_codes_reported_separately_from_complete_cases():
    frame = source_frame(np.full((6, 7), 10.0), weights=[1, 2, 3, -1, np.nan, 0])
    frame.loc[1, "DVLOSValR8_sum"] = np.nan
    frame.loc[2, "DVLUKValR8_sum"] = -9
    frame["DVOprValR8_sum"] = frame["DVOprValR8_sum"].astype(object)
    frame.loc[3, "DVOprValR8_sum"] = "unknown"
    frame.loc[4, "DVBlDValR8_sum"] = np.inf
    result = reconciliation.reconcile_property(frame)
    assert result["gross_value_invalid_counts"]["DVLOSValR8_sum"]["missing"] == 1
    assert result["gross_value_invalid_counts"]["DVLUKValR8_sum"]["negative"] == 1
    assert (
        result["gross_value_invalid_counts"]["DVLUKValR8_sum"]["sentinel_minus_9"] == 1
    )
    assert result["gross_value_invalid_counts"]["DVOprValR8_sum"]["non_numeric"] == 1
    assert result["gross_value_invalid_counts"]["DVBlDValR8_sum"]["non_finite"] == 1
    assert result["households_excluded_from_complete_cases"] == 4
    assert result["complete_cases"]["households"] == 2
    assert result["weight_invalid_counts"]["invalid"] == 3
    assert result["cleaned"]["source_total_minus_seven_components"] == {
        "unweighted_sum": 40.0,
        "weighted_sum": 50.0,
    }


def test_known_codes_and_other_negatives_are_distinguished():
    frame = source_frame(np.ones((3, 7)))
    frame["DVLOSValR8_sum"] = [-9, -8, -1]
    report = reconciliation.reconcile_property(frame)
    counts = report["gross_value_invalid_counts"]["DVLOSValR8_sum"]
    assert counts["sentinel_minus_9"] == 1
    assert counts["sentinel_minus_8"] == 1
    assert counts["other_negative"] == 1
    assert report["complete_cases"]["households"] == 0
    assert (
        report["cleaned"]["source_total_minus_seven_components"]["unweighted_sum"] == 3
    )


def test_offsets_cannot_hide_household_discrepancies():
    frame = source_frame(np.ones((2, 7)))
    frame["DVPropertyR8"] += [5, -5]
    result = reconciliation.reconcile_property(frame)["complete_cases"]
    assert result["source_total_minus_seven_components"]["unweighted_sum"] == 0
    assert result["absolute_seven_component_residual"]["unweighted_sum"] == 10
    assert result["households_with_residual_above_tolerance"]["unweighted_count"] == 2


def test_cli_outputs_aggregates_without_household_identifiers(tmp_path, capsys):
    frame = source_frame(np.ones((2, 7)))
    frame["household_identifier"] = ["private-synthetic-id-1", "private-synthetic-id-2"]
    path = tmp_path / "synthetic.tab"
    frame.rename(columns=str.lower).to_csv(path, sep="\t", index=False)
    reconciliation.main(["--household-tab", str(path)])
    output = capsys.readouterr().out
    report = json.loads(output)
    assert report["complete_cases"]["households"] == 2
    assert "private-synthetic-id" not in output
    assert "household_identifier" not in output


@pytest.mark.parametrize("tolerance", [-1, np.inf, np.nan])
def test_invalid_tolerance_rejected(tolerance):
    with pytest.raises(ValueError, match="Tolerance"):
        reconciliation.reconcile_property(
            source_frame(np.ones((1, 7))), tolerance=tolerance
        )
