"""End-to-end regression test: bus_fare_spending must survive the full build.

`generate_lcfs_table` is unit-tested to compute the bus_fare_spending column
(test_lcfs_consumption_ingestion), but nothing checks that it survives the
QRF train/predict and enhanced-dataset assembly/save into the published
dataset. It currently does not (see issue #430) — every other consumption
output lands, but bus_fare_spending is dropped somewhere downstream.

This test is marked xfail so it is mergeable and documents the known gap; it
will XPASS once the pipeline is fixed, prompting removal of the marker and
conversion to a hard assertion.
"""

import pytest


@pytest.mark.xfail(
    reason=(
        "bus_fare_spending is imputed but dropped downstream of "
        "generate_lcfs_table before reaching the enhanced dataset (issue #430). "
        "Remove this marker once the dataset carries the column."
    ),
    strict=False,
)
def test_enhanced_dataset_contains_bus_fare_spending(baseline):
    assert "bus_fare_spending" in baseline.input_variables, (
        "bus_fare_spending is not present in the enhanced dataset."
    )
    total = baseline.calculate(
        "bus_fare_spending", map_to="household", period=2025
    ).sum()
    # UK household bus/coach fare spend is ~£2.7bn; guard against an all-zero
    # column slipping through as 'present'.
    assert total > 1e9, (
        f"bus_fare_spending present but implausibly small: £{total / 1e9:.2f}bn"
    )
