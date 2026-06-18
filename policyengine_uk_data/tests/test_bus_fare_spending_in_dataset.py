"""End-to-end regression test: bus_fare_spending must reach the enhanced dataset.

`generate_lcfs_table` is unit-tested to compute the bus_fare_spending column;
this guards the other half — that it survives the QRF predict and the
enhanced-dataset assembly/save into the dataset the model loads. It is present
in the current release (enhanced_frs_2024_25.h5) and calibrated to the DfT
passenger-fare total in the build.
"""


def test_enhanced_dataset_contains_bus_fare_spending(baseline):
    assert "bus_fare_spending" in baseline.input_variables, (
        "bus_fare_spending is not present in the enhanced dataset."
    )
    total = baseline.calculate(
        "bus_fare_spending", map_to="household", period=2025
    ).sum()
    # Guard against an all-zero column slipping through as 'present'.
    assert total > 1e9, (
        f"bus_fare_spending present but implausibly small: £{total / 1e9:.2f}bn"
    )
