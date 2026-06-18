import pytest

AGGREGATES = {
    "nhs_spending": 200e9,
    # "dfe_education_spending": 70e9,
    # ORR/GOV.UK rail finance statistics report GBP 21.6bn of government
    # support to the rail industry in 2024-25.
    "rail_subsidy_spending": 21.6e9,
    # DfT Annual Bus Statistics (year ending March 2025, England), table
    # BUS05bii: total net government support for local bus services ~GBP 3.0bn.
    # bus_subsidy_spending is calibrated to this in the build.
    "bus_subsidy_spending": 3.0e9,
    # DfT Annual Bus Statistics (year ending March 2025, England), table
    # BUS05aii: passenger fare receipts ~GBP 3.4bn. bus_fare_spending is
    # calibrated to this in the build. Enable once a dataset built with that
    # calibration is published (the released dataset predates it).
    # "bus_fare_spending": 3.4e9,
}


@pytest.mark.parametrize("variable", AGGREGATES.keys())
def test_aggregates(baseline, variable: str):
    estimate = baseline.calculate(variable, map_to="household", period=2025).sum()

    assert abs(estimate / AGGREGATES[variable] - 1) < 0.7, (
        f"Expected {AGGREGATES[variable] / 1e9:.1f} billion for {variable}, got {estimate / 1e9:.1f} billion (relative error = {abs(estimate / AGGREGATES[variable] - 1):.1%})."
    )
