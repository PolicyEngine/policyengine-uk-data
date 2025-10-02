import pytest

AGGREGATES = {
    "nhs_spending": 200e9,
    # "dfe_education_spending": 70e9,
    "rail_subsidy_spending": 12e9,
    "bus_subsidy_spending": 2.5e9,
}


@pytest.mark.parametrize("variable", AGGREGATES.keys())
def test_aggregates(baseline, variable: str):
    estimate = baseline.calculate(
        variable, map_to="household", period=2025
    ).sum()

    assert (
        abs(estimate / AGGREGATES[variable] - 1) < 0.7
    ), f"Expected {AGGREGATES[variable]/1e9:.1f} billion for {variable}, got {estimate/1e9:.1f} billion (relative error = {abs(estimate / AGGREGATES[variable] - 1):.1%})."
