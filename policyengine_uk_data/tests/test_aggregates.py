import pytest

AGGREGATES = {
    "nhs_spending": 200e9,
    # "dfe_education_spending": 70e9,
    # ORR/GOV.UK rail finance statistics report GBP 21.6bn of government
    # support to the rail industry in 2024-25.
    "rail_subsidy_spending": 21.6e9,
    # Approximate public support for local bus services; kept as a loose
    # smoke-test target because source coverage and dataset coverage differ.
    "bus_subsidy_spending": 2.5e9,
    # DfT Annual Bus Statistics (year ending March 2025) report GBP 3.4bn
    # passenger fare receipts for local bus services in England. The LCFS input
    # is UK household bus/coach fare spending, so this is an order-of-magnitude
    # target. Enable once a dataset built with the bus_fare_spending imputation
    # is published — the column is absent from the currently-released dataset.
    # "bus_fare_spending": 3.4e9,
}


@pytest.mark.parametrize("variable", AGGREGATES.keys())
def test_aggregates(baseline, variable: str):
    estimate = baseline.calculate(variable, map_to="household", period=2025).sum()

    assert abs(estimate / AGGREGATES[variable] - 1) < 0.7, (
        f"Expected {AGGREGATES[variable] / 1e9:.1f} billion for {variable}, got {estimate / 1e9:.1f} billion (relative error = {abs(estimate / AGGREGATES[variable] - 1):.1%})."
    )
