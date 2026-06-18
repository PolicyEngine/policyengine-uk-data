"""Bus fare / subsidy totals in the built dataset must match the DfT targets.

These use the enhanced FRS dataset, which is produced by ``make data`` (the
build / push CI / local generation) and is *not* fetched by ``make download``.
So the `baseline` fixture skips them in PR CI (no built dataset) and runs them
after a build, against the freshly calibrated data — the same pattern as
test_energy_calibration. Both bus variables are calibrated to the official DfT
totals in the build, so the totals should match closely; a 20% band is allowed.
"""

import pytest

# DfT Annual Bus Statistics, year ending March 2025 (England), uplifted
# England -> UK by ONS mid-2023 population (x 68.3 / 57.7):
#   bus_fare_spending    -> BUS05aii passenger fare receipts £3.4bn (~£4.0bn UK)
#   bus_subsidy_spending -> BUS05bii net government support  £3.0bn (~£3.5bn UK)
# https://www.gov.uk/government/statistics/annual-bus-statistics-year-ending-march-2025/annual-bus-statistics-year-ending-march-2025
BUS_TARGETS = {
    "bus_fare_spending": 3.4e9 * 68.3 / 57.7,
    "bus_subsidy_spending": 3.0e9 * 68.3 / 57.7,
}


@pytest.mark.parametrize("variable,target", sorted(BUS_TARGETS.items()))
def test_bus_total_matches_dft_target(baseline, variable: str, target: float):
    total = baseline.calculate(variable, map_to="household", period=2025).sum()
    assert abs(total / target - 1) < 0.2, (
        f"{variable}: £{total / 1e9:.2f}bn vs DfT target £{target / 1e9:.2f}bn "
        f"(relative error {abs(total / target - 1):.1%})."
    )
