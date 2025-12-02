from policyengine_uk_data.utils.loss import (
    NTS_NO_VEHICLE_RATE,
    NTS_ONE_VEHICLE_RATE,
    NTS_TWO_PLUS_VEHICLE_RATE,
)

ABSOLUTE_TOLERANCE = 0.10


def test_vehicle_ownership(baseline):
    """Test that vehicle ownership distribution matches NTS 2024 targets."""
    num_vehicles = baseline.calculate(
        "num_vehicles", map_to="household", period=2025
    )
    weights = baseline.calculate("household_weight", period=2025)

    total_hh = weights.sum()

    no_vehicle_rate = ((num_vehicles == 0) * weights).sum() / total_hh
    one_vehicle_rate = ((num_vehicles == 1) * weights).sum() / total_hh
    two_plus_rate = ((num_vehicles >= 2) * weights).sum() / total_hh

    assert abs(no_vehicle_rate - NTS_NO_VEHICLE_RATE) < ABSOLUTE_TOLERANCE, (
        f"Expected {NTS_NO_VEHICLE_RATE:.0%} households with no vehicle, "
        f"got {no_vehicle_rate:.0%}"
    )
    assert abs(one_vehicle_rate - NTS_ONE_VEHICLE_RATE) < ABSOLUTE_TOLERANCE, (
        f"Expected {NTS_ONE_VEHICLE_RATE:.0%} households with one vehicle, "
        f"got {one_vehicle_rate:.0%}"
    )
    assert (
        abs(two_plus_rate - NTS_TWO_PLUS_VEHICLE_RATE) < ABSOLUTE_TOLERANCE
    ), (
        f"Expected {NTS_TWO_PLUS_VEHICLE_RATE:.0%} households with two+ vehicles, "
        f"got {two_plus_rate:.0%}"
    )
