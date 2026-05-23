from policyengine_uk_data.targets.sources.nts_vehicles import (
    NTS_NO_VEHICLE_RATE,
    NTS_ONE_VEHICLE_RATE,
    NTS_TWO_PLUS_VEHICLE_RATE,
)
from policyengine_uk_data.datasets.frs_release import CURRENT_FRS_RELEASE

ABSOLUTE_TOLERANCE = 0.30
PERIOD = CURRENT_FRS_RELEASE.calibration_year


def test_vehicle_ownership(baseline):
    """Test that vehicle ownership distribution roughly matches NTS targets."""
    num_vehicles = baseline.calculate("num_vehicles", map_to="household", period=PERIOD)
    weights = baseline.calculate("household_weight", period=PERIOD)

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
    assert abs(two_plus_rate - NTS_TWO_PLUS_VEHICLE_RATE) < ABSOLUTE_TOLERANCE, (
        f"Expected {NTS_TWO_PLUS_VEHICLE_RATE:.0%} households with two+ vehicles, "
        f"got {two_plus_rate:.0%}"
    )
