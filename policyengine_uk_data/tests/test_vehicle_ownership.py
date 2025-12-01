def test_vehicle_ownership(baseline):
    """Test that vehicle ownership distribution matches NTS 2024 targets."""
    num_cars = baseline.calculate("num_cars", map_to="household", period=2025)
    weights = baseline.calculate("household_weight", period=2025)

    total_hh = weights.sum()

    # NTS 2024 targets: 22% no car, 44% one car, 34% two+ cars
    no_car_rate = ((num_cars == 0) * weights).sum() / total_hh
    one_car_rate = ((num_cars == 1) * weights).sum() / total_hh
    two_plus_rate = ((num_cars >= 2) * weights).sum() / total_hh

    # Allow 15% relative tolerance
    assert (
        abs(no_car_rate / 0.22 - 1) < 0.15
    ), f"Expected 22% households with no car, got {no_car_rate:.0%}"
    assert (
        abs(one_car_rate / 0.44 - 1) < 0.15
    ), f"Expected 44% households with one car, got {one_car_rate:.0%}"
    assert (
        abs(two_plus_rate / 0.34 - 1) < 0.15
    ), f"Expected 34% households with two+ cars, got {two_plus_rate:.0%}"
