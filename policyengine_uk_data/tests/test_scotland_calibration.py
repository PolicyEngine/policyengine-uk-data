"""Tests for Scotland-specific calibration targets."""


def test_scotland_children_under_16(baseline):
    """Test that Scotland children under 16 population aligns with NRS targets.

    Target: 900k children under 16 in Scotland (NRS mid-2023 population estimates)
    Source: https://www.nrscotland.gov.uk/publications/mid-2023-population-estimates/
    """
    age = baseline.calculate("age")
    region = baseline.calculate("region", map_to="person")
    weight = baseline.calculate("household_weight", map_to="person")

    scotland_children = ((region == "SCOTLAND") & (age < 16)).values
    scotland_children_count = (scotland_children * weight.values).sum() / 1e3

    CHILDREN_TARGET_K = 900  # 900k children under 16 in Scotland
    TOLERANCE = 0.15  # 15% tolerance

    assert abs(scotland_children_count / CHILDREN_TARGET_K - 1) < TOLERANCE, (
        f"Expected ~{CHILDREN_TARGET_K}k children under 16 in Scotland, "
        f"got {scotland_children_count:.0f}k "
        f"({(scotland_children_count / CHILDREN_TARGET_K - 1) * 100:+.1f}%)"
    )


def test_scotland_households_3plus_children(baseline):
    """Test that Scotland households with 3+ children aligns with Census 2022.

    Target: 82k households with 3+ children in Scotland (Census 2022)
    Source: https://www.scotlandscensus.gov.uk/search-the-census#/location/topics/topic?topic=Household%20composition
    """
    is_child = baseline.calculate("is_child")
    household_weight = baseline.calculate("household_weight")
    region = baseline.calculate("region", map_to="household")

    # Count children per household
    children_per_hh = baseline.map_result(is_child, "person", "household")

    scotland_3plus = (
        (region.values == "SCOTLAND") & (children_per_hh >= 3)
    ).astype(float)
    scotland_3plus_count = (scotland_3plus * household_weight.values).sum() / 1e3

    HOUSEHOLDS_TARGET_K = 82  # 82k households with 3+ children in Scotland
    TOLERANCE = 0.20  # 20% tolerance

    assert abs(scotland_3plus_count / HOUSEHOLDS_TARGET_K - 1) < TOLERANCE, (
        f"Expected ~{HOUSEHOLDS_TARGET_K}k households with 3+ children in Scotland, "
        f"got {scotland_3plus_count:.0f}k "
        f"({(scotland_3plus_count / HOUSEHOLDS_TARGET_K - 1) * 100:+.1f}%)"
    )
