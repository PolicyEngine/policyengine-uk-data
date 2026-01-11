def test_scotland_children_under_16(baseline):
    """Test Scotland children under 16 aligns with NRS mid-2023 estimates."""
    age = baseline.calculate("age", period=2025)
    region = baseline.calculate("region", map_to="person", period=2025)
    weight = baseline.calculate(
        "household_weight", map_to="person", period=2025
    )

    scotland_children = ((region == "SCOTLAND") & (age < 16)).values
    scotland_children_count = (scotland_children * weight.values).sum()

    CHILDREN_TARGET = (
        896e3  # 896k children under 16 in Scotland (NRS, 2025 projection)
    )

    assert (
        abs(scotland_children_count / CHILDREN_TARGET - 1) < 0.15
    ), f"Expected {CHILDREN_TARGET/1e3:.0f}k children under 16 in Scotland, got {scotland_children_count/1e3:.0f}k."


def test_scotland_households_3plus_children(baseline):
    """Test Scotland households with 3+ children aligns with Census 2022."""
    is_child = baseline.calculate("is_child", period=2025)
    household_weight = baseline.calculate(
        "household_weight", period=2025
    ).values
    region = baseline.calculate("region", map_to="household", period=2025)

    children_per_hh = baseline.map_result(is_child, "person", "household")
    scotland_3plus = (region.values == "SCOTLAND") & (children_per_hh >= 3)
    scotland_3plus_count = (scotland_3plus * household_weight).sum()

    HOUSEHOLDS_TARGET = 82e3  # 82k households with 3+ children (Census 2022)

    assert (
        abs(scotland_3plus_count / HOUSEHOLDS_TARGET - 1) < 0.20
    ), f"Expected {HOUSEHOLDS_TARGET/1e3:.0f}k households with 3+ children in Scotland, got {scotland_3plus_count/1e3:.0f}k."
