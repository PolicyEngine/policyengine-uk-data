def test_child_limit():
    from policyengine_uk import Microsimulation
    from policyengine_uk_data.datasets import EnhancedFRS_2022_23

    # Initialize simulation
    sim = Microsimulation(dataset=EnhancedFRS_2022_23)

    child_is_affected = (
        sim.map_result(
            sim.calculate(
                "uc_is_child_limit_affected", map_to="household", period=2025
            ),
            "household",
            "person",
        )
        > 0
    ) * sim.calculate("is_child", map_to="person").values
    child_in_uc_household = (
        sim.calculate("universal_credit", map_to="person", period=2025).values
        > 0
    )
    children_in_capped_households = sim.map_result(
        child_is_affected * child_in_uc_household, "person", "household"
    )
    capped_households = (children_in_capped_households > 0) * 1.0
    household_weight = sim.calculate("household_weight", period=2025).values
    children_affected = (
        children_in_capped_households * household_weight
    ).sum()
    households_affected = (capped_households * household_weight).sum()

    UPRATING_24_25 = 1.12  # https://ifs.org.uk/articles/two-child-limit-poverty-incentives-and-cost, table at the end
    child_target = (
        1.6e6 * UPRATING_24_25
    )  # Expected number of affected children
    household_target = (
        440e3 * UPRATING_24_25
    )  # Expected number of affected households

    assert (
        abs(children_affected / child_target - 1) < 0.1
    ), f"Expected {child_target/1e6:.1f} million affected children, got {children_affected/1e6:.1f} million."
    assert (
        abs(households_affected / household_target - 1) < 0.1
    ), f"Expected {household_target/1e3:.0f} thousand affected households, got {households_affected/1e3:.0f} thousand."
