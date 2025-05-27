def test_child_limit():
    from policyengine_uk import Microsimulation
    from policyengine_uk_data.datasets import EnhancedFRS_2022_23

    # Initialize simulation
    sim = Microsimulation(dataset=EnhancedFRS_2022_23)

    child_is_affected = sim.calculate(
        "uc_is_child_limit_affected", map_to="household"
    )
    child_in_uc_household = sim.calculate("is_child", map_to="household") * (
        sim.calculate("universal_credit", map_to="household") > 0
    )

    children_affected = (child_is_affected * child_in_uc_household).sum()

    households_affected = (child_is_affected * child_in_uc_household > 0).sum()

    child_target = 1.6e6  # Expected number of affected children
    household_target = 440e3  # Expected number of affected households

    assert (
        abs(children_affected / child_target - 1) < 0.1
    ), f"Expected {child_target/1e6:.1f} million affected children, got {children_affected/1e6:.1f} million."
    assert (
        abs(households_affected / household_target - 1) < 0.1
    ), f"Expected {household_target/1e3:.0f} thousand affected households, got {households_affected/1e3:.0f} thousand."
