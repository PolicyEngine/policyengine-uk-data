import numpy as np
from scipy.optimize import minimize
from policyengine_uk import Microsimulation

# 🎯 Calibration targets
targets = {
    "spending": {
        "tfc": 0.6,
        "extended": 2.5,
        "targeted": 0.6,
        "universal": 1.7,
    },
    "caseload": {
        "tfc": 660,
        "extended": 740,
        "targeted": 130,
        "universal": 490,
    },
}

# Here is the link to the UK government’s aggregate data for Tax-Free Childcare:
# https://www.gov.uk/government/statistics/tax-free-childcare-statistics-september-2024

# This is the Department for Education (DfE) data for the other childcare programmes:
# https://skillsfunding.service.gov.uk/view-latest-funding/national-funding-allocations/DSG/2024-to-2025

# For our calculations, please refer to this file:
# https://docs.google.com/spreadsheets/d/1HLwxCJAJQNHa64peQFfV47MuNqoOHBJ9lnFENXMTguE/edit?gid=2100110594#gid=2100110594


# 📦 Simulation runner
def simulate_childcare_programs(
    params: list[float], seed: int = 42
) -> tuple[dict[str, float], dict[str, float]]:
    """
    Run a simulation with given takeup rates and maximum extended hours for childcare programs.

    Args:
        params: List of parameter values
               [tfc_rate, extended_rate, targeted_rate, universal_rate, ext_hours_mean, ext_hours_sd]
        seed: Random seed for reproducibility

    Returns:
        tuple: (spending, caseload) dictionaries with results for each childcare program
    """
    # Unpack parameters - now with 6 parameters
    tfc, extended, targeted, universal, ext_hours_mean, ext_hours_sd = params

    # Initialize sim
    sim = Microsimulation(
        dataset="hf://policyengine/policyengine-uk-data/enhanced_frs_2022_23.h5"
    )

    # Get counts of people and benefit units
    benunit_count = sim.calculate("benunit_id").values.shape[0]
    person_count = sim.calculate("person_id").values.shape[0]

    # Set seed
    np.random.seed(seed)

    # Take-up flags
    sim.set_input(
        "would_claim_tfc", 2024, np.random.random(benunit_count) < tfc
    )
    sim.set_input(
        "would_claim_extended_childcare",
        2024,
        np.random.random(benunit_count) < extended,
    )
    sim.set_input(
        "would_claim_targeted_childcare",
        2024,
        np.random.random(benunit_count) < targeted,
    )
    sim.set_input(
        "would_claim_universal_childcare",
        2024,
        np.random.random(benunit_count) < universal,
    )

    # Generate extended childcare hours usage values
    # Using truncated normal distribution with mean and sd
    extended_hours_values = np.random.normal(
        ext_hours_mean, ext_hours_sd, benunit_count
    )
    # Clip values to be between 0 and 30 hours
    extended_hours_values = np.clip(extended_hours_values, 0, 30)

    # Set the maximum extended childcare hours usage variable
    sim.set_input(
        "maximum_extended_childcare_hours_usage", 2024, extended_hours_values
    )

    # Calculate outputs
    df = sim.calculate_dataframe(
        [
            "age",
            "tax_free_childcare",
            "extended_childcare_entitlement",
            "universal_childcare_entitlement",
            "targeted_childcare_entitlement",
            "would_claim_tfc",
            "would_claim_extended_childcare",
            "would_claim_targeted_childcare",
            "would_claim_universal_childcare",
            "is_child_receiving_tax_free_childcare",
            "is_child_receiving_extended_childcare",
            "is_child_receiving_universal_childcare",
            "is_child_receiving_targeted_childcare",
            "maximum_extended_childcare_hours_usage",
        ],
        2024,
    )

    spending = {
        "tfc": sim.calculate("tax_free_childcare", 2024).sum() / 1e9,
        "extended": sim.calculate("extended_childcare_entitlement", 2024).sum()
        / 1e9,
        "targeted": sim.calculate("targeted_childcare_entitlement", 2024).sum()
        / 1e9,
        "universal": sim.calculate(
            "universal_childcare_entitlement", 2024
        ).sum()
        / 1e9,
    }

    caseload = {
        "tfc": df["is_child_receiving_tax_free_childcare"].sum() / 1e3,
        "extended": df["is_child_receiving_extended_childcare"].sum() / 1e3,
        "universal": df["is_child_receiving_universal_childcare"].sum() / 1e3,
        "targeted": df["is_child_receiving_targeted_childcare"].sum() / 1e3,
    }

    return spending, caseload


# 🧮 Objective function
def objective(params: list[float]) -> float:
    """
    Calculate the loss between simulated and target values for childcare programs.

    Args:
        params: List of parameter values [tfc_rate, extended_rate, targeted_rate, universal_rate,
                ext_hours_mean, ext_hours_sd]

    Returns:
        float: Combined loss value measuring distance from targets
    """
    spending, caseload = simulate_childcare_programs(params)
    loss = 0
    for key in targets["spending"]:
        loss += (spending[key] / targets["spending"][key] - 1) ** 2
    for key in targets["caseload"]:
        loss += (caseload[key] / targets["caseload"][key] - 1) ** 2

    # Print current parameters and results for monitoring
    print("\nParameters:", np.round(params, 3))
    print(f"Loss: {loss:.4f}")

    # Print comparison with targets
    print("\nSpending (£ billion):")
    for key in targets["spending"]:
        print(
            f"  {key.upper()}: {spending[key]:.3f} (Target: {targets['spending'][key]:.3f}, Ratio: {spending[key]/targets['spending'][key]:.3f})"
        )

    print("\nCaseload (thousands):")
    for key in targets["caseload"]:
        print(
            f"  {key.upper()}: {caseload[key]:.1f} (Target: {targets['caseload'][key]:.1f}, Ratio: {caseload[key]/targets['caseload'][key]:.3f})"
        )

    return loss


if __name__ == "__main__":
    # 🧠 Initial values and bounds - now with 6 parameters
    x0 = [0.5, 0.5, 0.5, 0.5, 15.0, 5.0]  # take-up rates + hours mean & sd
    bounds = [
        (0, 1),
        (0, 1),
        (0, 1),
        (0, 1),
        (5.0, 30.0),
        (1.0, 10.0),
    ]  # bounds for all parameters

    # 🚀 Run optimization
    result = minimize(
        objective,
        x0,
        bounds=bounds,
        method="L-BFGS-B",
        options={"maxiter": 5, "eps": 1e-2, "disp": True},
    )

    # ✅ Final output
    print("\n✅ Optimized Parameters:")
    print(f"Tax-Free Childcare: {result.x[0]:.3f}")
    print(f"Extended Childcare: {result.x[1]:.3f}")
    print(f"Targeted Childcare: {result.x[2]:.3f}")
    print(f"Universal Childcare: {result.x[3]:.3f}")
    print(f"Extended Hours Mean: {result.x[4]:.1f} hours")
    print(f"Extended Hours SD: {result.x[5]:.1f} hours")
    print(f"Final Loss: {result.fun:.4f}")

    # Simulate with final parameters and show detailed results
    final_spending, final_caseload = simulate_childcare_programs(result.x)

    print("\n📊 Final Results:")
    print("\nSpending (£ billion):")
    for key in targets["spending"]:
        print(
            f"  {key.upper()}: {final_spending[key]:.3f} (Target: {targets['spending'][key]:.3f}, Ratio: {final_spending[key]/targets['spending'][key]:.3f})"
        )

    print("\nCaseload (thousands):")
    for key in targets["caseload"]:
        print(
            f"  {key.upper()}: {final_caseload[key]:.1f} (Target: {targets['caseload'][key]:.1f}, Ratio: {final_caseload[key]/targets['caseload'][key]:.3f})"
        )
