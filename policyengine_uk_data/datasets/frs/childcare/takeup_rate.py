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
        "tfc": 960,
        "extended": 740,
        "targeted": 130,
        "universal": 490,
    },
}


# 📦 Simulation runner
def run_simulation(params, seed=42):
    tfc, extended, targeted, universal, mean_hours, stderr_hours = params

    # Initialize sim
    sim = Microsimulation(
        dataset="hf://policyengine/policyengine-uk-data/enhanced_frs_2022_23.h5"
    )

    # Get counts
    benunit_count = sim.calculate("benunit_id").values.shape[0]
    person_count = sim.calculate("person_id").values.shape[0]

    # Set seed
    np.random.seed(seed)

    # Take-up flags
    sim.set_input(
        "would_claim_tfc", 2025, np.random.random(benunit_count) < tfc
    )
    sim.set_input(
        "would_claim_extended_childcare",
        2025,
        np.random.random(benunit_count) < extended,
    )
    sim.set_input(
        "would_claim_targeted_childcare",
        2025,
        np.random.random(benunit_count) < targeted,
    )
    sim.set_input(
        "would_claim_universal_childcare",
        2025,
        np.random.random(benunit_count) < universal,
    )

    # Entitlement hours (clipped normal distribution per person)
    hours = np.clip(
        np.random.normal(
            loc=mean_hours, scale=stderr_hours, size=person_count
        ),
        0,
        30,
    )
    sim.set_input("max_free_entitlement_hours_used", 2025, hours)

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
        ],
        2025,
    )

    spending = {
        "tfc": sim.calculate("tax_free_childcare", 2025).sum() / 1e9,
        "extended": sim.calculate("extended_childcare_entitlement", 2025).sum()
        / 1e9,
        "targeted": sim.calculate("targeted_childcare_entitlement", 2025).sum()
        / 1e9,
        "universal": sim.calculate(
            "universal_childcare_entitlement", 2025
        ).sum()
        / 1e9,
    }

    caseload = {
        "tfc": df[(df["tax_free_childcare"] > 0) & (df["age"] < 12)][
            "would_claim_tfc"
        ].sum()
        / 1e3,
        "extended": df[
            (df["extended_childcare_entitlement"] > 0)
            & (df["age"] < 4)
            & (df["age"] > 0.9)
        ]["would_claim_extended_childcare"].sum()
        / 1e3,
        "universal": df[
            (df["universal_childcare_entitlement"] > 0)
            & (df["age"] < 5)
            & (df["age"] >= 3)
        ]["would_claim_universal_childcare"].sum()
        / 1e3,
        "targeted": df[
            (df["targeted_childcare_entitlement"] > 0)
            & (df["age"] < 3)
            & (df["age"] >= 2)
        ]["would_claim_targeted_childcare"].sum()
        / 1e3,
    }

    return spending, caseload


# 🧮 Objective function
def objective(params):
    spending, caseload = run_simulation(params)
    loss = 0
    for key in targets["spending"]:
        loss += (spending[key] / targets["spending"][key] - 1) ** 2
    for key in targets["caseload"]:
        loss += (caseload[key] / targets["caseload"][key] - 1) ** 2
    print(np.round(params, 3), f"Loss: {loss:.4f}")
    return loss


# 🧠 Initial values and bounds
x0 = [0.5, 0.5, 0.5, 0.5, 20, 5]  # take-up rates + mean hours + stderr
bounds = [(0, 1)] * 4 + [(0, 30), (0, 30)]

# 🚀 Run optimization
result = minimize(
    objective,
    x0,
    bounds=bounds,
    method="L-BFGS-B",
    options={"maxiter": 50, "eps": 1e-2, "disp": True},
)

# ✅ Final output
print("\n✅ Optimized Parameters:")
print(f"Tax-Free Childcare: {result.x[0]:.3f}")
print(f"Extended Childcare: {result.x[1]:.3f}")
print(f"Targeted Childcare: {result.x[2]:.3f}")
print(f"Universal Childcare: {result.x[3]:.3f}")
print(f"Mean Hours Used: {result.x[4]:.2f}")
print(f"StdDev Hours Used: {result.x[5]:.2f}")
print(f"Final Loss: {result.fun:.4f}")
