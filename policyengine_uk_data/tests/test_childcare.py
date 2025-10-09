def test_childcare(baseline, enhanced_frs):
    import numpy as np

    # Define targets (same as in the optimization script)
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

    # Calculate dataframe with all required variables
    df = baseline.calculate_dataframe(
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
            "maximum_extended_childcare_hours_usage",  # Added this variable
        ],
        2024,
    )

    # Calculate actual spending values
    spending = {
        "tfc": baseline.calculate("tax_free_childcare", 2024).sum() / 1e9,
        "extended": baseline.calculate(
            "extended_childcare_entitlement", 2024
        ).sum()
        / 1e9,
        "targeted": baseline.calculate(
            "targeted_childcare_entitlement", 2024
        ).sum()
        / 1e9,
        "universal": baseline.calculate(
            "universal_childcare_entitlement", 2024
        ).sum()
        / 1e9,
    }

    # Calculate actual caseload values
    caseload = {
        "tfc": df["is_child_receiving_tax_free_childcare"].sum() / 1e3,
        "extended": df["is_child_receiving_extended_childcare"].sum() / 1e3,
        "universal": df["is_child_receiving_universal_childcare"].sum() / 1e3,
        "targeted": df["is_child_receiving_targeted_childcare"].sum() / 1e3,
    }

    # Calculate take-up rates for reporting
    take_up_rates = {
        "tfc": df["would_claim_tfc"].mean(),
        "extended": df["would_claim_extended_childcare"].mean(),
        "universal": df["would_claim_universal_childcare"].mean(),
        "targeted": df["would_claim_targeted_childcare"].mean(),
    }

    # Report extended hours usage statistics
    hours_mean = df["maximum_extended_childcare_hours_usage"].mean()
    hours_std = df["maximum_extended_childcare_hours_usage"].std()

    # Print results table
    print("\n===== CHILDCARE TEST RESULTS =====")

    print("\nTAKE-UP RATES:")
    for key, rate in take_up_rates.items():
        print(f"{key.upper():<12} {rate:.3f}")

    print(
        f"\nEXTENDED HOURS: Mean = {hours_mean:.2f}, Std Dev = {hours_std:.2f}"
    )

    print("\nSPENDING (£ billion):")
    print(
        f"{'PROGRAM':<12} {'ACTUAL':<10} {'TARGET':<10} {'RATIO':<10} {'PASS?':<10}"
    )
    print("-" * 55)

    failed_any = False
    # Test spending for each program
    for key in targets["spending"]:
        target_spending = targets["spending"][key]
        ratio = spending[key] / target_spending
        passed = abs(ratio - 1) < 1
        status = "✓" if passed else "✗"
        print(
            f"{key.upper():<12} {spending[key]:<10.3f} {target_spending:<10.3f} {ratio:<10.3f} {status:<10}"
        )
        if not passed:
            failed_any = True

    print("\nCASELOAD (thousands):")
    print(
        f"{'PROGRAM':<12} {'ACTUAL':<10} {'TARGET':<10} {'RATIO':<10} {'PASS?':<10}"
    )
    print("-" * 55)

    # Test caseload for each program
    for key in targets["caseload"]:
        target_caseload = targets["caseload"][key]
        ratio = caseload[key] / target_caseload
        passed = abs(ratio - 1) < 1
        status = "✓" if passed else "✗"
        print(
            f"{key.upper():<12} {caseload[key]:<10.1f} {target_caseload:<10.1f} {ratio:<10.3f} {status:<10}"
        )
        if not passed:
            failed_any = True

    assert not failed_any
