"""CLI entry point for adversarial weight regularisation.

Usage:
    uv run python -m policyengine_uk_data.diagnostics [command] [options]

Commands:
    diagnose    Run Phase 1 influence diagnostics (read-only)
    train       Train the generative model on FRS attributes
    regularise  Run the full adversarial loop (detect + spawn + recalibrate)
"""

import argparse
import json
import logging
import sys

import numpy as np


def cmd_diagnose(args):
    """Run influence diagnostics on a dataset."""
    from policyengine_uk.data import UKSingleYearDataset
    from policyengine_uk_data.diagnostics.influence import run_diagnostics

    dataset = UKSingleYearDataset(file_path=args.dataset)
    results = run_diagnostics(
        dataset,
        time_period=args.year,
        n_reforms=args.n_reforms,
        threshold=args.threshold,
        seed=args.seed,
    )

    print("\n=== Weight distribution ===")
    for k, v in results["weight_stats"].items():
        print(f"  {k}: {v:,.1f}")

    print(f"\n=== Flagged records (threshold={args.threshold}) ===")
    flagged = results["flagged_records"]
    if flagged.empty:
        print("  No records exceed the influence threshold.")
    else:
        print(f"  {len(flagged)} records flagged")
        print(flagged.head(20).to_string(index=False))

    print("\n=== Kish effective sample size (top 20 worst) ===")
    kish = results["kish_by_slice"]
    sorted_kish = sorted(kish.items(), key=lambda x: x[1])
    for name, val in sorted_kish[:20]:
        print(f"  {name}: {val:,.0f}")

    if args.output:
        # Save full results to JSON
        serialisable = {
            "weight_stats": results["weight_stats"],
            "kish_by_slice": {k: float(v) for k, v in kish.items()},
            "flagged_records": flagged.to_dict(orient="records"),
        }
        with open(args.output, "w") as f:
            json.dump(serialisable, f, indent=2)
        print(f"\nFull results saved to {args.output}")


def cmd_train(args):
    """Train the generative model."""
    from policyengine_uk.data import UKSingleYearDataset
    from policyengine_uk_data.diagnostics.generative_model import (
        train_generative_model,
        extract_household_features,
        validate_generative_model,
    )
    import pickle

    dataset = UKSingleYearDataset(file_path=args.dataset)
    model = train_generative_model(
        dataset,
        epochs=args.epochs,
        seed=args.seed,
    )

    # Validate
    features = extract_household_features(dataset)
    validation = validate_generative_model(model, features)

    print("\n=== Generative model validation ===")
    print("Marginal KS statistics (lower is better):")
    for col, ks in sorted(validation["marginal_ks"].items(), key=lambda x: -x[1])[:10]:
        print(f"  {col}: {ks:.3f}")

    if validation["correlation_diff"] is not None:
        print(f"Max correlation difference: {validation['correlation_diff']:.3f}")

    # Save model
    with open(args.output, "wb") as f:
        pickle.dump(model, f)
    print(f"\nModel saved to {args.output}")


def cmd_regularise(args):
    """Run the full adversarial weight regularisation pipeline."""
    from policyengine_uk.data import UKSingleYearDataset
    from policyengine_uk_data.diagnostics.offspring import (
        run_adversarial_loop,
    )
    from policyengine_uk_data.diagnostics.recalibrate import (
        prune_zero_weight_records,
    )
    import pickle

    dataset = UKSingleYearDataset(file_path=args.dataset)

    # Load or train generative model
    if args.model:
        with open(args.model, "rb") as f:
            model = pickle.load(f)
    else:
        from policyengine_uk_data.diagnostics.generative_model import (
            train_generative_model,
        )

        print("No model provided, training generative model...")
        model = train_generative_model(dataset, epochs=args.train_epochs)

    result = run_adversarial_loop(
        dataset,
        model,
        time_period=args.year,
        threshold=args.threshold,
        max_rounds=args.max_rounds,
        n_offspring=args.n_offspring,
        weight_target=args.weight_target,
        seed=args.seed,
    )

    expanded = result["expanded_dataset"]

    # Prune zero-weight records
    pruned = prune_zero_weight_records(expanded, epsilon=1.0)

    # Save
    pruned.save(args.output)

    weights = pruned.household.household_weight.values
    print("\n=== Results ===")
    print(f"  Rounds completed: {result['rounds_completed']}")
    print(f"  Records added: {result['records_expanded']}")
    print(f"  Final dataset size: {len(pruned.household)} households")
    print(f"  Max weight: {weights.max():,.0f}")
    print(f"  Median weight: {np.median(weights):,.0f}")
    print(f"  Influence history: {[f'{x:.3f}' for x in result['influence_history']]}")
    print(f"\nExpanded dataset saved to {args.output}")


def main():
    parser = argparse.ArgumentParser(
        description="Adversarial weight regularisation for PolicyEngine UK",
    )
    subparsers = parser.add_subparsers(dest="command")

    # diagnose
    diag = subparsers.add_parser(
        "diagnose",
        help="Run influence diagnostics",
    )
    diag.add_argument("dataset", help="Path to .h5 dataset")
    diag.add_argument("--year", default="2025", help="Time period")
    diag.add_argument(
        "--n-reforms",
        type=int,
        default=50,
        help="Number of random reforms for influence sampling",
    )
    diag.add_argument(
        "--threshold",
        type=float,
        default=0.05,
        help="Influence threshold",
    )
    diag.add_argument("--seed", type=int, default=42)
    diag.add_argument("--output", "-o", help="Output JSON file for full results")

    # train
    tr = subparsers.add_parser(
        "train",
        help="Train generative model",
    )
    tr.add_argument("dataset", help="Path to .h5 dataset")
    tr.add_argument(
        "--epochs",
        type=int,
        default=300,
        help="TVAE training epochs",
    )
    tr.add_argument("--seed", type=int, default=42)
    tr.add_argument(
        "--output",
        "-o",
        default="generative_model.pkl",
        help="Output pickle file",
    )

    # regularise
    reg = subparsers.add_parser(
        "regularise",
        help="Run full adversarial loop",
    )
    reg.add_argument("dataset", help="Path to .h5 dataset")
    reg.add_argument(
        "--model",
        help="Path to trained generative model (.pkl)",
    )
    reg.add_argument("--year", default="2025", help="Time period")
    reg.add_argument(
        "--threshold",
        type=float,
        default=0.05,
        help="Influence threshold",
    )
    reg.add_argument(
        "--max-rounds",
        type=int,
        default=10,
        help="Max adversarial rounds",
    )
    reg.add_argument(
        "--n-offspring",
        type=int,
        default=50,
        help="Offspring per flagged record",
    )
    reg.add_argument(
        "--weight-target",
        type=float,
        default=None,
        help="Target max weight for offspring splitting",
    )
    reg.add_argument(
        "--train-epochs",
        type=int,
        default=300,
        help="TVAE epochs if training from scratch",
    )
    reg.add_argument("--seed", type=int, default=42)
    reg.add_argument(
        "--output",
        "-o",
        default="regularised_dataset.h5",
        help="Output .h5 file",
    )

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(1)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    commands = {
        "diagnose": cmd_diagnose,
        "train": cmd_train,
        "regularise": cmd_regularise,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
