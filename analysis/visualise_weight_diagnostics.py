"""Visualise weight distribution diagnostics for the enhanced FRS.

Produces a set of charts showing:
1. Weight distribution histogram (before regularisation)
2. Per-slice Kish effective sample sizes
3. Top high-influence records
4. Influence heatmap (top records x statistics)
"""

import json
import logging
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATASET_PATH = "policyengine_uk_data/storage/enhanced_frs_2023_24.h5"
OUTPUT_PREFIX = "analysis/weight_diagnostics"
TIME_PERIOD = "2025"
# Use fewer reforms for speed; increase for production
N_REFORMS = 10
THRESHOLD = 0.05


def main():
    from policyengine_uk.data import UKSingleYearDataset
    from policyengine_uk_data.diagnostics.influence import (
        compute_influence_matrix,
        compute_kish_effective_sample_size,
        find_high_influence_records,
        _build_slice_assignments,
    )
    from policyengine_uk import Microsimulation

    logger.info("Loading dataset from %s", DATASET_PATH)
    dataset = UKSingleYearDataset(file_path=DATASET_PATH)

    sim = Microsimulation(dataset=dataset)
    sim.default_calculation_period = TIME_PERIOD

    weights = np.asarray(
        sim.calculate("household_weight", map_to="household"),
        dtype=float,
    )

    # ── 1. Weight distribution ──────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.hist(weights, bins=100, edgecolor="white", alpha=0.8, color="#2563eb")
    ax.set_xlabel("Household weight")
    ax.set_ylabel("Count")
    ax.set_title("Weight distribution (all households)")
    ax.axvline(
        np.median(weights),
        color="red",
        linestyle="--",
        label=f"Median: {np.median(weights):,.0f}",
    )
    ax.axvline(
        np.percentile(weights, 90),
        color="orange",
        linestyle="--",
        label=f"P90: {np.percentile(weights, 90):,.0f}",
    )
    ax.axvline(
        np.percentile(weights, 99),
        color="darkred",
        linestyle="--",
        label=f"P99: {np.percentile(weights, 99):,.0f}",
    )
    ax.legend()

    ax = axes[1]
    log_weights = np.log10(np.maximum(weights, 1))
    ax.hist(log_weights, bins=80, edgecolor="white", alpha=0.8, color="#7c3aed")
    ax.set_xlabel("log₁₀(weight)")
    ax.set_ylabel("Count")
    ax.set_title("Weight distribution (log scale)")

    plt.tight_layout()
    fig.savefig(f"{OUTPUT_PREFIX}_weight_dist.png", dpi=150, bbox_inches="tight")
    logger.info("Saved weight distribution plot")
    plt.close()

    # ── 2. Kish effective sample size by slice ──────────────────────
    slices = _build_slice_assignments(sim, TIME_PERIOD)
    kish_data = {"overall": compute_kish_effective_sample_size(weights)}
    for slice_name, labels in slices.items():
        for label in np.unique(labels):
            if label is None:
                continue
            mask = labels == label
            n_actual = mask.sum()
            n_eff = compute_kish_effective_sample_size(weights, mask)
            kish_data[f"{slice_name}={label}"] = n_eff

    kish_df = pd.DataFrame(
        {"slice": list(kish_data.keys()), "kish_n_eff": list(kish_data.values())}
    ).sort_values("kish_n_eff")

    fig, ax = plt.subplots(figsize=(10, max(6, len(kish_df) * 0.3)))
    colors = [
        "#ef4444" if v < 100 else "#f59e0b" if v < 500 else "#22c55e"
        for v in kish_df.kish_n_eff
    ]
    ax.barh(kish_df.slice, kish_df.kish_n_eff, color=colors, edgecolor="white")
    ax.set_xlabel("Kish effective sample size")
    ax.set_title("Effective sample size by population slice")
    ax.axvline(100, color="red", linestyle=":", alpha=0.5, label="n_eff = 100")
    ax.axvline(500, color="orange", linestyle=":", alpha=0.5, label="n_eff = 500")
    ax.legend()
    plt.tight_layout()
    fig.savefig(f"{OUTPUT_PREFIX}_kish.png", dpi=150, bbox_inches="tight")
    logger.info("Saved Kish ESS plot")
    plt.close()

    # ── 3. Influence matrix ─────────────────────────────────────────
    logger.info("Computing baseline influence matrix...")
    infl = compute_influence_matrix(sim, TIME_PERIOD)
    flagged = find_high_influence_records(infl, THRESHOLD)

    if not flagged.empty:
        # Top flagged records table
        fig, ax = plt.subplots(figsize=(12, max(4, len(flagged.head(20)) * 0.4)))
        ax.axis("off")
        table_data = flagged.head(20).copy()
        table_data["max_influence"] = table_data["max_influence"].map(
            lambda x: f"{x:.3f}"
        )
        table = ax.table(
            cellText=table_data.values,
            colLabels=table_data.columns,
            cellLoc="center",
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.auto_set_column_width(col=list(range(len(table_data.columns))))
        ax.set_title(
            f"Top {min(20, len(flagged))} high-influence records "
            f"(threshold={THRESHOLD})",
            fontsize=12,
            pad=20,
        )
        plt.tight_layout()
        fig.savefig(
            f"{OUTPUT_PREFIX}_flagged_records.png", dpi=150, bbox_inches="tight"
        )
        logger.info("Saved flagged records table")
        plt.close()

        # Influence heatmap for top records
        top_n = min(15, len(flagged))
        top_indices = flagged.record_idx.iloc[:top_n].values

        # Select columns with highest max influence
        col_maxes = infl.max(axis=0).sort_values(ascending=False)
        top_cols = col_maxes.head(30).index
        heatmap_data = infl.iloc[top_indices][top_cols]

        fig, ax = plt.subplots(figsize=(16, max(4, top_n * 0.5)))
        im = ax.imshow(heatmap_data.values, aspect="auto", cmap="YlOrRd")
        ax.set_yticks(range(top_n))
        ax.set_yticklabels([f"HH #{idx}" for idx in top_indices], fontsize=7)
        ax.set_xticks(range(len(top_cols)))
        ax.set_xticklabels(
            [c.split("/")[-1][:25] for c in top_cols],
            rotation=90,
            fontsize=6,
        )
        ax.set_title("Influence heatmap: top records × top statistics")
        plt.colorbar(im, ax=ax, label="Influence fraction")
        plt.tight_layout()
        fig.savefig(f"{OUTPUT_PREFIX}_heatmap.png", dpi=150, bbox_inches="tight")
        logger.info("Saved influence heatmap")
        plt.close()
    else:
        logger.info("No records exceed influence threshold — no flagged records plot")

    # ── 4. Weight vs influence scatter ──────────────────────────────
    max_infl_per_record = infl.max(axis=1) if not infl.empty else pd.Series(dtype=float)

    if not max_infl_per_record.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        sc = ax.scatter(
            weights,
            max_infl_per_record.values,
            alpha=0.3,
            s=5,
            c=np.log10(np.maximum(weights, 1)),
            cmap="viridis",
        )
        ax.set_xlabel("Household weight")
        ax.set_ylabel("Max influence across all statistics")
        ax.set_title("Weight vs maximum influence")
        ax.axhline(THRESHOLD, color="red", linestyle="--", label=f"Threshold={THRESHOLD}")
        ax.set_xscale("log")
        ax.legend()
        plt.colorbar(sc, ax=ax, label="log₁₀(weight)")
        plt.tight_layout()
        fig.savefig(f"{OUTPUT_PREFIX}_scatter.png", dpi=150, bbox_inches="tight")
        logger.info("Saved weight vs influence scatter")
        plt.close()

    # ── 5. Summary statistics ───────────────────────────────────────
    summary = {
        "n_households": int(len(weights)),
        "weight_mean": float(np.mean(weights)),
        "weight_median": float(np.median(weights)),
        "weight_p90": float(np.percentile(weights, 90)),
        "weight_p99": float(np.percentile(weights, 99)),
        "weight_max": float(np.max(weights)),
        "weight_skewness": float(
            np.mean(((weights - np.mean(weights)) / np.std(weights)) ** 3)
        ),
        "kish_overall": float(kish_data["overall"]),
        "n_flagged_records": int(len(flagged)) if not flagged.empty else 0,
        "threshold": THRESHOLD,
    }

    with open(f"{OUTPUT_PREFIX}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Saved summary to %s_summary.json", OUTPUT_PREFIX)

    # Print summary
    print("\n" + "=" * 60)
    print("WEIGHT DIAGNOSTICS SUMMARY")
    print("=" * 60)
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k}: {v:,.2f}")
        else:
            print(f"  {k}: {v}")
    print("=" * 60)


if __name__ == "__main__":
    main()
