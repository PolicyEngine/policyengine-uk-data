"""Influence detector for survey record weights.

Computes per-record influence across a reporting surface of
(metric x slice) statistics.  A record has high influence when it
contributes a large fraction of a slice-level aggregate, meaning
small perturbations to that record propagate into published outputs.

The reporting surface is built from:
  - metrics:  net income, income tax, NI, universal credit, child
              benefit, pension credit, council tax, housing benefit
  - slices:   income decile, region, age band, family type, tenure

Influence is computed under a sample of policy reforms (random
parameter perturbations) so that structurally high-influence records
are identified regardless of which reform is being analysed.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Reporting surface definition ────────────────────────────────────

METRICS = [
    "household_net_income",
    "income_tax",
    "national_insurance",
    "universal_credit",
    "child_benefit",
    "pension_credit",
    "council_tax",
    "housing_benefit_reported",
    "employment_income",
    "self_employment_income",
]

SLICE_DEFINITIONS = {
    "income_decile": {
        "variable": "household_net_income",
        "bins": 10,
        "labels": [f"decile_{i}" for i in range(1, 11)],
    },
    "region": {
        "variable": "region",
        "categorical": True,
    },
    "age_band": {
        "variable": "age",
        "bins": [0, 16, 25, 35, 45, 55, 65, 75, 100],
        "labels": [
            "0-15",
            "16-24",
            "25-34",
            "35-44",
            "45-54",
            "55-64",
            "65-74",
            "75+",
        ],
    },
    "tenure": {
        "variable": "tenure_type",
        "categorical": True,
    },
}


def _build_slice_assignments(
    sim,
    time_period: str,
) -> dict[str, np.ndarray]:
    """Compute household-level slice assignments.

    Returns a dict mapping slice_name -> array of labels, one per
    household.
    """
    slices = {}

    for name, defn in SLICE_DEFINITIONS.items():
        variable = defn["variable"]

        if defn.get("categorical"):
            values = sim.calculate(variable, map_to="household")
            slices[name] = np.asarray(values)
            continue

        values = sim.calculate(variable, map_to="household").astype(float)
        weights = sim.calculate("household_weight", map_to="household").astype(float)

        if "bins" in defn and isinstance(defn["bins"], int):
            # Weighted quantile bins
            n_bins = defn["bins"]
            sorted_idx = np.argsort(values)
            cum_weight = np.cumsum(weights[sorted_idx])
            total_weight = cum_weight[-1]
            labels = np.empty(len(values), dtype=object)
            for b in range(n_bins):
                lo = b / n_bins * total_weight
                hi = (b + 1) / n_bins * total_weight
                mask_sorted = (cum_weight > lo) & (cum_weight <= hi)
                if b == 0:
                    mask_sorted[0] = True
                labels[sorted_idx[mask_sorted]] = defn["labels"][b]
            slices[name] = labels
        else:
            bins = defn["bins"]
            label_list = defn["labels"]
            digitised = np.digitize(values, bins) - 1
            digitised = np.clip(digitised, 0, len(label_list) - 1)
            slices[name] = np.array([label_list[d] for d in digitised])

    return slices


def _compute_metric_values(
    sim,
    time_period: str,
) -> dict[str, np.ndarray]:
    """Compute household-level metric values.

    Returns a dict mapping metric_name -> array of values, one per
    household.
    """
    result = {}
    for metric in METRICS:
        try:
            result[metric] = np.asarray(
                sim.calculate(metric, map_to="household"),
                dtype=float,
            )
        except Exception:
            logger.debug("Metric %s not available, skipping", metric)
    return result


def compute_influence_matrix(
    sim,
    time_period: str,
    reform_sim=None,
) -> pd.DataFrame:
    """Compute per-record influence across the reporting surface.

    Args:
        sim: a policyengine_uk Microsimulation (baseline).
        time_period: the period string (e.g. "2025").
        reform_sim: optional reform Microsimulation.  When provided
            the metric is the *change* between baseline and reform.

    Returns:
        DataFrame with shape (n_households, n_statistics) where each
        cell I[i,s] is the fractional influence of household i on
        statistic s.
    """
    weights = np.asarray(
        sim.calculate("household_weight", map_to="household"),
        dtype=float,
    )
    slices = _build_slice_assignments(sim, time_period)

    if reform_sim is not None:
        baseline_vals = _compute_metric_values(sim, time_period)
        reform_vals = _compute_metric_values(reform_sim, time_period)
        metric_values = {
            m: reform_vals[m] - baseline_vals[m]
            for m in baseline_vals
            if m in reform_vals
        }
    else:
        metric_values = _compute_metric_values(sim, time_period)

    records = []
    stat_names = []

    for metric_name, values in metric_values.items():
        for slice_name, labels in slices.items():
            unique_labels = np.unique(labels)
            for label in unique_labels:
                if label is None or (isinstance(label, float) and np.isnan(label)):
                    continue
                mask = labels == label
                weighted_total = np.sum(weights[mask] * values[mask])
                denom = max(abs(weighted_total), 1e-10)
                influence = np.abs(weights * values * mask) / denom
                records.append(influence)
                stat_names.append(f"{metric_name}/{slice_name}={label}")

    if not records:
        return pd.DataFrame()

    matrix = np.column_stack(records)
    return pd.DataFrame(matrix, columns=stat_names)


def find_high_influence_records(
    influence_matrix: pd.DataFrame,
    threshold: float = 0.05,
) -> pd.DataFrame:
    """Identify records exceeding the influence threshold.

    Args:
        influence_matrix: output of compute_influence_matrix.
        threshold: max allowable influence fraction (default 5%).

    Returns:
        DataFrame with columns:
          - record_idx: household index
          - max_influence: maximum influence across all statistics
          - worst_statistic: the statistic where influence is highest
          - n_violations: number of statistics exceeding threshold
    """
    if influence_matrix.empty:
        return pd.DataFrame(
            columns=[
                "record_idx",
                "max_influence",
                "worst_statistic",
                "n_violations",
            ]
        )

    max_influence = influence_matrix.max(axis=1)
    worst_stat_idx = influence_matrix.values.argmax(axis=1)
    worst_stat = influence_matrix.columns[worst_stat_idx]
    n_violations = (influence_matrix > threshold).sum(axis=1)

    flagged_mask = max_influence > threshold
    result = pd.DataFrame(
        {
            "record_idx": np.where(flagged_mask)[0],
            "max_influence": max_influence[flagged_mask].values,
            "worst_statistic": worst_stat[flagged_mask],
            "n_violations": n_violations[flagged_mask].values,
        }
    )
    return result.sort_values("max_influence", ascending=False).reset_index(drop=True)


def compute_kish_effective_sample_size(
    weights: np.ndarray,
    slice_mask: np.ndarray | None = None,
) -> float:
    """Compute Kish's effective sample size.

    n_eff = (sum w_i)^2 / sum(w_i^2)

    Args:
        weights: array of household weights.
        slice_mask: optional boolean mask to restrict to a subgroup.

    Returns:
        Effective sample size.
    """
    if slice_mask is not None:
        w = weights[slice_mask]
    else:
        w = weights
    w = w[w > 0]
    if len(w) == 0:
        return 0.0
    return float(np.sum(w) ** 2 / np.sum(w**2))


def generate_random_reforms(
    n_reforms: int = 50,
    seed: int = 42,
) -> list[dict]:
    """Generate random parameter perturbations for influence sampling.

    Each reform is a dict of parameter_path -> multiplier pairs.
    The reforms perturb tax rates and benefit amounts by +-20%.

    Args:
        n_reforms: number of reforms to generate.
        seed: random seed.

    Returns:
        List of reform specification dicts.
    """
    rng = np.random.default_rng(seed)

    # Parameters amenable to perturbation
    rate_params = [
        "gov.hmrc.income_tax.rates.uk[0].rate",
        "gov.hmrc.income_tax.rates.uk[1].rate",
        "gov.hmrc.income_tax.rates.uk[2].rate",
        "gov.hmrc.national_insurance.class_1.rates.employee.main.rate",
    ]
    amount_params = [
        "gov.hmrc.income_tax.allowances.personal_allowance.amount",
        "gov.dwp.universal_credit.elements.standard_allowance.amount.single.over_25",
        "gov.dwp.universal_credit.elements.child.amount.first",
    ]

    reforms = []
    for _ in range(n_reforms):
        reform = {}
        # Perturb 2-4 parameters per reform
        n_params = rng.integers(2, 5)
        all_params = rate_params + amount_params
        chosen = rng.choice(
            len(all_params),
            size=min(n_params, len(all_params)),
            replace=False,
        )
        for idx in chosen:
            param = all_params[idx]
            if param in rate_params:
                # Rates: multiply by 0.8-1.2
                reform[param] = float(rng.uniform(0.8, 1.2))
            else:
                # Amounts: multiply by 0.8-1.2
                reform[param] = float(rng.uniform(0.8, 1.2))
        reforms.append(reform)

    return reforms


def run_diagnostics(
    dataset,
    time_period: str = "2025",
    n_reforms: int = 50,
    threshold: float = 0.05,
    seed: int = 42,
) -> dict:
    """Run the full Phase 1 influence diagnostics.

    Args:
        dataset: a UKSingleYearDataset.
        time_period: calendar year as string.
        n_reforms: number of random reforms for influence sampling.
        threshold: max allowable influence fraction.
        seed: random seed.

    Returns:
        Dict with keys:
          - baseline_influence: DataFrame of influence matrix under
            current law
          - flagged_records: DataFrame of high-influence records
          - weight_stats: dict of weight distribution statistics
          - kish_by_slice: dict of Kish effective sample sizes
          - reform_influence_summary: DataFrame summarising influence
            across reforms
    """
    from policyengine_uk import Microsimulation

    sim = Microsimulation(dataset=dataset)
    sim.default_calculation_period = time_period

    weights = np.asarray(
        sim.calculate("household_weight", map_to="household"),
        dtype=float,
    )

    # Weight distribution statistics
    weight_stats = {
        "n_households": len(weights),
        "mean": float(np.mean(weights)),
        "median": float(np.median(weights)),
        "p90": float(np.percentile(weights, 90)),
        "p99": float(np.percentile(weights, 99)),
        "max": float(np.max(weights)),
        "min": float(np.min(weights[weights > 0])),
        "skewness": float(
            np.mean(((weights - np.mean(weights)) / np.std(weights)) ** 3)
        ),
    }

    # Baseline influence
    logger.info("Computing baseline influence matrix...")
    baseline_influence = compute_influence_matrix(sim, time_period)
    flagged = find_high_influence_records(baseline_influence, threshold)

    # Kish effective sample size by slice
    slices = _build_slice_assignments(sim, time_period)
    kish_by_slice = {"overall": compute_kish_effective_sample_size(weights)}
    for slice_name, labels in slices.items():
        for label in np.unique(labels):
            if label is None:
                continue
            mask = labels == label
            kish_by_slice[f"{slice_name}={label}"] = compute_kish_effective_sample_size(
                weights, mask
            )

    # Reform-level influence sampling
    reforms = generate_random_reforms(n_reforms, seed)
    reform_max_influences = []

    for i, reform_spec in enumerate(reforms):
        logger.info(
            "Computing influence for reform %d/%d...",
            i + 1,
            len(reforms),
        )
        try:
            reform_sim = _create_reform_sim(dataset, time_period, reform_spec)
            infl = compute_influence_matrix(sim, time_period, reform_sim=reform_sim)
            if not infl.empty:
                max_per_record = infl.max(axis=1)
                reform_max_influences.append(max_per_record)
        except Exception as e:
            logger.warning("Reform %d failed: %s", i, e)

    if reform_max_influences:
        reform_matrix = pd.concat(reform_max_influences, axis=1).fillna(0)
        reform_summary = pd.DataFrame(
            {
                "mean_max_influence": reform_matrix.mean(axis=1),
                "max_max_influence": reform_matrix.max(axis=1),
                "n_reforms_above_threshold": (reform_matrix > threshold).sum(axis=1),
            }
        )
    else:
        reform_summary = pd.DataFrame()

    return {
        "baseline_influence": baseline_influence,
        "flagged_records": flagged,
        "weight_stats": weight_stats,
        "kish_by_slice": kish_by_slice,
        "reform_influence_summary": reform_summary,
    }


def _create_reform_sim(dataset, time_period, reform_spec):
    """Create a Microsimulation with parameter perturbations applied."""
    from policyengine_uk import Microsimulation

    sim = Microsimulation(dataset=dataset)
    sim.default_calculation_period = time_period

    for param_path, multiplier in reform_spec.items():
        try:
            param = sim.tax_benefit_system.parameters.get_child(param_path)
            current = param(time_period)
            param.update(
                period=f"year:{time_period}:1",
                value=current * multiplier,
            )
        except Exception:
            pass

    sim.tax_benefit_system.reset_parameter_caches()
    return sim
