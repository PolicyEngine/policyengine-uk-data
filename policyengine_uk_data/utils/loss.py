"""Loss functions and target matrices for dataset calibration.

Delegates to the targets registry and build_loss_matrix module
for all target definitions and simulation column construction.
"""

import numpy as np
import pandas as pd

from policyengine_uk_data.targets.build_loss_matrix import (
    create_target_matrix,
)


def get_loss_results(
    dataset, time_period, reform=None, household_weights=None
):
    """Calculate loss metrics comparing model outputs to targets.

    Args:
        dataset: PolicyEngine UK dataset to evaluate.
        time_period: year for comparison.
        reform: policy reform to apply.
        household_weights: custom weights (uses dataset weights if None).

    Returns:
        DataFrame with estimate vs target comparisons and error metrics.
    """
    matrix, targets = create_target_matrix(dataset, time_period, reform)
    from policyengine_uk import Microsimulation

    if household_weights is None:
        weights = (
            Microsimulation(dataset=dataset, reform=reform)
            .calculate("household_weight", time_period)
            .values
        )
    else:
        weights = household_weights
    estimates = weights @ matrix
    df = pd.DataFrame(
        {
            "name": estimates.index,
            "estimate": estimates.values,
            "target": targets,
        },
    )
    df["error"] = df["estimate"] - df["target"]
    df["abs_error"] = df["error"].abs()
    df["rel_error"] = df["error"] / df["target"]
    df["abs_rel_error"] = df["rel_error"].abs()
    return df.reset_index(drop=True)
