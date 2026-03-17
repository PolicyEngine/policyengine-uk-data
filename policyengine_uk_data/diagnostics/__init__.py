"""Adversarial weight regularisation for PolicyEngine UK.

Detects high-influence survey records, generates synthetic offspring
to diffuse their weight, and recalibrates to population targets with
entropy regularisation.
"""

from policyengine_uk_data.diagnostics.influence import (
    compute_influence_matrix,
    find_high_influence_records,
    compute_kish_effective_sample_size,
    run_diagnostics,
)
from policyengine_uk_data.diagnostics.generative_model import (
    train_generative_model,
    extract_household_features,
    validate_generative_model,
)
from policyengine_uk_data.diagnostics.offspring import (
    run_adversarial_loop,
)
from policyengine_uk_data.diagnostics.recalibrate import (
    recalibrate_with_regularisation,
    prune_zero_weight_records,
)

__all__ = [
    "compute_influence_matrix",
    "find_high_influence_records",
    "compute_kish_effective_sample_size",
    "run_diagnostics",
    "train_generative_model",
    "extract_household_features",
    "validate_generative_model",
    "run_adversarial_loop",
    "recalibrate_with_regularisation",
    "prune_zero_weight_records",
]
