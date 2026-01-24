"""PolicyEngine UK data pipeline.

This package builds representative microdata for UK tax-benefit modelling.
Orchestrated with Dagster for reproducibility and observability.

Usage:
    # Run Dagster UI
    dagster dev -m policyengine_uk_data.definitions

    # Or materialise assets programmatically
    from policyengine_uk_data.definitions import defs
"""

from policyengine_uk_data.definitions import defs

__all__ = ["defs"]
