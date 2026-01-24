"""Dagster assets for the policyengine-uk-data pipeline."""

from policyengine_uk_data.assets.raw_surveys import (
    raw_frs,
    raw_was,
    raw_lcfs,
    raw_etb,
    raw_spi,
)
from policyengine_uk_data.assets.frs import base_frs
from policyengine_uk_data.assets.imputations import (
    frs_with_wealth,
    frs_with_consumption,
    frs_with_vat,
    frs_with_services,
    frs_with_income,
    frs_with_capital_gains,
    frs_with_salary_sacrifice,
    frs_with_student_loans,
    uprated_frs,
)
from policyengine_uk_data.assets.calibration import (
    constituency_weights,
    la_weights,
)
from policyengine_uk_data.assets.outputs import enhanced_frs
from policyengine_uk_data.assets.targets import targets_db

__all__ = [
    # Raw data
    "raw_frs",
    "raw_was",
    "raw_lcfs",
    "raw_etb",
    "raw_spi",
    # Dataset pipeline
    "base_frs",
    "frs_with_wealth",
    "frs_with_consumption",
    "frs_with_vat",
    "frs_with_services",
    "frs_with_income",
    "frs_with_capital_gains",
    "frs_with_salary_sacrifice",
    "frs_with_student_loans",
    "uprated_frs",
    # Calibration
    "constituency_weights",
    "la_weights",
    # Outputs
    "enhanced_frs",
    # Targets
    "targets_db",
]
