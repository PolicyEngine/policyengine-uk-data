"""Dagster definitions entry point for policyengine-uk-data."""

import os
from pathlib import Path

from dagster import Definitions, define_asset_job, AssetSelection

from policyengine_uk_data.assets import (
    raw_frs,
    raw_was,
    raw_lcfs,
    raw_etb,
    raw_spi,
    base_frs,
    frs_with_wealth,
    frs_with_consumption,
    frs_with_vat,
    frs_with_services,
    frs_with_income,
    frs_with_capital_gains,
    frs_with_salary_sacrifice,
    frs_with_student_loans,
    uprated_frs,
    constituency_weights,
    la_weights,
    enhanced_frs,
    targets_areas,
    targets_metrics,
    obr_receipts_observations,
    dwp_benefit_observations,
    ons_demographics_observations,
    observations_from_official_stats,
    observations_council_tax,
    targets_db,
)
from policyengine_uk_data.assets.checks import all_checks
from policyengine_uk_data.resources.bucket import BucketResource
from policyengine_uk_data.resources.database import DatabaseResource
from policyengine_uk_data.resources.compute import ComputeResource


BASE_PATH = Path(__file__).parent

local_resources = {
    "bucket": BucketResource(
        backend="local",
        local_path=str(BASE_PATH / "data"),
    ),
    "database": DatabaseResource(
        connection_string=os.environ.get(
            "DATABASE_URL",
            "postgresql://localhost:5432/policyengine_uk_data",
        ),
    ),
    "compute": ComputeResource(
        backend="local",
        testing=os.environ.get("TESTING", "0") == "1",
    ),
}

production_resources = {
    "bucket": BucketResource(
        backend="gcs",
        gcs_bucket=os.environ.get("GCS_BUCKET", "policyengine-uk-data"),
        gcs_prefix="",
    ),
    "database": DatabaseResource(
        connection_string=os.environ.get(
            "DATABASE_URL",
            "postgresql://localhost:5432/policyengine_uk_data",
        ),
    ),
    "compute": ComputeResource(
        backend="modal",
        testing=False,
        modal_gpu="T4",
    ),
}

resources = (
    production_resources
    if os.environ.get("DAGSTER_ENV") == "production"
    else local_resources
)

all_assets = [
    # Raw data sources
    raw_frs,
    raw_was,
    raw_lcfs,
    raw_etb,
    raw_spi,
    # Dataset pipeline
    base_frs,
    frs_with_wealth,
    frs_with_consumption,
    frs_with_vat,
    frs_with_services,
    frs_with_income,
    frs_with_capital_gains,
    frs_with_salary_sacrifice,
    frs_with_student_loans,
    uprated_frs,
    # Calibration
    constituency_weights,
    la_weights,
    # Final output
    enhanced_frs,
    # Targets database
    targets_areas,
    targets_metrics,
    obr_receipts_observations,
    dwp_benefit_observations,
    ons_demographics_observations,
    observations_from_official_stats,
    observations_council_tax,
    targets_db,
]

# Jobs for easy materialization from the UI
targets_job = define_asset_job(
    name="materialize_targets",
    selection=AssetSelection.groups("targets"),
    description="Materialize all calibration targets (OBR, DWP, ONS data)",
)

defs = Definitions(
    assets=all_assets,
    asset_checks=all_checks,
    jobs=[targets_job],
    resources=resources,
)
