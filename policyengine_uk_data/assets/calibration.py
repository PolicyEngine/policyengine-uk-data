"""Weight calibration assets using original calibration code with real targets."""

import os
from pathlib import Path

import h5py
import numpy as np
from dagster import asset, AssetExecutionContext, Config
from policyengine_uk.data import UKSingleYearDataset
from pydantic import Field

from policyengine_uk_data.resources.bucket import BucketResource
from policyengine_uk_data.resources.compute import ComputeResource
from policyengine_uk_data.storage import STORAGE_FOLDER


class CalibrationConfig(Config):
    epochs: int | None = Field(
        default=None, description="Override epochs (None uses compute default)"
    )


def _load_dataset(data: dict) -> UKSingleYearDataset:
    return UKSingleYearDataset(
        person=data["person"],
        benunit=data["benunit"],
        household=data["household"],
        fiscal_year=data.get("fiscal_year", 2025),
    )


@asset(group_name="calibration")
def constituency_weights(
    context: AssetExecutionContext,
    config: CalibrationConfig,
    uprated_frs: dict,
    compute: ComputeResource,
) -> np.ndarray:
    """Calibrated weights for 650 parliamentary constituencies."""
    from policyengine_uk_data.utils.calibrate import calibrate_local_areas
    from policyengine_uk_data.datasets.local_areas.constituencies.loss import (
        create_constituency_target_matrix,
        create_national_target_matrix,
    )
    from policyengine_uk_data.datasets.local_areas.constituencies.calibrate import (
        get_performance,
    )

    dataset = _load_dataset(uprated_frs)
    epochs = config.epochs or compute.epochs

    context.log.info(f"Calibrating constituency weights ({epochs} epochs)")

    weight_file = "parliamentary_constituency_weights.h5"
    calibrate_local_areas(
        dataset=dataset,
        epochs=epochs,
        matrix_fn=create_constituency_target_matrix,
        national_matrix_fn=create_national_target_matrix,
        area_count=650,
        weight_file=weight_file,
        excluded_training_targets=[],
        log_csv="constituency_calibration_log.csv",
        verbose=True,
        area_name="Constituency",
        get_performance=get_performance,
    )

    # Read weights from saved file
    with h5py.File(STORAGE_FOLDER / weight_file, "r") as f:
        weights = f["2025"][:]

    context.add_output_metadata({
        "shape": list(weights.shape),
        "epochs": epochs,
        "areas": 650,
    })

    return weights


@asset(group_name="calibration")
def la_weights(
    context: AssetExecutionContext,
    config: CalibrationConfig,
    uprated_frs: dict,
    compute: ComputeResource,
) -> np.ndarray:
    """Calibrated weights for 360 local authorities."""
    from policyengine_uk_data.utils.calibrate import calibrate_local_areas
    from policyengine_uk_data.datasets.local_areas.local_authorities.loss import (
        create_local_authority_target_matrix,
    )
    from policyengine_uk_data.datasets.local_areas.constituencies.loss import (
        create_national_target_matrix,
    )
    from policyengine_uk_data.datasets.local_areas.local_authorities.calibrate import (
        get_performance,
    )

    dataset = _load_dataset(uprated_frs)
    epochs = config.epochs or compute.epochs

    context.log.info(f"Calibrating local authority weights ({epochs} epochs)")

    weight_file = "local_authority_weights.h5"
    calibrate_local_areas(
        dataset=dataset,
        epochs=epochs,
        matrix_fn=create_local_authority_target_matrix,
        national_matrix_fn=create_national_target_matrix,
        area_count=360,
        weight_file=weight_file,
        excluded_training_targets=[],
        log_csv="la_calibration_log.csv",
        verbose=True,
        area_name="Local Authority",
        get_performance=get_performance,
    )

    # Read weights from saved file
    with h5py.File(STORAGE_FOLDER / weight_file, "r") as f:
        weights = f["2025"][:]

    context.add_output_metadata({
        "shape": list(weights.shape),
        "epochs": epochs,
        "areas": 360,
    })

    return weights
