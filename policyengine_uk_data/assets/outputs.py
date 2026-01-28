"""Final output dataset assets."""

from pathlib import Path

import numpy as np
from dagster import asset, AssetExecutionContext, Config
from policyengine_uk.data import UKSingleYearDataset
from pydantic import Field

from policyengine_uk_data.resources.bucket import BucketResource


class OutputConfig(Config):
    output_year: int = Field(default=2023, description="Final output year")


def _load_dataset(data: dict) -> UKSingleYearDataset:
    return UKSingleYearDataset(
        person=data["person"],
        benunit=data["benunit"],
        household=data["household"],
        fiscal_year=data.get("fiscal_year", 2025),
    )


@asset(group_name="outputs")
def enhanced_frs(
    context: AssetExecutionContext,
    config: OutputConfig,
    uprated_frs: dict,
    constituency_weights: np.ndarray,
    la_weights: np.ndarray,
    bucket: BucketResource,
) -> dict:
    """Final enhanced FRS dataset with calibrated weights."""
    from policyengine_uk_data.utils.uprating import uprate_dataset

    dataset = _load_dataset(uprated_frs)

    context.log.info("Creating final enhanced FRS dataset")

    # Apply calibrated national weights (sum across areas)
    national_weights = constituency_weights.sum(axis=0)
    dataset.household["household_weight"] = national_weights

    # Downrate to output year
    if dataset.time_period != config.output_year:
        context.log.info(
            f"Downrating from {dataset.time_period} to {config.output_year}"
        )
        dataset = uprate_dataset(dataset, config.output_year)

    # Save final output
    output_path = (
        Path(bucket.local_path) / f"output/enhanced_frs_{config.output_year}.h5"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.save(output_path)

    context.add_output_metadata({
        "output_year": config.output_year,
        "num_persons": len(dataset.person),
        "num_benuints": len(dataset.benunit),
        "num_households": len(dataset.household),
        "total_weight": float(dataset.household["household_weight"].sum()),
        "output_path": str(output_path),
    })

    return {
        "person": dataset.person,
        "benunit": dataset.benunit,
        "household": dataset.household,
        "fiscal_year": dataset.time_period,
    }
