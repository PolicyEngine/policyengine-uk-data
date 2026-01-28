"""Base FRS dataset creation asset."""

from pathlib import Path

from dagster import asset, AssetExecutionContext, Config
from pydantic import Field

from policyengine_uk_data.resources.bucket import BucketResource


class FRSConfig(Config):
    year: int = Field(default=2023, description="FRS survey year")


@asset(group_name="datasets")
def base_frs(
    context: AssetExecutionContext,
    config: FRSConfig,
    bucket: BucketResource,
    raw_frs: dict,
) -> dict:
    """Base FRS dataset processed into PolicyEngine format."""
    from policyengine_uk_data.datasets.frs import create_frs

    context.log.info(f"Creating base FRS dataset for year {config.year}")

    # Get raw FRS path from bucket - raw_frs asset provides the path
    raw_path = raw_frs["path"]
    if bucket.backend == "local":
        raw_frs_folder = Path(bucket.local_path) / raw_path
    else:
        raise NotImplementedError("GCS backend not yet supported for base_frs")

    if not raw_frs_folder.exists():
        raise FileNotFoundError(
            f"Raw FRS folder {raw_frs_folder} does not exist. "
            f"Ensure raw survey data is available at {raw_path}"
        )

    dataset = create_frs(
        raw_frs_folder=raw_frs_folder,
        year=config.year,
    )

    # Save intermediate
    base_path = Path(bucket.local_path)
    output_path = base_path / f"intermediate/frs_{config.year}.h5"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.save(output_path)

    context.add_output_metadata({
        "year": config.year,
        "num_persons": len(dataset.person),
        "num_benunits": len(dataset.benunit),
        "num_households": len(dataset.household),
        "output_path": str(output_path),
    })

    return {
        "person": dataset.person,
        "benunit": dataset.benunit,
        "household": dataset.household,
        "fiscal_year": dataset.time_period,
    }
