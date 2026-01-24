"""Raw survey data source assets."""

from pathlib import Path

from dagster import asset, AssetExecutionContext

from policyengine_uk_data.resources.bucket import BucketResource


@asset(group_name="raw_data")
def raw_frs(context: AssetExecutionContext, bucket: BucketResource) -> dict:
    """Raw Family Resources Survey data."""
    path = "raw/frs_2023_24"
    files = bucket.list(path)

    context.add_output_metadata({
        "path": path,
        "file_count": len(files),
        "files": files[:10],  # First 10 files
    })

    return {"path": path, "files": files}


@asset(group_name="raw_data")
def raw_was(context: AssetExecutionContext, bucket: BucketResource) -> dict:
    """Raw Wealth and Assets Survey data."""
    path = "raw/was_2006_20"
    files = bucket.list(path)

    context.add_output_metadata({
        "path": path,
        "file_count": len(files),
    })

    return {"path": path, "files": files}


@asset(group_name="raw_data")
def raw_lcfs(context: AssetExecutionContext, bucket: BucketResource) -> dict:
    """Raw Living Costs and Food Survey data."""
    path = "raw/lcfs_2021_22"
    files = bucket.list(path)

    context.add_output_metadata({
        "path": path,
        "file_count": len(files),
    })

    return {"path": path, "files": files}


@asset(group_name="raw_data")
def raw_etb(context: AssetExecutionContext, bucket: BucketResource) -> dict:
    """Raw Effects of Taxes and Benefits data."""
    path = "raw/etb_1977_21"
    files = bucket.list(path)

    context.add_output_metadata({
        "path": path,
        "file_count": len(files),
    })

    return {"path": path, "files": files}


@asset(group_name="raw_data")
def raw_spi(context: AssetExecutionContext, bucket: BucketResource) -> dict:
    """Raw Survey of Personal Incomes data."""
    path = "raw/spi_2020_21"
    files = bucket.list(path)

    context.add_output_metadata({
        "path": path,
        "file_count": len(files),
    })

    return {"path": path, "files": files}
