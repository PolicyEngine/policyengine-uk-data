"""Dagster resources for storage, database, and compute backends."""

from policyengine_uk_data.resources.bucket import BucketResource
from policyengine_uk_data.resources.database import DatabaseResource
from policyengine_uk_data.resources.compute import ComputeResource

__all__ = [
    "BucketResource",
    "DatabaseResource",
    "ComputeResource",
]
