"""IO manager for UKSingleYearDataset HDF5 files."""

from __future__ import annotations

import tempfile
import os
from pathlib import Path

import pandas as pd
from dagster import (
    ConfigurableIOManager,
    InputContext,
    OutputContext,
)

from policyengine_uk_data.resources.bucket import BucketResource


class DatasetIOManager(ConfigurableIOManager):
    """IO manager for PolicyEngine dataset HDF5 files.

    Stores datasets as HDF5 with person, benunit, household tables.
    """

    bucket: BucketResource

    def _get_path(self, context: OutputContext | InputContext) -> str:
        """Get storage path from context."""
        asset_key = context.asset_key.path
        return f"intermediate/{'/'.join(asset_key)}.h5"

    def handle_output(self, context: OutputContext, obj: dict) -> None:
        """Write dataset tables to HDF5."""
        path = self._get_path(context)

        if isinstance(obj, dict):
            self.bucket.write_h5(path, obj)
            context.add_output_metadata({
                "path": path,
                "tables": list(obj.keys()),
                "rows": {k: len(v) for k, v in obj.items()},
            })
        else:
            raise TypeError(f"Expected dict of DataFrames, got {type(obj)}")

    def load_input(self, context: InputContext) -> dict[str, pd.DataFrame]:
        """Load dataset tables from HDF5."""
        path = self._get_path(context)
        store = self.bucket.read_h5(path)
        try:
            return {key.lstrip("/"): store[key] for key in store.keys()}
        finally:
            store.close()


class FinalDatasetIOManager(ConfigurableIOManager):
    """IO manager for final output datasets."""

    bucket: BucketResource

    def _get_path(self, context: OutputContext | InputContext) -> str:
        asset_key = context.asset_key.path
        return f"output/{asset_key[-1]}.h5"

    def handle_output(self, context: OutputContext, obj: dict) -> None:
        path = self._get_path(context)
        self.bucket.write_h5(path, obj)

        total_rows = sum(len(v) for v in obj.values())
        context.add_output_metadata({
            "path": path,
            "tables": list(obj.keys()),
            "total_rows": total_rows,
        })

    def load_input(self, context: InputContext) -> dict[str, pd.DataFrame]:
        path = self._get_path(context)
        store = self.bucket.read_h5(path)
        try:
            return {key.lstrip("/"): store[key] for key in store.keys()}
        finally:
            store.close()
