"""IO manager for calibration weight matrices."""

from __future__ import annotations

import numpy as np
import pandas as pd
from dagster import (
    ConfigurableIOManager,
    InputContext,
    OutputContext,
)

from policyengine_uk_data.resources.bucket import BucketResource


class WeightsIOManager(ConfigurableIOManager):
    """IO manager for calibration weight matrices.

    Stores weight matrices as HDF5 with shape (num_areas, num_households).
    """

    bucket: BucketResource

    def _get_path(self, context: OutputContext | InputContext) -> str:
        asset_key = context.asset_key.path
        return f"output/{asset_key[-1]}.h5"

    def handle_output(self, context: OutputContext, obj: np.ndarray | pd.DataFrame) -> None:
        path = self._get_path(context)

        if isinstance(obj, np.ndarray):
            df = pd.DataFrame(obj)
        else:
            df = obj

        self.bucket.write_h5(path, {"weights": df})

        context.add_output_metadata({
            "path": path,
            "shape": list(df.shape),
            "mean_weight": float(df.values.mean()),
            "total_weight": float(df.values.sum()),
        })

    def load_input(self, context: InputContext) -> np.ndarray:
        path = self._get_path(context)
        store = self.bucket.read_h5(path)
        try:
            return store["weights"].values
        finally:
            store.close()
