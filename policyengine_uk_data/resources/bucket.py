"""Bucket resource for local and GCS storage."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Literal

import pandas as pd
from dagster import ConfigurableResource
from google.cloud import storage
from pydantic import Field


class BucketResource(ConfigurableResource):
    """Storage abstraction for local filesystem or GCS bucket."""

    backend: Literal["local", "gcs"] = Field(
        default="local",
        description="Storage backend: 'local' or 'gcs'",
    )
    local_path: str = Field(
        default="policyengine_uk_data/data",
        description="Local directory for file storage",
    )
    gcs_bucket: str | None = Field(
        default=None,
        description="GCS bucket name (required if backend='gcs')",
    )
    gcs_prefix: str = Field(
        default="",
        description="Prefix path within GCS bucket",
    )

    def _local(self, path: str) -> Path:
        return Path(self.local_path) / path

    def _gcs_path(self, path: str) -> str:
        return f"{self.gcs_prefix}/{path}" if self.gcs_prefix else path

    def _bucket(self) -> storage.Bucket:
        if not self.gcs_bucket:
            raise ValueError("gcs_bucket required for GCS backend")
        return storage.Client().bucket(self.gcs_bucket)

    def exists(self, path: str) -> bool:
        if self.backend == "local":
            return self._local(path).exists()
        return self._bucket().blob(self._gcs_path(path)).exists()

    def read(self, path: str) -> bytes:
        if self.backend == "local":
            return self._local(path).read_bytes()
        return self._bucket().blob(self._gcs_path(path)).download_as_bytes()

    def write(self, path: str, data: bytes) -> None:
        if self.backend == "local":
            p = self._local(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(data)
        else:
            self._bucket().blob(self._gcs_path(path)).upload_from_string(data)

    def read_csv(self, path: str, **kwargs) -> pd.DataFrame:
        if self.backend == "local":
            return pd.read_csv(self._local(path), **kwargs)
        import io
        return pd.read_csv(io.BytesIO(self.read(path)), **kwargs)

    def write_csv(self, path: str, df: pd.DataFrame, **kwargs) -> None:
        if self.backend == "local":
            p = self._local(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(p, **kwargs)
        else:
            self.write(path, df.to_csv(**kwargs).encode())

    def read_h5(self, path: str) -> pd.HDFStore:
        if self.backend == "local":
            return pd.HDFStore(str(self._local(path)), mode="r")
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
            tmp.write(self.read(path))
            return pd.HDFStore(tmp.name, mode="r")

    def write_h5(self, path: str, data: dict[str, pd.DataFrame]) -> None:
        if self.backend == "local":
            p = self._local(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            with pd.HDFStore(str(p), mode="w") as store:
                for key, df in data.items():
                    store.put(key, df)
        else:
            with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
                tmp_path = tmp.name
            with pd.HDFStore(tmp_path, mode="w") as store:
                for key, df in data.items():
                    store.put(key, df)
            with open(tmp_path, "rb") as f:
                self.write(path, f.read())
            os.unlink(tmp_path)

    def list(self, prefix: str = "") -> list[str]:
        if self.backend == "local":
            base = self._local(prefix)
            if not base.exists():
                return []
            return [
                str(p.relative_to(self._local("")))
                for p in base.rglob("*")
                if p.is_file()
            ]
        return [
            b.name for b in self._bucket().list_blobs(prefix=self._gcs_path(prefix))
        ]

    def delete(self, path: str) -> None:
        if self.backend == "local":
            p = self._local(path)
            if p.exists():
                p.unlink()
        else:
            self._bucket().blob(self._gcs_path(path)).delete()
