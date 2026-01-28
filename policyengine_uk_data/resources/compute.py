"""Compute resource for local and Modal execution."""

from __future__ import annotations

import os
from typing import Callable, Literal, Any

from dagster import ConfigurableResource
from pydantic import Field


class ComputeResource(ConfigurableResource):
    """Compute abstraction for local or Modal execution."""

    backend: Literal["local", "modal"] = Field(
        default="local",
        description="Compute backend: 'local' or 'modal'",
    )
    testing: bool = Field(
        default_factory=lambda: os.environ.get("TESTING", "0") == "1",
        description="Testing mode (reduced epochs)",
    )
    modal_gpu: str = Field(
        default="T4",
        description="GPU type for Modal (T4, A10G, A100, etc.)",
    )

    @property
    def epochs(self) -> int:
        return 32 if self.testing else 512

    def run(self, fn: Callable, *args, **kwargs) -> Any:
        """Run a function locally or on Modal."""
        if self.backend == "local":
            return fn(*args, **kwargs)
        else:
            return self._run_on_modal(fn, *args, **kwargs)

    def _run_on_modal(self, fn: Callable, *args, **kwargs) -> Any:
        """Execute function on Modal with GPU."""
        import modal

        app = modal.App("policyengine-uk-data")

        image = modal.Image.debian_slim(python_version="3.13").pip_install(
            "torch",
            "numpy",
            "pandas",
        )

        @app.function(gpu=self.modal_gpu, image=image, timeout=7200)
        def remote_fn(*args, **kwargs):
            return fn(*args, **kwargs)

        with app.run():
            return remote_fn.remote(*args, **kwargs)
