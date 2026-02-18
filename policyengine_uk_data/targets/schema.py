"""Pydantic schema for calibration targets."""

from enum import Enum
from typing import Callable
from pydantic import BaseModel, Field


class GeographicLevel(str, Enum):
    NATIONAL = "national"
    COUNTRY = "country"
    REGION = "region"
    CONSTITUENCY = "constituency"
    LOCAL_AUTHORITY = "local_authority"


class Unit(str, Enum):
    GBP = "gbp"
    COUNT = "count"
    RATE = "rate"


class Target(BaseModel):
    """A single calibration target from an official statistical source.

    Each target represents one number that the microsimulation should
    reproduce when household weights are correctly calibrated, e.g.
    "total income tax receipts in 2025 = £328.4bn".
    """

    name: str
    variable: str
    source: str
    unit: Unit
    geographic_level: GeographicLevel = GeographicLevel.NATIONAL
    geo_code: str | None = None
    geo_name: str | None = None
    values: dict[int, float]
    breakdown_variable: str | None = None
    lower_bound: float | None = None
    upper_bound: float | None = None
    is_count: bool = False
    reference_url: str | None = None
    forecast_vintage: str | None = None

    # For targets needing custom simulation logic (UC splits,
    # counterfactuals). Excluded from serialisation.
    custom_compute: Callable | None = Field(default=None, exclude=True)

    model_config = {"arbitrary_types_allowed": True}
