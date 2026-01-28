"""SQLModel schemas for calibration targets with bitemporal tracking."""

from datetime import date, datetime
from typing import Optional

from sqlmodel import Field, SQLModel, Relationship


class Area(SQLModel, table=True):
    """Geographic area for statistics aggregation.

    Supports hierarchical structure: UK → country → region → constituency.
    """
    __tablename__ = "areas"

    code: str = Field(primary_key=True)  # E14000530, SCT, UK, etc.
    name: str
    area_type: str = Field(index=True)  # uk, country, region, constituency
    parent_code: str | None = Field(default=None, index=True)

    # Optional geometry for mapping (GeoJSON string)
    geometry: str | None = None


class Metric(SQLModel, table=True):
    """Definition of a measurable statistic.

    Separates the definition (what is income_tax?) from observations (values).
    """
    __tablename__ = "metrics"

    code: str = Field(primary_key=True)  # income_tax, population, etc.
    name: str  # Human-readable: "Income tax revenue"
    category: str = Field(index=True)  # obr, dwp, ons, hmrc, voa
    unit: str = Field(default="count")  # gbp, count, rate
    description: str | None = None


class Observation(SQLModel, table=True):
    """A single observation with bitemporal tracking.

    Two time dimensions:
    - valid_year: The year this statistic applies to (2026 = tax year 2026-27)
    - snapshot_date: When this information was published/known

    Example: OBR March 2024 forecast for 2026 income tax:
        metric_code="income_tax", area_code="UK", valid_year=2026,
        snapshot_date=date(2024, 3, 6), value=268.5e9, is_forecast=True

    When OBR October 2024 revises this forecast:
        Same metric/area/valid_year but snapshot_date=date(2024, 10, 30)

    Query "latest as of X" = most recent snapshot_date <= X.
    """
    __tablename__ = "observations"

    id: int | None = Field(default=None, primary_key=True)
    metric_code: str = Field(index=True)
    area_code: str = Field(index=True, default="UK")
    valid_year: int = Field(index=True)
    snapshot_date: date = Field(index=True)
    value: float
    source: str  # "OBR March 2024 EFO", "HMRC 2023-24 actuals"
    source_url: str | None = None
    is_forecast: bool = Field(default=False)
    created_at: datetime = Field(default_factory=datetime.utcnow)

    def display_value(self) -> str:
        """Format value for display."""
        v = self.value
        if abs(v) >= 1e9:
            return f"£{v/1e9:.1f}bn"
        elif abs(v) >= 1e6:
            return f"{v/1e6:.1f}m"
        elif abs(v) >= 1e3:
            return f"{v/1e3:.0f}k"
        else:
            return f"{v:,.0f}"
