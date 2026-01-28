"""SQLModel database models for calibration targets."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlmodel import Field, SQLModel


class CalibrationTarget(SQLModel, table=True):
    """Generic calibration target for any metric."""

    __tablename__ = "calibration_targets"

    id: int | None = Field(default=None, primary_key=True)
    area_type: str = Field(index=True)  # constituency, local_authority, national
    area_code: str = Field(index=True)  # E14000530, E09000001, etc.
    year: int = Field(index=True)
    metric: str = Field(index=True)
    value: float
    source: str
    source_url: str | None = None
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class DemographicTarget(SQLModel, table=True):
    """Population demographics by area."""

    __tablename__ = "demographic_targets"

    id: int | None = Field(default=None, primary_key=True)
    area_type: str = Field(index=True)
    area_code: str = Field(index=True)
    year: int = Field(index=True)
    age_band: str  # 0-15, 16-64, 65+, etc.
    sex: str | None = None  # male, female, all
    population: float
    source: str
    source_url: str | None = None
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class IncomeTarget(SQLModel, table=True):
    """Income distribution targets from HMRC SPI."""

    __tablename__ = "income_targets"

    id: int | None = Field(default=None, primary_key=True)
    area_type: str = Field(index=True)
    area_code: str = Field(index=True)
    year: int = Field(index=True)
    income_type: str  # employment, self_employment, pension, savings, dividends
    percentile: int | None = None  # for distribution targets
    total: float | None = None  # aggregate
    mean: float | None = None
    median: float | None = None
    source: str
    source_url: str | None = None
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class BenefitTarget(SQLModel, table=True):
    """Benefit caseload targets from DWP."""

    __tablename__ = "benefit_targets"

    id: int | None = Field(default=None, primary_key=True)
    area_type: str = Field(index=True)
    area_code: str = Field(index=True)
    year: int = Field(index=True)
    benefit: str  # universal_credit, housing_benefit, etc.
    caseload: float | None = None
    expenditure: float | None = None
    source: str
    source_url: str | None = None
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class TaxTarget(SQLModel, table=True):
    """Tax aggregate targets from OBR."""

    __tablename__ = "tax_targets"

    id: int | None = Field(default=None, primary_key=True)
    year: int = Field(index=True)
    tax: str  # income_tax, ni, vat, council_tax, etc.
    revenue: float | None = None
    taxpayers: float | None = None
    source: str
    source_url: str | None = None
    updated_at: datetime = Field(default_factory=datetime.utcnow)


def create_tables(engine) -> None:
    """Create all tables."""
    SQLModel.metadata.create_all(engine)
