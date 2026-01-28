"""PostgreSQL database resource for calibration targets."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Generator

import pandas as pd
from dagster import ConfigurableResource
from pydantic import Field
from sqlmodel import Session, create_engine, select

from policyengine_uk_data.models.targets import (
    CalibrationTarget,
    DemographicTarget,
    IncomeTarget,
    BenefitTarget,
    TaxTarget,
)


class DatabaseResource(ConfigurableResource):
    """PostgreSQL database resource for calibration targets."""

    connection_string: str = Field(
        description="PostgreSQL connection string",
    )

    def _engine(self):
        return create_engine(self.connection_string)

    @contextmanager
    def session(self) -> Generator[Session, None, None]:
        with Session(self._engine()) as s:
            yield s

    def get_targets(
        self,
        area_type: str,
        year: int,
        metric: str | None = None,
    ) -> pd.DataFrame:
        """Get calibration targets as DataFrame."""
        with self.session() as s:
            q = select(CalibrationTarget).where(
                CalibrationTarget.area_type == area_type,
                CalibrationTarget.year == year,
            )
            if metric:
                q = q.where(CalibrationTarget.metric == metric)
            results = s.exec(q).all()
            if not results:
                return pd.DataFrame(columns=["area_code", "metric", "value", "source"])
            return pd.DataFrame([
                {"area_code": r.area_code, "metric": r.metric, "value": r.value, "source": r.source}
                for r in results
            ])

    def get_demographic_targets(self, area_type: str, year: int) -> pd.DataFrame:
        with self.session() as s:
            results = s.exec(
                select(DemographicTarget).where(
                    DemographicTarget.area_type == area_type,
                    DemographicTarget.year == year,
                )
            ).all()
            return pd.DataFrame([r.model_dump() for r in results]) if results else pd.DataFrame()

    def get_income_targets(self, area_type: str, year: int) -> pd.DataFrame:
        with self.session() as s:
            results = s.exec(
                select(IncomeTarget).where(
                    IncomeTarget.area_type == area_type,
                    IncomeTarget.year == year,
                )
            ).all()
            return pd.DataFrame([r.model_dump() for r in results]) if results else pd.DataFrame()

    def get_benefit_targets(self, area_type: str, year: int) -> pd.DataFrame:
        with self.session() as s:
            results = s.exec(
                select(BenefitTarget).where(
                    BenefitTarget.area_type == area_type,
                    BenefitTarget.year == year,
                )
            ).all()
            return pd.DataFrame([r.model_dump() for r in results]) if results else pd.DataFrame()

    def get_tax_targets(self, year: int) -> pd.DataFrame:
        with self.session() as s:
            results = s.exec(select(TaxTarget).where(TaxTarget.year == year)).all()
            return pd.DataFrame([r.model_dump() for r in results]) if results else pd.DataFrame()

    def pivot_to_matrix(
        self,
        df: pd.DataFrame,
        area_codes: list[str],
        metrics: list[str],
    ) -> pd.DataFrame:
        """Pivot targets to matrix form for calibration."""
        pivoted = df.pivot(index="area_code", columns="metric", values="value")
        return pivoted.reindex(index=area_codes, columns=metrics)
