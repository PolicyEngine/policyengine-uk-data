"""SQLite database for calibration targets with bitemporal querying."""

from datetime import date
from pathlib import Path

import pandas as pd
from sqlmodel import Session, SQLModel, create_engine, select, func, and_

from policyengine_uk_data.targets.models import Area, Metric, Observation

DB_PATH = Path(__file__).parent / "targets.db"


class TargetsDB:
    """Query interface for bitemporal calibration targets.

    Key concepts:
    - valid_year: The year a statistic applies to
    - snapshot_date: When this information was published (as-of time)

    Example queries:
        db.get_latest("income_tax", "UK", 2026, as_of=date(2024, 3, 15))
        db.get_trajectory("income_tax", "UK", as_of=date(2024, 3, 15))
        db.get_revision_history("income_tax", "UK", 2026)
    """

    def __init__(self, db_path: Path | str | None = None):
        self.db_path = Path(db_path) if db_path else DB_PATH
        self.engine = create_engine(f"sqlite:///{self.db_path}")
        SQLModel.metadata.create_all(self.engine)

    # === Area operations ===

    def add_area(self, area: Area) -> None:
        with Session(self.engine) as session:
            session.merge(area)
            session.commit()

    def get_area(self, code: str) -> Area | None:
        with Session(self.engine) as session:
            return session.get(Area, code)

    def list_areas(self, area_type: str | None = None) -> list[Area]:
        with Session(self.engine) as session:
            stmt = select(Area)
            if area_type:
                stmt = stmt.where(Area.area_type == area_type)
            return list(session.exec(stmt).all())

    def get_children(self, parent_code: str) -> list[Area]:
        with Session(self.engine) as session:
            return list(session.exec(
                select(Area).where(Area.parent_code == parent_code)
            ).all())

    # === Metric operations ===

    def add_metric(self, metric: Metric) -> None:
        with Session(self.engine) as session:
            session.merge(metric)
            session.commit()

    def get_metric(self, code: str) -> Metric | None:
        with Session(self.engine) as session:
            return session.get(Metric, code)

    def list_metrics(self, category: str | None = None) -> list[Metric]:
        with Session(self.engine) as session:
            stmt = select(Metric)
            if category:
                stmt = stmt.where(Metric.category == category)
            return list(session.exec(stmt).all())

    # === Observation operations ===

    def add_observation(self, obs: Observation) -> None:
        with Session(self.engine) as session:
            session.add(obs)
            session.commit()

    def bulk_add_observations(self, observations: list[Observation]) -> None:
        with Session(self.engine) as session:
            session.add_all(observations)
            session.commit()

    def get_latest(
        self,
        metric_code: str,
        area_code: str = "UK",
        valid_year: int | None = None,
        as_of: date | None = None,
    ) -> Observation | None:
        """Get the most recent observation for a metric/area/year.

        If as_of is provided, returns the latest snapshot before that date.
        If valid_year is None, returns the latest observation for any year.
        """
        as_of = as_of or date.today()
        with Session(self.engine) as session:
            stmt = (
                select(Observation)
                .where(
                    Observation.metric_code == metric_code,
                    Observation.area_code == area_code,
                    Observation.snapshot_date <= as_of,
                )
                .order_by(Observation.snapshot_date.desc())
            )
            if valid_year:
                stmt = stmt.where(Observation.valid_year == valid_year)
            return session.exec(stmt).first()

    def get_value(
        self,
        metric_code: str,
        area_code: str = "UK",
        valid_year: int | None = None,
        as_of: date | None = None,
    ) -> float | None:
        """Convenience method to get just the value."""
        obs = self.get_latest(metric_code, area_code, valid_year, as_of)
        return obs.value if obs else None

    def get_trajectory(
        self,
        metric_code: str,
        area_code: str = "UK",
        as_of: date | None = None,
    ) -> dict[int, float]:
        """Get forecast trajectory (all years) as of a snapshot date.

        Returns dict mapping valid_year -> value for the latest observation
        of each year as of the snapshot date.
        """
        as_of = as_of or date.today()
        with Session(self.engine) as session:
            # Get all observations up to as_of date
            stmt = (
                select(Observation)
                .where(
                    Observation.metric_code == metric_code,
                    Observation.area_code == area_code,
                    Observation.snapshot_date <= as_of,
                )
                .order_by(
                    Observation.valid_year,
                    Observation.snapshot_date.desc()
                )
            )
            observations = session.exec(stmt).all()

            # Take latest snapshot for each year
            trajectory = {}
            for obs in observations:
                if obs.valid_year not in trajectory:
                    trajectory[obs.valid_year] = obs.value

            return dict(sorted(trajectory.items()))

    def get_revision_history(
        self,
        metric_code: str,
        area_code: str = "UK",
        valid_year: int | None = None,
    ) -> list[Observation]:
        """Get all revisions for a metric/area/year, ordered by snapshot date.

        Useful for seeing how forecasts changed over time.
        """
        with Session(self.engine) as session:
            stmt = (
                select(Observation)
                .where(
                    Observation.metric_code == metric_code,
                    Observation.area_code == area_code,
                )
                .order_by(Observation.snapshot_date)
            )
            if valid_year:
                stmt = stmt.where(Observation.valid_year == valid_year)
            return list(session.exec(stmt).all())

    def query_observations(
        self,
        metric_code: str | None = None,
        area_code: str | None = None,
        valid_year: int | None = None,
        category: str | None = None,
        is_forecast: bool | None = None,
    ) -> list[Observation]:
        """Flexible query with optional filters."""
        with Session(self.engine) as session:
            stmt = select(Observation)
            if metric_code:
                stmt = stmt.where(Observation.metric_code == metric_code)
            if area_code:
                stmt = stmt.where(Observation.area_code == area_code)
            if valid_year:
                stmt = stmt.where(Observation.valid_year == valid_year)
            if is_forecast is not None:
                stmt = stmt.where(Observation.is_forecast == is_forecast)
            if category:
                # Join with metrics to filter by category
                stmt = stmt.join(
                    Metric, Observation.metric_code == Metric.code
                ).where(Metric.category == category)
            return list(session.exec(stmt).all())

    def to_dataframe(
        self,
        metric_code: str | None = None,
        area_code: str | None = None,
        as_of: date | None = None,
    ) -> pd.DataFrame:
        """Export observations as a DataFrame."""
        as_of = as_of or date.today()
        with Session(self.engine) as session:
            stmt = select(Observation).where(
                Observation.snapshot_date <= as_of
            )
            if metric_code:
                stmt = stmt.where(Observation.metric_code == metric_code)
            if area_code:
                stmt = stmt.where(Observation.area_code == area_code)

            observations = session.exec(stmt).all()
            if not observations:
                return pd.DataFrame()

            return pd.DataFrame([
                {
                    "metric": o.metric_code,
                    "area": o.area_code,
                    "year": o.valid_year,
                    "snapshot": o.snapshot_date,
                    "value": o.value,
                    "is_forecast": o.is_forecast,
                    "source": o.source,
                }
                for o in observations
            ])

    def clear(self) -> None:
        """Remove all data (for reseeding)."""
        with Session(self.engine) as session:
            for obs in session.exec(select(Observation)).all():
                session.delete(obs)
            for metric in session.exec(select(Metric)).all():
                session.delete(metric)
            for area in session.exec(select(Area)).all():
                session.delete(area)
            session.commit()

    def stats(self) -> dict:
        """Get summary statistics about the database."""
        with Session(self.engine) as session:
            areas = session.exec(select(Area)).all()
            metrics = session.exec(select(Metric)).all()
            observations = session.exec(select(Observation)).all()

            if not observations:
                return {"observations": 0, "metrics": 0, "areas": 0}

            years = set(o.valid_year for o in observations)
            snapshots = set(o.snapshot_date for o in observations)
            categories = set(m.category for m in metrics)

            return {
                "observations": len(observations),
                "metrics": len(metrics),
                "areas": len(areas),
                "categories": sorted(categories),
                "valid_years": sorted(years),
                "snapshot_dates": sorted(snapshots),
                "area_types": sorted(set(a.area_type for a in areas)),
            }
