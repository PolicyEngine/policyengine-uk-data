"""Query API for the target database.

Convenience functions for retrieving targets by geographic level,
area code, variable, source, and year.
"""

from pathlib import Path
from typing import Optional

import pandas as pd
from sqlmodel import Session, col, select

from policyengine_uk_data.db.schema import (
    Area,
    Target,
    TargetValue,
    get_session,
)


def get_targets(
    *,
    geographic_level: Optional[str] = None,
    geo_code: Optional[str] = None,
    variable: Optional[str] = None,
    source: Optional[str] = None,
    year: Optional[int] = None,
    db_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Query targets with optional filters.

    Returns a DataFrame with columns: name, variable, source, unit,
    geographic_level, geo_code, geo_name, year, value.
    """
    with get_session(db_path) as session:
        stmt = select(
            Target.name,
            Target.variable,
            Target.source,
            Target.unit,
            Target.geographic_level,
            Target.geo_code,
            Target.geo_name,
            TargetValue.year,
            TargetValue.value,
        ).join(TargetValue, col(Target.id) == col(TargetValue.target_id))

        if geographic_level:
            stmt = stmt.where(col(Target.geographic_level) == geographic_level)
        if geo_code:
            stmt = stmt.where(col(Target.geo_code) == geo_code)
        if variable:
            stmt = stmt.where(col(Target.variable) == variable)
        if source:
            stmt = stmt.where(col(Target.source) == source)
        if year:
            stmt = stmt.where(col(TargetValue.year) == year)

        stmt = stmt.order_by(
            col(Target.geographic_level),
            col(Target.geo_code),
            col(Target.name),
            col(TargetValue.year),
        )

        rows = session.exec(stmt).all()
        columns = [
            "name",
            "variable",
            "source",
            "unit",
            "geographic_level",
            "geo_code",
            "geo_name",
            "year",
            "value",
        ]
        return pd.DataFrame(rows, columns=columns)


def get_area_targets(
    geo_code: str,
    year: int = 2025,
    db_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Get all targets for a specific area and year."""
    with get_session(db_path) as session:
        stmt = (
            select(
                Target.name,
                Target.variable,
                Target.source,
                Target.unit,
                TargetValue.value,
            )
            .join(TargetValue, col(Target.id) == col(TargetValue.target_id))
            .where(col(Target.geo_code) == geo_code)
            .where(col(TargetValue.year) == year)
            .order_by(col(Target.source), col(Target.variable))
        )
        rows = session.exec(stmt).all()
        return pd.DataFrame(
            rows, columns=["name", "variable", "source", "unit", "value"]
        )


def get_area_children(
    parent_code: str,
    db_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Get child areas of a given parent in the hierarchy."""
    with get_session(db_path) as session:
        stmt = (
            select(Area.code, Area.name, Area.level, Area.country)
            .where(col(Area.parent_code) == parent_code)
            .order_by(col(Area.code))
        )
        rows = session.exec(stmt).all()
        return pd.DataFrame(rows, columns=["code", "name", "level", "country"])


def get_area_hierarchy(
    code: str,
    db_path: Optional[Path] = None,
) -> list[dict]:
    """Walk up the area hierarchy from a given code to the root.

    Returns a list of dicts from most specific to least:
    [{"code": "E00...", "level": "oa"}, ...].
    """
    with get_session(db_path) as session:
        result = []
        current = code
        while current:
            area = session.get(Area, current)
            if area is None:
                break
            result.append(
                {
                    "code": area.code,
                    "name": area.name,
                    "level": area.level,
                    "parent_code": area.parent_code,
                    "country": area.country,
                }
            )
            current = area.parent_code
        return result


def count_areas_by_level(db_path: Optional[Path] = None) -> dict[str, int]:
    """Return area counts grouped by geographic level."""
    with get_session(db_path) as session:
        from sqlalchemy import func

        stmt = (
            select(Area.level, func.count())
            .group_by(col(Area.level))
            .order_by(func.count().desc())
        )
        rows = session.exec(stmt).all()
        return {level: count for level, count in rows}


def count_targets_by_source(db_path: Optional[Path] = None) -> dict[str, int]:
    """Return target counts grouped by source."""
    with get_session(db_path) as session:
        from sqlalchemy import func

        stmt = (
            select(Target.source, func.count())
            .group_by(col(Target.source))
            .order_by(func.count().desc())
        )
        rows = session.exec(stmt).all()
        return {source: count for source, count in rows}
