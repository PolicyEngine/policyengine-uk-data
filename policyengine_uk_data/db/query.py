"""Query API for the target database.

Convenience functions for retrieving targets by geographic level,
area code, variable, source, and year.
"""

import sqlite3
from pathlib import Path
from typing import Optional

import pandas as pd

from policyengine_uk_data.db.schema import DB_PATH, get_connection


def _conn(db_path: Path | None = None) -> sqlite3.Connection:
    return get_connection(db_path)


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
    conn = _conn(db_path)
    clauses = []
    params = []

    if geographic_level:
        clauses.append("t.geographic_level = ?")
        params.append(geographic_level)
    if geo_code:
        clauses.append("t.geo_code = ?")
        params.append(geo_code)
    if variable:
        clauses.append("t.variable = ?")
        params.append(variable)
    if source:
        clauses.append("t.source = ?")
        params.append(source)
    if year:
        clauses.append("tv.year = ?")
        params.append(year)

    where = " AND ".join(clauses)
    if where:
        where = "WHERE " + where

    sql = f"""
        SELECT t.name, t.variable, t.source, t.unit,
               t.geographic_level, t.geo_code, t.geo_name,
               tv.year, tv.value
        FROM targets t
        JOIN target_values tv ON t.id = tv.target_id
        {where}
        ORDER BY t.geographic_level, t.geo_code, t.name, tv.year
    """
    df = pd.read_sql_query(sql, conn, params=params)
    conn.close()
    return df


def get_area_targets(
    geo_code: str,
    year: int = 2025,
    db_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Get all targets for a specific area and year.

    Args:
        geo_code: Area GSS code (e.g. "E14001063" for a constituency).
        year: Target year.
        db_path: Override database path.

    Returns:
        DataFrame with columns: name, variable, source, unit, value.
    """
    conn = _conn(db_path)
    sql = """
        SELECT t.name, t.variable, t.source, t.unit, tv.value
        FROM targets t
        JOIN target_values tv ON t.id = tv.target_id
        WHERE t.geo_code = ? AND tv.year = ?
        ORDER BY t.source, t.variable
    """
    df = pd.read_sql_query(sql, conn, params=[geo_code, year])
    conn.close()
    return df


def get_area_children(
    parent_code: str,
    db_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Get child areas of a given parent in the hierarchy.

    Args:
        parent_code: Parent area GSS code.

    Returns:
        DataFrame with columns: code, name, level, country.
    """
    conn = _conn(db_path)
    sql = """
        SELECT code, name, level, country
        FROM areas
        WHERE parent_code = ?
        ORDER BY code
    """
    df = pd.read_sql_query(sql, conn, params=[parent_code])
    conn.close()
    return df


def get_area_hierarchy(
    code: str,
    db_path: Optional[Path] = None,
) -> list[dict]:
    """Walk up the area hierarchy from a given code to the root.

    Returns a list of dicts from most specific to least:
    [{"code": "E00...", "level": "oa"}, {"code": "E01...", "level": "lsoa"}, ...].
    """
    conn = _conn(db_path)
    result = []
    current = code
    while current:
        row = conn.execute(
            "SELECT code, name, level, parent_code, country FROM areas WHERE code = ?",
            (current,),
        ).fetchone()
        if row is None:
            break
        result.append(
            {
                "code": row[0],
                "name": row[1],
                "level": row[2],
                "parent_code": row[3],
                "country": row[4],
            }
        )
        current = row[3]
    conn.close()
    return result


def count_areas_by_level(db_path: Optional[Path] = None) -> dict[str, int]:
    """Return area counts grouped by geographic level."""
    conn = _conn(db_path)
    rows = conn.execute(
        "SELECT level, COUNT(*) FROM areas GROUP BY level ORDER BY COUNT(*) DESC"
    ).fetchall()
    conn.close()
    return {level: count for level, count in rows}


def count_targets_by_source(db_path: Optional[Path] = None) -> dict[str, int]:
    """Return target counts grouped by source."""
    conn = _conn(db_path)
    rows = conn.execute(
        "SELECT source, COUNT(*) FROM targets GROUP BY source ORDER BY COUNT(*) DESC"
    ).fetchall()
    conn.close()
    return {source: count for source, count in rows}
