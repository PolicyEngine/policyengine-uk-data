"""SQLite schema for the hierarchical target database.

Tables:
    areas       Geographic hierarchy from country down to Output Area.
    targets     One row per calibration target definition.
    target_values   One row per (target, year) pair.

The ``areas`` table encodes two parallel geographic branches via
``parent_code`` foreign keys:

    Administrative:  country → region → LA → MSOA → LSOA → OA
    Parliamentary:   country → constituency

LA and constituency are parallel — a constituency can span multiple
LAs and vice versa. OAs map to both via the crosswalk, but the
parent_code chain follows the administrative branch only.
"""

from pathlib import Path
from typing import Optional

from sqlmodel import Field, Session, SQLModel, create_engine

from policyengine_uk_data.storage import STORAGE_FOLDER

DB_PATH = STORAGE_FOLDER / "targets.db"


class Area(SQLModel, table=True):
    __tablename__ = "areas"

    code: str = Field(primary_key=True)
    name: Optional[str] = None
    level: str = Field(index=True)
    parent_code: Optional[str] = Field(
        default=None, foreign_key="areas.code", index=True
    )
    country: Optional[str] = Field(default=None, index=True)


class Target(SQLModel, table=True):
    __tablename__ = "targets"

    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(unique=True)
    variable: str = Field(index=True)
    source: str = Field(index=True)
    unit: str
    geographic_level: str = Field(index=True)
    geo_code: Optional[str] = Field(default=None, index=True)
    geo_name: Optional[str] = None
    breakdown_variable: Optional[str] = None
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None
    is_count: int = 0
    reference_url: Optional[str] = None
    forecast_vintage: Optional[str] = None


class TargetValue(SQLModel, table=True):
    __tablename__ = "target_values"

    target_id: int = Field(foreign_key="targets.id", primary_key=True)
    year: int = Field(primary_key=True, index=True)
    value: float


def get_engine(db_path: Path | None = None):
    """Create a SQLAlchemy engine for the target database."""
    path = db_path or DB_PATH
    engine = create_engine(f"sqlite:///{path}", echo=False)
    SQLModel.metadata.create_all(engine)
    return engine


def get_session(db_path: Path | None = None) -> Session:
    """Open a session with schema auto-created."""
    return Session(get_engine(db_path))


# Backward-compatible helper used by etl.py and tests
def get_connection(db_path: Path | None = None):
    """Return a raw sqlite3 connection with schema created.

    Prefer get_session() for new code.
    """
    import sqlite3

    path = db_path or DB_PATH
    # Ensure schema exists via SQLModel
    engine = get_engine(path)
    engine.dispose()
    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn
