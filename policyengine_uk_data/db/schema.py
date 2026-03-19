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

import sqlite3
from pathlib import Path

from policyengine_uk_data.storage import STORAGE_FOLDER

DB_PATH = STORAGE_FOLDER / "targets.db"

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS areas (
    code        TEXT PRIMARY KEY,
    name        TEXT,
    level       TEXT NOT NULL,  -- country, region, la, constituency, msoa, lsoa, oa
    parent_code TEXT,
    country     TEXT,
    FOREIGN KEY (parent_code) REFERENCES areas(code)
);

CREATE INDEX IF NOT EXISTS idx_areas_level ON areas(level);
CREATE INDEX IF NOT EXISTS idx_areas_parent ON areas(parent_code);
CREATE INDEX IF NOT EXISTS idx_areas_country ON areas(country);

CREATE TABLE IF NOT EXISTS targets (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    name                TEXT NOT NULL UNIQUE,
    variable            TEXT NOT NULL,
    source              TEXT NOT NULL,
    unit                TEXT NOT NULL,  -- gbp, count, rate
    geographic_level    TEXT NOT NULL,  -- national, country, region, constituency, local_authority
    geo_code            TEXT,
    geo_name            TEXT,
    breakdown_variable  TEXT,
    lower_bound         REAL,
    upper_bound         REAL,
    is_count            INTEGER NOT NULL DEFAULT 0,
    reference_url       TEXT,
    forecast_vintage    TEXT
);

CREATE INDEX IF NOT EXISTS idx_targets_variable ON targets(variable);
CREATE INDEX IF NOT EXISTS idx_targets_source ON targets(source);
CREATE INDEX IF NOT EXISTS idx_targets_geo_level ON targets(geographic_level);
CREATE INDEX IF NOT EXISTS idx_targets_geo_code ON targets(geo_code);

CREATE TABLE IF NOT EXISTS target_values (
    target_id   INTEGER NOT NULL,
    year        INTEGER NOT NULL,
    value       REAL NOT NULL,
    PRIMARY KEY (target_id, year),
    FOREIGN KEY (target_id) REFERENCES targets(id)
);

CREATE INDEX IF NOT EXISTS idx_target_values_year ON target_values(year);
"""


def get_connection(db_path: Path | None = None) -> sqlite3.Connection:
    """Open (or create) the target database and ensure schema exists."""
    path = db_path or DB_PATH
    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(_SCHEMA_SQL)
    return conn
