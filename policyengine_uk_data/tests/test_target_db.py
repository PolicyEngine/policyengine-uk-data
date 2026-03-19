"""Tests for the Phase 5 SQLite target database."""

import sqlite3
import tempfile
from pathlib import Path

import pytest

from policyengine_uk_data.db.schema import get_connection
from policyengine_uk_data.db.query import (
    get_area_children,
    get_area_hierarchy,
    get_area_targets,
    get_targets,
    count_areas_by_level,
    count_targets_by_source,
)


@pytest.fixture
def db_path(tmp_path):
    """Create an empty database in a temp directory."""
    return tmp_path / "test_targets.db"


@pytest.fixture
def conn(db_path):
    """Connection with schema created."""
    c = get_connection(db_path)
    yield c
    c.close()


class TestSchema:
    def test_tables_created(self, conn):
        tables = [
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        ]
        assert "areas" in tables
        assert "targets" in tables
        assert "target_values" in tables

    def test_idempotent_creation(self, db_path):
        """Creating connection twice doesn't error."""
        c1 = get_connection(db_path)
        c1.close()
        c2 = get_connection(db_path)
        c2.close()


class TestAreaHierarchy:
    def _insert_areas(self, conn):
        """Insert a small area hierarchy for testing."""
        conn.executemany(
            "INSERT INTO areas (code, name, level, parent_code, country) "
            "VALUES (?, ?, ?, ?, ?)",
            [
                ("E92000001", "England", "country", None, "England"),
                ("E12000001", "North East", "region", "E92000001", "England"),
                ("E06000001", "Hartlepool", "la", "E12000001", "England"),
                ("E14001063", "Aldershot", "constituency", "E92000001", "England"),
                ("E02001001", "MSOA1", "msoa", "E06000001", "England"),
                ("E01001001", "LSOA1", "lsoa", "E02001001", "England"),
                ("E00012345", "OA1", "oa", "E01001001", "England"),
            ],
        )
        conn.commit()

    def test_children(self, conn, db_path):
        self._insert_areas(conn)
        children = get_area_children("E92000001", db_path=db_path)
        # England's children: region + constituency (parallel branches)
        assert len(children) == 2
        levels = set(children["level"])
        assert "region" in levels
        assert "constituency" in levels

    def test_hierarchy_walk(self, conn, db_path):
        self._insert_areas(conn)
        hierarchy = get_area_hierarchy("E00012345", db_path=db_path)
        levels = [h["level"] for h in hierarchy]
        assert levels == ["oa", "lsoa", "msoa", "la", "region", "country"]

    def test_la_parents_to_region(self, conn, db_path):
        self._insert_areas(conn)
        hierarchy = get_area_hierarchy("E06000001", db_path=db_path)
        assert hierarchy[0]["level"] == "la"
        assert hierarchy[1]["level"] == "region"
        assert hierarchy[2]["level"] == "country"

    def test_constituency_parents_to_country(self, conn, db_path):
        self._insert_areas(conn)
        hierarchy = get_area_hierarchy("E14001063", db_path=db_path)
        assert hierarchy[0]["level"] == "constituency"
        assert hierarchy[1]["level"] == "country"

    def test_count_by_level(self, conn, db_path):
        self._insert_areas(conn)
        counts = count_areas_by_level(db_path=db_path)
        assert counts["country"] == 1
        assert counts["la"] == 1
        assert counts["oa"] == 1


class TestTargets:
    def _insert_targets(self, conn):
        """Insert test targets."""
        conn.execute(
            """INSERT INTO targets
               (name, variable, source, unit, geographic_level, geo_code, is_count)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            ("test/age_0_10", "age", "ons", "count", "constituency", "E14001063", 1),
        )
        target_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.execute(
            "INSERT INTO target_values (target_id, year, value) VALUES (?, ?, ?)",
            (target_id, 2025, 12345.0),
        )

        conn.execute(
            """INSERT INTO targets
               (name, variable, source, unit, geographic_level, geo_code, is_count)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                "test/income",
                "employment_income",
                "hmrc",
                "gbp",
                "local_authority",
                "E06000001",
                0,
            ),
        )
        target_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.executemany(
            "INSERT INTO target_values (target_id, year, value) VALUES (?, ?, ?)",
            [(target_id, 2024, 1e9), (target_id, 2025, 1.1e9)],
        )
        conn.commit()

    def test_query_by_geo_level(self, conn, db_path):
        self._insert_targets(conn)
        df = get_targets(geographic_level="constituency", db_path=db_path)
        assert len(df) == 1
        assert df.iloc[0]["variable"] == "age"

    def test_query_by_year(self, conn, db_path):
        self._insert_targets(conn)
        df = get_targets(year=2024, db_path=db_path)
        assert len(df) == 1
        assert df.iloc[0]["variable"] == "employment_income"

    def test_query_by_source(self, conn, db_path):
        self._insert_targets(conn)
        df = get_targets(source="hmrc", db_path=db_path)
        assert len(df) == 2  # Two years for the income target

    def test_area_targets(self, conn, db_path):
        self._insert_targets(conn)
        df = get_area_targets("E14001063", year=2025, db_path=db_path)
        assert len(df) == 1
        assert df.iloc[0]["value"] == 12345.0

    def test_count_by_source(self, conn, db_path):
        self._insert_targets(conn)
        counts = count_targets_by_source(db_path=db_path)
        assert counts["ons"] == 1
        assert counts["hmrc"] == 1
