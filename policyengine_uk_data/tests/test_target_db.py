"""Tests for the Phase 5 SQLite target database."""

import tempfile
from pathlib import Path

import pytest
from sqlmodel import Session, inspect

from policyengine_uk_data.db.schema import (
    Area,
    Target,
    TargetValue,
    get_engine,
    get_connection,
)
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
def session(db_path):
    """Session with schema created."""
    engine = get_engine(db_path)
    with Session(engine) as s:
        yield s


class TestSchema:
    def test_tables_created(self, db_path):
        engine = get_engine(db_path)
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        assert "areas" in tables
        assert "targets" in tables
        assert "target_values" in tables

    def test_idempotent_creation(self, db_path):
        """Creating engine twice doesn't error."""
        get_engine(db_path)
        get_engine(db_path)


class TestAreaHierarchy:
    def _insert_areas(self, session):
        """Insert a small area hierarchy for testing."""
        areas = [
            Area(
                code="E92000001",
                name="England",
                level="country",
                country="England",
            ),
            Area(
                code="E12000001",
                name="North East",
                level="region",
                parent_code="E92000001",
                country="England",
            ),
            Area(
                code="E06000001",
                name="Hartlepool",
                level="la",
                parent_code="E12000001",
                country="England",
            ),
            Area(
                code="E14001063",
                name="Aldershot",
                level="constituency",
                parent_code="E92000001",
                country="England",
            ),
            Area(
                code="E02001001",
                name="MSOA1",
                level="msoa",
                parent_code="E06000001",
                country="England",
            ),
            Area(
                code="E01001001",
                name="LSOA1",
                level="lsoa",
                parent_code="E02001001",
                country="England",
            ),
            Area(
                code="E00012345",
                name="OA1",
                level="oa",
                parent_code="E01001001",
                country="England",
            ),
        ]
        for area in areas:
            session.add(area)
        session.commit()

    def test_children(self, session, db_path):
        self._insert_areas(session)
        children = get_area_children("E92000001", db_path=db_path)
        assert len(children) == 2
        levels = set(children["level"])
        assert "region" in levels
        assert "constituency" in levels

    def test_hierarchy_walk(self, session, db_path):
        self._insert_areas(session)
        hierarchy = get_area_hierarchy("E00012345", db_path=db_path)
        levels = [h["level"] for h in hierarchy]
        assert levels == ["oa", "lsoa", "msoa", "la", "region", "country"]

    def test_la_parents_to_region(self, session, db_path):
        self._insert_areas(session)
        hierarchy = get_area_hierarchy("E06000001", db_path=db_path)
        assert hierarchy[0]["level"] == "la"
        assert hierarchy[1]["level"] == "region"
        assert hierarchy[2]["level"] == "country"

    def test_constituency_parents_to_country(self, session, db_path):
        self._insert_areas(session)
        hierarchy = get_area_hierarchy("E14001063", db_path=db_path)
        assert hierarchy[0]["level"] == "constituency"
        assert hierarchy[1]["level"] == "country"

    def test_count_by_level(self, session, db_path):
        self._insert_areas(session)
        counts = count_areas_by_level(db_path=db_path)
        assert counts["country"] == 1
        assert counts["la"] == 1
        assert counts["oa"] == 1


class TestTargets:
    def _insert_targets(self, session):
        """Insert test targets."""
        t1 = Target(
            name="test/age_0_10",
            variable="age",
            source="ons",
            unit="count",
            geographic_level="constituency",
            geo_code="E14001063",
            is_count=1,
        )
        session.add(t1)
        session.flush()
        session.add(TargetValue(target_id=t1.id, year=2025, value=12345.0))

        t2 = Target(
            name="test/income",
            variable="employment_income",
            source="hmrc",
            unit="gbp",
            geographic_level="local_authority",
            geo_code="E06000001",
            is_count=0,
        )
        session.add(t2)
        session.flush()
        session.add(TargetValue(target_id=t2.id, year=2024, value=1e9))
        session.add(TargetValue(target_id=t2.id, year=2025, value=1.1e9))
        session.commit()

    def test_query_by_geo_level(self, session, db_path):
        self._insert_targets(session)
        df = get_targets(geographic_level="constituency", db_path=db_path)
        assert len(df) == 1
        assert df.iloc[0]["variable"] == "age"

    def test_query_by_year(self, session, db_path):
        self._insert_targets(session)
        df = get_targets(year=2024, db_path=db_path)
        assert len(df) == 1
        assert df.iloc[0]["variable"] == "employment_income"

    def test_query_by_source(self, session, db_path):
        self._insert_targets(session)
        df = get_targets(source="hmrc", db_path=db_path)
        assert len(df) == 2

    def test_area_targets(self, session, db_path):
        self._insert_targets(session)
        df = get_area_targets("E14001063", year=2025, db_path=db_path)
        assert len(df) == 1
        assert df.iloc[0]["value"] == 12345.0

    def test_count_by_source(self, session, db_path):
        self._insert_targets(session)
        counts = count_targets_by_source(db_path=db_path)
        assert counts["ons"] == 1
        assert counts["hmrc"] == 1
