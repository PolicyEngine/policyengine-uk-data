"""Local UC database targets keep the public workbook's matched area codes."""

from unittest.mock import Mock

import pandas as pd

from policyengine_uk_data.db import etl
from policyengine_uk_data.storage import STORAGE_FOLDER
from policyengine_uk_data.targets.sources.local_uc import (
    get_constituency_uc_targets,
    get_la_uc_targets,
)


def test_local_uc_etl_inserts_all_targets_with_matched_codes_and_names(monkeypatch):
    session = Mock()
    inserted = []

    def insert_target(actual_session, **target):
        assert actual_session is session
        inserted.append(target)

    monkeypatch.setattr(etl, "_insert_target", insert_target)

    count = etl._load_local_uc_targets(session)

    assert count == len(inserted) == 1_010
    session.commit.assert_called_once_with()
    for level, getter, area_file in [
        ("constituency", get_constituency_uc_targets, "constituencies_2024.csv"),
        ("local_authority", get_la_uc_targets, "local_authorities_2021.csv"),
    ]:
        expected = getter()
        areas = pd.read_csv(STORAGE_FOLDER / area_file).set_index(
            "code", verify_integrity=True
        )
        local = [row for row in inserted if row["geographic_level"] == level]
        codes = [row["geo_code"] for row in local]
        assert len(codes) == len(set(codes)) == len(expected)
        assert set(codes) == set(expected.index)
        for target in local:
            code = target["geo_code"]
            assert target["name"] == f"dwp/{level}/{code}/uc_households"
            assert target["geo_name"] == areas.loc[code, "name"]
            assert target["values"] == {2025: float(expected.loc[code])}
