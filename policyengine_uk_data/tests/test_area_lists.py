"""The committed constituency and local authority lists."""

import csv

import pytest

from policyengine_uk_data.storage import STORAGE_FOLDER


@pytest.mark.parametrize(
    ("file_name", "expected_columns"),
    [
        ("constituencies_2024.csv", ["code", "name", "region", "country"]),
        ("local_authorities_2021.csv", ["code", "name"]),
    ],
)
def test_area_list_has_exactly_the_expected_columns(file_name, expected_columns):
    with open(STORAGE_FOLDER / file_name, newline="", encoding="utf-8") as area_file:
        header, *rows = csv.reader(area_file)
    assert header == expected_columns
    assert all(len(row) == len(header) for row in rows)
