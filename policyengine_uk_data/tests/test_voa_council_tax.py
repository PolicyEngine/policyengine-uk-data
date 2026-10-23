"""The Scottish council tax band targets come from the committed workbook only.

They used to come from a live gov.scot download, with a fixed substitute
distribution whenever the download failed. Builds of the same code then
calibrated to different Scottish targets (#596).
"""

from __future__ import annotations

import shutil

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from policyengine_uk_data.storage import STORAGE_FOLDER
from policyengine_uk_data.targets.sources import voa_council_tax

# The workbook's Scotland row (Number of Chargeable Dwellings, September 2025).
PUBLISHED = {
    "A": 498_707,
    "B": 583_705,
    "C": 426_388,
    "D": 369_621,
    "E": 366_881,
    "F": 221_282,
    "G": 142_084,
    "H": 14_481,
    "Total": 2_623_149,
}


def test_scotland_band_counts_are_the_committed_workbook():
    counts = voa_council_tax._get_scotland_band_counts()
    assert counts == PUBLISHED
    assert sum(counts[band] for band in "ABCDEFGH") == counts["Total"]


def test_no_download_is_attempted(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("the Scottish workbook must not be downloaded")

    monkeypatch.setattr(voa_council_tax.requests, "get", refuse)
    assert voa_council_tax._get_scotland_band_counts()["Total"] == 2_623_149


@settings(
    max_examples=20,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(data=st.data())
def test_any_change_to_the_workbook_bytes_fails(tmp_path, monkeypatch, data):
    """No substitute distribution: other bytes stop the build."""
    target = tmp_path / voa_council_tax._SCOTLAND_WORKBOOK_FILE
    target.parent.mkdir(parents=True, exist_ok=True)
    original = (STORAGE_FOLDER / voa_council_tax._SCOTLAND_WORKBOOK_FILE).read_bytes()
    i = data.draw(st.integers(0, len(original) - 1))
    target.write_bytes(original[:i] + bytes([original[i] ^ 0x01]) + original[i + 1 :])
    monkeypatch.setattr("policyengine_uk_data.storage.STORAGE_FOLDER", tmp_path)
    with pytest.raises(ValueError, match="sha256"):
        voa_council_tax._get_scotland_band_counts()


def test_a_missing_workbook_fails(tmp_path, monkeypatch):
    monkeypatch.setattr("policyengine_uk_data.storage.STORAGE_FOLDER", tmp_path)
    with pytest.raises(FileNotFoundError):
        voa_council_tax._get_scotland_band_counts()


def test_the_committed_workbook_still_reads_after_a_copy(tmp_path, monkeypatch):
    """Positive control for the two tests above: identical bytes elsewhere pass."""
    target = tmp_path / voa_council_tax._SCOTLAND_WORKBOOK_FILE
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(STORAGE_FOLDER / voa_council_tax._SCOTLAND_WORKBOOK_FILE, target)
    monkeypatch.setattr("policyengine_uk_data.storage.STORAGE_FOLDER", tmp_path)
    assert voa_council_tax._get_scotland_band_counts() == PUBLISHED
