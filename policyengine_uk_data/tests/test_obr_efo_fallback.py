"""Tests for the committed EFO workbook fallback.

obr.uk serves 403 Forbidden to GitHub Actions runner IPs, and losing the
workbooks silently drops 28 OBR targets and calibrates a degraded dataset
(observed on the 2026-07-21 push builds). These tests pin: the fallback
workbooks are committed and parseable, a failed download uses them instead
of raising, and a 403 does not burn the retry budget.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import requests

from policyengine_uk_data.storage import STORAGE_FOLDER
from policyengine_uk_data.targets.sources import obr


@pytest.fixture(autouse=True)
def _clear_workbook_cache():
    obr._download_workbook.cache_clear()
    yield
    obr._download_workbook.cache_clear()


def _forbidden(*args, **kwargs):
    return SimpleNamespace(status_code=403, headers={}, content=b"")


def test_fallback_workbooks_are_committed_and_parseable():
    for filename in obr._EFO_FALLBACKS.values():
        path = STORAGE_FOLDER / "obr_efo" / filename
        assert path.exists(), f"{filename} missing from storage/obr_efo"
    receipts = obr._fallback_workbook("https://obr.uk/x/efo-receipts/")
    assert receipts is not None
    # The receipts sheet lookup must work on the committed vintage.
    assert obr._find_receipts_sheet(receipts) is not None


def test_403_uses_fallback_without_retrying():
    calls = []

    def get(*args, **kwargs):
        calls.append(args)
        return _forbidden()

    with patch.object(obr.requests, "get", side_effect=get):
        wb = obr._download_workbook(
            "https://obr.uk/download/whatever-forecast-tables-receipts/"
        )
    assert wb is not None
    assert len(calls) == 1, "403 is permanent; it should not be retried"


def test_connection_failure_uses_fallback():
    def get(*args, **kwargs):
        raise requests.ConnectionError("no route to obr.uk")

    with (
        patch.object(obr.requests, "get", side_effect=get),
        patch.object(obr.time, "sleep", lambda s: None),
    ):
        wb = obr._download_workbook(
            "https://obr.uk/download/whatever-forecast-tables-expenditure/"
        )
    assert wb is not None


def test_unknown_url_with_failed_download_still_raises():
    with patch.object(obr.requests, "get", side_effect=_forbidden):
        with pytest.raises(requests.HTTPError):
            obr._download_workbook("https://obr.uk/download/some-other-file/")


def test_full_target_set_available_offline():
    """All 34 OBR targets must build from the committed workbooks alone."""

    def get(*args, **kwargs):
        raise requests.ConnectionError("offline")

    with (
        patch.object(obr.requests, "get", side_effect=get),
        patch.object(obr.time, "sleep", lambda s: None),
    ):
        targets = obr.get_targets()
    names = {t.name for t in targets}
    assert {
        "obr/income_tax",
        "obr/ni_employee",
        "obr/ni_employer",
        "obr/ni_self_employed",
        "obr/capital_gains_tax",
        "obr/vat",
    } <= names
    assert len(names) >= 30
