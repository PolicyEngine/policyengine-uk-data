"""Tests for the committed EFO workbook fallback.

obr.uk serves 403 Forbidden to GitHub Actions runner IPs, and losing the
workbooks silently drops 28 OBR targets and calibrates a degraded dataset
(observed on the 2026-07-21 push builds). obr.uk can also answer 200 with
an HTML "No Access" page, which once escaped the fallback as a BadZipFile
and dropped the nine receipts and NICs targets (observed 2026-10-04). These
tests pin: the fallback workbooks are committed and parseable, every kind of
failed download uses them instead of raising, and permanent failures do not
burn the retry budget.
"""

import io
import logging
import zipfile
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import requests

from policyengine_uk_data.storage import STORAGE_FOLDER
from policyengine_uk_data.targets.sources import obr
from policyengine_uk_data.targets.sources._common import load_config


@pytest.fixture(autouse=True)
def _clear_workbook_cache():
    obr._download_workbook.cache_clear()
    yield
    obr._download_workbook.cache_clear()


def _response(status_code, content=b"", content_type=None):
    headers = {"Content-Type": content_type} if content_type else {}
    return SimpleNamespace(status_code=status_code, headers=headers, content=content)


def _zip(members: dict[str, str]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    return buffer.getvalue()


def _connection_error():
    raise requests.ConnectionError("no route to obr.uk")


_NO_ACCESS_PAGE = (
    b"<!DOCTYPE html><html><head>"
    b"<title>No Access - Office for Budget Responsibility</title>"
    b"</head><body><h1>No Access</h1></body></html>"
)

# Every way the download can fail, with the number of requests it should
# cost: transient failures use the full retry budget, permanent ones one.
_FAILED_DOWNLOADS = {
    "200-html-page": (
        lambda: _response(200, _NO_ACCESS_PAGE, "text/html; charset=UTF-8"),
        1,
    ),
    "200-empty-body": (lambda: _response(200), 1),
    "200-truncated-zip": (lambda: _response(200, b"PK\x03\x04garbage"), 1),
    "200-zip-without-xlsx-manifest": (
        lambda: _response(200, _zip({"readme.txt": "not a workbook"})),
        1,
    ),
    "200-zip-without-workbook-part": (
        lambda: _response(
            200,
            _zip(
                {
                    "[Content_Types].xml": '<Types xmlns="http://schemas.'
                    'openxmlformats.org/package/2006/content-types"/>'
                }
            ),
        ),
        1,
    ),
    "200-zip-with-malformed-xml": (
        lambda: _response(200, _zip({"[Content_Types].xml": "<Types"})),
        1,
    ),
    "403": (lambda: _response(403), 1),
    "404": (lambda: _response(404), 1),
    "500-until-retries-run-out": (
        lambda: _response(500),
        obr._DOWNLOAD_MAX_ATTEMPTS,
    ),
    "connection-error": (_connection_error, obr._DOWNLOAD_MAX_ATTEMPTS),
}

_EFO_URL_KEYS = ["efo_receipts", "efo_expenditure"]

_RECEIPTS_AND_NICS_TARGETS = {
    "obr/income_tax",
    "obr/ni",
    "obr/vat",
    "obr/fuel_duties",
    "obr/capital_gains_tax",
    "obr/sdlt",
    "obr/ni_employee",
    "obr/ni_employer",
    "obr/ni_self_employed",
}


@contextmanager
def _obr_answering(make_response):
    """Answer every requests.get with make_response(), skip retry sleeps and
    spy on the fallback. Yields the request and fallback call logs."""
    requests_made, fallbacks_used = [], []
    real_fallback = obr._fallback_workbook

    def get(url, *args, **kwargs):
        requests_made.append(url)
        return make_response()

    def fallback(url):
        wb = real_fallback(url)
        fallbacks_used.append((url, wb))
        return wb

    with (
        patch.object(obr.requests, "get", side_effect=get),
        patch.object(obr.time, "sleep", lambda s: None),
        patch.object(obr, "_fallback_workbook", side_effect=fallback),
    ):
        yield requests_made, fallbacks_used


def test_fallback_workbooks_are_committed_and_parseable():
    for filename in obr._EFO_FALLBACKS.values():
        path = STORAGE_FOLDER / "obr_efo" / filename
        assert path.exists(), f"{filename} missing from storage/obr_efo"
    receipts = obr._fallback_workbook("https://obr.uk/x/efo-receipts/")
    assert receipts is not None
    # The receipts sheet lookup must work on the committed vintage.
    assert obr._find_receipts_sheet(receipts) is not None


@pytest.mark.parametrize(
    "kind, error, cause",
    [
        ("403", requests.HTTPError, type(None)),
        ("200-html-page", ValueError, zipfile.BadZipFile),
    ],
)
def test_unknown_url_with_failed_download_still_raises(kind, error, cause):
    url = "https://obr.uk/download/some-other-file/"
    with _obr_answering(_FAILED_DOWNLOADS[kind][0]):
        with pytest.raises(error, match=f"for url: {url}") as raised:
            obr._download_workbook(url)
    assert isinstance(raised.value.__cause__, cause)


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


@pytest.mark.parametrize("url_key", _EFO_URL_KEYS)
@pytest.mark.parametrize("kind", list(_FAILED_DOWNLOADS))
def test_every_failed_download_uses_committed_workbook(kind, url_key):
    make_response, expected_requests = _FAILED_DOWNLOADS[kind]
    url = load_config()["obr"][url_key]
    with _obr_answering(make_response) as (requests_made, fallbacks_used):
        wb = obr._download_workbook(url)
    assert fallbacks_used == [(url, wb)], "the committed workbook was not returned"
    assert len(requests_made) == expected_requests


@pytest.mark.parametrize("kind", list(_FAILED_DOWNLOADS))
def test_every_failed_download_keeps_receipts_and_nics_targets(kind):
    make_response, _ = _FAILED_DOWNLOADS[kind]
    with _obr_answering(make_response):
        names = {t.name for t in obr.get_targets()}
    assert _RECEIPTS_AND_NICS_TARGETS <= names, sorted(
        _RECEIPTS_AND_NICS_TARGETS - names
    )


def test_workbook_response_is_parsed_without_fallback():
    """Positive control: a real xlsx body is used as served."""
    body = (STORAGE_FOLDER / "obr_efo" / "efo_receipts.xlsx").read_bytes()
    with _obr_answering(lambda: _response(200, body)) as (
        requests_made,
        fallbacks_used,
    ):
        wb = obr._download_workbook(load_config()["obr"]["efo_receipts"])
    assert fallbacks_used == []
    assert len(requests_made) == 1
    assert obr._find_receipts_sheet(wb) is not None


def test_non_workbook_200_warning_names_url_status_and_parse_error(caplog):
    caplog.set_level(logging.WARNING, logger=obr.logger.name)
    url = load_config()["obr"]["efo_receipts"]
    with _obr_answering(_FAILED_DOWNLOADS["200-html-page"][0]):
        obr._download_workbook(url)
    assert (
        f"200 for url: {url}, but the body (text/html; charset=UTF-8) could "
        "not be read as an xlsx workbook (BadZipFile: File is not a zip "
        "file)); using committed workbook fallback"
    ) in caplog.text
