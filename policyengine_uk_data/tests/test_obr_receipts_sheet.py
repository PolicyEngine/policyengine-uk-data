"""Tests for locating the OBR current-receipts table.

EFO releases renumber their worksheets between vintages. The cash-basis
receipts rows (NICs, VAT, fuel duties, capital gains tax, SDLT) were read from
a hard-coded sheet "3.9", but in the March 2026 tables current receipts sit on
3.8 and 3.9 is the APD forecast. Every lookup therefore raised, was caught, and
logged a warning — so all five targets vanished from the calibration surface
without failing anything.

These tests pin the title-based lookup and the loud failure that replaces the
silent drop.
"""

import openpyxl
import pytest

from policyengine_uk_data.targets.sources.obr import _find_receipts_sheet


def _wb(sheets: dict[str, str]) -> openpyxl.Workbook:
    """Build a workbook whose sheets carry their EFO title in cell B2."""
    wb = openpyxl.Workbook()
    wb.remove(wb.active)
    for name, title in sheets.items():
        ws = wb.create_sheet(name)
        ws["B2"] = title
    return wb


def test_finds_receipts_sheet_by_title_not_number():
    """The March 2026 layout: receipts on 3.8, APD on 3.9."""
    wb = _wb(
        {
            "3.8": "3.8 Current receipts (on a cash basis)",
            "3.9": "3.9 APD forecast - projection of passenger numbers",
        }
    )
    assert _find_receipts_sheet(wb).title == "3.8"


def test_finds_receipts_sheet_after_renumbering():
    """A future vintage may move the table again; the title still finds it."""
    wb = _wb(
        {
            "3.9": "3.9 Some other forecast",
            "3.11": "3.11 Current receipts (on a cash basis)",
        }
    )
    assert _find_receipts_sheet(wb).title == "3.11"


def test_raises_when_no_receipts_sheet():
    """Missing the table must fail loudly rather than yield zero targets."""
    wb = _wb({"3.9": "3.9 APD forecast"})
    with pytest.raises(ValueError, match="Current receipts"):
        _find_receipts_sheet(wb)
