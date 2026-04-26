"""Regression test for OBR NIC target parsing.

Bug-hunt finding U8: `_parse_nics` only covered Class 1 employee and
employer NICs, omitting Classes 2 (self-employed flat-rate), 3 (voluntary)
and 4 (self-employed profit-based). Calibration had no target for those
receipts and pushed self-employment income downward to compensate.

This test drives the parser with a minimal in-memory openpyxl workbook
that mimics the OBR EFO Table 3.4 layout, and asserts that targets for
all five NIC variables are produced.
"""

from __future__ import annotations

import importlib.util
from unittest.mock import patch

import pytest

if importlib.util.find_spec("openpyxl") is None:
    pytest.skip("openpyxl not installed", allow_module_level=True)

import openpyxl  # noqa: E402


def _build_fake_obr_workbook() -> openpyxl.Workbook:
    """Create a stand-in for OBR EFO receipts with a minimal Table 3.4."""
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "3.4"

    # Column B holds row labels; columns C-I hold FY values in £bn.
    rows = [
        ("Class 1 Employee NICs", [120.0] * 7),
        ("Class 1 Employer NICs", [140.0] * 7),
        ("Class 2 NICs", [0.3] * 7),
        ("Class 3 NICs", [0.05] * 7),
        ("Class 4 NICs", [4.5] * 7),
    ]
    for row_idx, (label, values) in enumerate(rows, start=2):
        ws.cell(row=row_idx, column=2, value=label)  # col B
        for col_idx, value in enumerate(values, start=3):  # cols C-I
            ws.cell(row=row_idx, column=col_idx, value=value)

    return wb


def test_parse_nics_covers_all_five_classes():
    from policyengine_uk_data.targets.sources import obr as obr_module

    wb = _build_fake_obr_workbook()

    fake_config = {
        "obr": {
            "vintage": "test",
            "efo_receipts": "https://example.invalid/receipts",
            "efo_expenditure": "https://example.invalid/expenditure",
        }
    }
    with patch.object(obr_module, "load_config", return_value=fake_config):
        targets = obr_module._parse_nics(wb)

    names = {t.name for t in targets}
    assert names == {
        "obr/ni_employee",
        "obr/ni_employer",
        "obr/ni_class_2",
        "obr/ni_class_3",
        "obr/ni_class_4",
    }

    variables = {t.variable for t in targets}
    # Variable names must match the policyengine-uk variable identifiers so
    # calibration can map them to simulated totals.
    assert variables == {
        "ni_employee",
        "ni_employer",
        "ni_class_2",
        "ni_class_3",
        "ni_class_4",
    }

    # Values are scaled by 1e9 (£bn → £) inside _read_row_values.
    class_4_target = next(t for t in targets if t.variable == "ni_class_4")
    assert class_4_target.values[2024] == pytest.approx(4.5e9)


def test_parse_nics_tolerates_alt_label_wording():
    """The parser should accept common wording variants for self-employed rows."""
    from policyengine_uk_data.targets.sources import obr as obr_module

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "3.4"

    # Use alternative labels that the parser should still find.
    rows = [
        ("Class 1 Employee NICs", [100.0] * 7),
        ("Class 1 Employer NICs", [110.0] * 7),
        ("Class 2 Self-Employed NICs", [0.2] * 7),
        ("Class 3 Voluntary NICs", [0.04] * 7),
        ("Class 4 Self-Employed NICs", [4.0] * 7),
    ]
    for row_idx, (label, values) in enumerate(rows, start=2):
        ws.cell(row=row_idx, column=2, value=label)
        for col_idx, value in enumerate(values, start=3):
            ws.cell(row=row_idx, column=col_idx, value=value)

    fake_config = {
        "obr": {
            "vintage": "test",
            "efo_receipts": "https://example.invalid/receipts",
            "efo_expenditure": "https://example.invalid/expenditure",
        }
    }
    with patch.object(obr_module, "load_config", return_value=fake_config):
        targets = obr_module._parse_nics(wb)

    assert {t.variable for t in targets} == {
        "ni_employee",
        "ni_employer",
        "ni_class_2",
        "ni_class_3",
        "ni_class_4",
    }
