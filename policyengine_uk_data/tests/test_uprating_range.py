"""Tests for year-range validation in `policyengine_uk_data.utils.uprating`.

The uprating factor table covers ``[START_YEAR, END_YEAR]`` (2020–2034 on
main). Callers that request a year outside this range previously got a
raw pandas ``KeyError`` or a silently wrong value. After the fix in
finding U5 they must get ``UpratingYearOutOfRangeError`` with a clear
message.
"""

from __future__ import annotations

import importlib.util

import pandas as pd
import pytest

if importlib.util.find_spec("policyengine_uk") is None:
    pytest.skip(
        "policyengine_uk not available in test environment",
        allow_module_level=True,
    )


def _seed_uprating_table(tmp_path, start=2020, end=2034):
    """Write a minimal uprating_factors.csv covering a known year range."""

    storage = tmp_path / "storage"
    storage.mkdir()
    years = list(range(start, end + 1))
    table = pd.DataFrame(
        {str(y): [1.0 + 0.02 * (y - start)] for y in years},
        index=pd.Index(["employment_income"], name="Variable"),
    )
    table.to_csv(storage / "uprating_factors.csv")
    return storage


def test_uprate_dataset_rejects_year_above_end(tmp_path, monkeypatch):
    from policyengine_uk_data.utils import uprating as uprating_module
    from policyengine_uk_data.utils.uprating import (
        UpratingYearOutOfRangeError,
        uprate_dataset,
    )

    storage = _seed_uprating_table(tmp_path)
    monkeypatch.setattr(uprating_module, "STORAGE_FOLDER", storage)

    # target_year=2035 is above END_YEAR=2034 → must raise with clear message.
    class _Dataset:
        time_period = 2023
        tables = []

        def copy(self):
            return self

    with pytest.raises(UpratingYearOutOfRangeError, match="target_year=2035"):
        uprate_dataset(_Dataset(), target_year=2035)


def test_uprate_values_rejects_years_outside_range(tmp_path, monkeypatch):
    from policyengine_uk_data.utils import uprating as uprating_module
    from policyengine_uk_data.utils.uprating import (
        UpratingYearOutOfRangeError,
        uprate_values,
    )

    storage = _seed_uprating_table(tmp_path)
    monkeypatch.setattr(uprating_module, "STORAGE_FOLDER", storage)

    with pytest.raises(UpratingYearOutOfRangeError, match="end_year=2099"):
        uprate_values(100.0, "employment_income", start_year=2020, end_year=2099)

    with pytest.raises(UpratingYearOutOfRangeError, match="start_year=1999"):
        uprate_values(100.0, "employment_income", start_year=1999, end_year=2034)


def test_uprate_values_accepts_supported_range(tmp_path, monkeypatch):
    from policyengine_uk_data.utils import uprating as uprating_module
    from policyengine_uk_data.utils.uprating import uprate_values

    storage = _seed_uprating_table(tmp_path)
    monkeypatch.setattr(uprating_module, "STORAGE_FOLDER", storage)

    # A supported year range still works and returns a sensible factor.
    result = uprate_values(100.0, "employment_income", start_year=2020, end_year=2034)
    # Seed table has linear growth 0.02 per year; 2020→2034 = 14 years ×
    # 0.02 = 1.28 multiplier off a 1.0 base.
    assert result == pytest.approx(128.0)
