"""Tests for the panel ID contract utilities."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from policyengine_uk_data.utils.panel_ids import (
    PANEL_ID_COLUMNS,
    assert_panel_id_consistency,
    get_panel_ids,
)


@dataclass
class FakeDataset:
    """Duck-typed stand-in for ``UKSingleYearDataset``.

    The panel ID utilities only need ``.person``, ``.benunit`` and
    ``.household`` DataFrames, so the tests don't need to build a real
    ``UKSingleYearDataset`` (which pulls in the full PolicyEngine UK
    microsim stack).
    """

    person: pd.DataFrame
    benunit: pd.DataFrame
    household: pd.DataFrame


def _make_dataset(
    household_ids=(1, 2),
    benunit_ids=(101, 201),
    person_ids=(1001, 1002, 2001),
) -> FakeDataset:
    return FakeDataset(
        household=pd.DataFrame({"household_id": list(household_ids)}),
        benunit=pd.DataFrame({"benunit_id": list(benunit_ids)}),
        person=pd.DataFrame({"person_id": list(person_ids)}),
    )


def test_panel_id_columns_covers_all_entities():
    """The public constant must document all three entity tables."""
    assert set(PANEL_ID_COLUMNS) == {"household", "benunit", "person"}


def test_get_panel_ids_returns_sorted_unique_int64():
    ds = _make_dataset(
        household_ids=(2, 1, 2),
        benunit_ids=(201, 101),
        person_ids=(2001, 1001, 1002, 1001),
    )
    ids = get_panel_ids(ds)
    assert ids.household.tolist() == [1, 2]
    assert ids.benunit.tolist() == [101, 201]
    assert ids.person.tolist() == [1001, 1002, 2001]
    # All three arrays must share the int64 dtype so downstream joins
    # don't silently coerce.
    assert ids.household.dtype == np.int64
    assert ids.benunit.dtype == np.int64
    assert ids.person.dtype == np.int64


def test_get_panel_ids_raises_when_id_column_missing():
    ds = _make_dataset()
    ds.person = pd.DataFrame({"wrong_column": [1, 2, 3]})
    with pytest.raises(KeyError, match="person_id"):
        get_panel_ids(ds)


def test_assert_consistency_passes_for_identical_datasets():
    base = _make_dataset()
    other = _make_dataset()
    # Must not raise.
    assert_panel_id_consistency(base, other)


def test_assert_consistency_passes_when_row_order_differs():
    """ID equality is set-based; within-table row order must not matter."""
    base = _make_dataset(person_ids=(1001, 1002, 2001))
    other = _make_dataset(person_ids=(2001, 1001, 1002))
    assert_panel_id_consistency(base, other)


def test_assert_consistency_detects_extra_person():
    base = _make_dataset()
    other = _make_dataset(person_ids=(1001, 1002, 2001, 9999))
    with pytest.raises(AssertionError) as excinfo:
        assert_panel_id_consistency(base, other)
    assert "person" in str(excinfo.value)
    assert "9999" in str(excinfo.value)


def test_assert_consistency_detects_missing_household():
    base = _make_dataset()
    other = _make_dataset(household_ids=(1,))
    with pytest.raises(AssertionError) as excinfo:
        assert_panel_id_consistency(base, other)
    msg = str(excinfo.value)
    assert "household" in msg
    assert "2" in msg


def test_assert_consistency_reports_multiple_entities_together():
    """Both mismatched entities should appear in a single error."""
    base = _make_dataset()
    other = _make_dataset(
        household_ids=(1,),
        person_ids=(1001, 1002),
    )
    with pytest.raises(AssertionError) as excinfo:
        assert_panel_id_consistency(base, other)
    msg = str(excinfo.value)
    assert "household" in msg
    assert "person" in msg


def test_assert_consistency_respects_entities_filter():
    """Callers can restrict the check to a subset of entities."""
    base = _make_dataset()
    other = _make_dataset(person_ids=(9999,))
    # Full check fails because the person table diverges...
    with pytest.raises(AssertionError):
        assert_panel_id_consistency(base, other)
    # ...but skipping the person entity should pass.
    assert_panel_id_consistency(base, other, entities=("household", "benunit"))


def test_assert_consistency_uses_provided_labels_in_message():
    base = _make_dataset()
    other = _make_dataset(household_ids=(1,))
    with pytest.raises(AssertionError) as excinfo:
        assert_panel_id_consistency(
            base, other, label_base="year_2023", label_other="year_2024"
        )
    msg = str(excinfo.value)
    assert "year_2023" in msg
    assert "year_2024" in msg
