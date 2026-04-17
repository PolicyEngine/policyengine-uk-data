"""Tests for the per-year snapshot uploader (step 6 of #345)."""

from pathlib import Path
from unittest import mock

import pytest

from policyengine_uk_data.storage import upload_yearly_snapshots as uys


def _touch(path: Path) -> None:
    path.write_bytes(b"")


def test_yearly_snapshot_paths_is_pure(tmp_path):
    """No filesystem access, no upload; just a name mapping."""
    paths = uys.yearly_snapshot_paths([2023, 2024, 2025], storage_folder=tmp_path)
    assert paths == [
        tmp_path / "enhanced_frs_2023.h5",
        tmp_path / "enhanced_frs_2024.h5",
        tmp_path / "enhanced_frs_2025.h5",
    ]


def test_yearly_snapshot_paths_accepts_any_iterable(tmp_path):
    """Generators should work, not only lists."""
    paths = uys.yearly_snapshot_paths(
        (y for y in range(2023, 2026)), storage_folder=tmp_path
    )
    assert [p.name for p in paths] == [
        "enhanced_frs_2023.h5",
        "enhanced_frs_2024.h5",
        "enhanced_frs_2025.h5",
    ]


def test_upload_rejects_empty_year_list(tmp_path):
    with pytest.raises(ValueError, match="at least one year"):
        uys.upload_yearly_snapshots([], storage_folder=tmp_path)


def test_upload_refuses_when_any_file_missing(tmp_path):
    _touch(tmp_path / "enhanced_frs_2023.h5")
    # 2024 is deliberately absent.
    with pytest.raises(FileNotFoundError, match="enhanced_frs_2024.h5"):
        with mock.patch.object(uys, "upload_data_files") as upload_spy:
            uys.upload_yearly_snapshots([2023, 2024], storage_folder=tmp_path)
    # No partial upload allowed.
    upload_spy.assert_not_called()


def test_upload_calls_upload_data_files_with_private_destination(tmp_path):
    for y in (2023, 2024):
        _touch(tmp_path / f"enhanced_frs_{y}.h5")

    with mock.patch.object(uys, "upload_data_files") as upload_spy:
        result = uys.upload_yearly_snapshots([2023, 2024], storage_folder=tmp_path)

    upload_spy.assert_called_once()
    kwargs = upload_spy.call_args.kwargs
    assert kwargs["hf_repo_name"] == "policyengine/policyengine-uk-data-private"
    assert kwargs["hf_repo_type"] == "model"
    assert kwargs["gcs_bucket_name"] == "policyengine-uk-data-private"
    assert [p.name for p in kwargs["files"]] == [
        "enhanced_frs_2023.h5",
        "enhanced_frs_2024.h5",
    ]
    # Return value must match the uploaded set in order.
    assert [p.name for p in result] == [
        "enhanced_frs_2023.h5",
        "enhanced_frs_2024.h5",
    ]


def test_destination_constants_locked_to_private_repo():
    """A change to these values should be a conscious, reviewed edit.

    The CLAUDE.md rule is explicit: the upload destination must remain
    the private repo, and only the data controller can approve a change.
    Locking them in a test makes accidental drift loud instead of quiet.
    """
    assert uys.PRIVATE_HF_REPO == "policyengine/policyengine-uk-data-private"
    assert uys.PRIVATE_HF_REPO_TYPE == "model"
    assert uys.PRIVATE_GCS_BUCKET == "policyengine-uk-data-private"


def test_destination_not_overridable_via_kwargs():
    """No keyword arguments should allow redirecting the upload."""
    import inspect

    sig = inspect.signature(uys.upload_yearly_snapshots)
    # Only ``years`` and ``storage_folder`` are tunable by callers.
    assert set(sig.parameters) == {"years", "storage_folder"}
