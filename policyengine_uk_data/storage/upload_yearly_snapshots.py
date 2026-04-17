"""Upload per-year panel snapshots to the private dataset repository.

Parallel to ``upload_completed_datasets.py`` (intentionally separate so
that the audited default upload path stays untouched). The upload
destination — private HuggingFace repo and private GCS bucket — is hard-
coded, matching the rule in ``CLAUDE.md``: only aggregate results from
the FRS-derived datasets are ever allowed to leave our infrastructure,
and only via the private channels listed here.

Callers supply the list of years they want shipped; filenames follow the
``enhanced_frs_<year>.h5`` convention established by
``policyengine_uk_data.datasets.yearly_snapshots``. All files must exist
locally — there is no silent-skip, because a partial panel upload is
almost certainly a bug rather than an intent.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from policyengine_uk_data.storage import STORAGE_FOLDER
from policyengine_uk_data.utils.data_upload import upload_data_files


# These constants are deliberately not function arguments: changing the
# upload destination requires a code change (and review by the data
# controller, per ``CLAUDE.md``), not just a keyword argument at a call
# site.
PRIVATE_HF_REPO = "policyengine/policyengine-uk-data-private"
PRIVATE_HF_REPO_TYPE = "model"
PRIVATE_GCS_BUCKET = "policyengine-uk-data-private"

SNAPSHOT_FILENAME_TEMPLATE = "enhanced_frs_{year}.h5"


def yearly_snapshot_paths(
    years: Iterable[int],
    storage_folder: Path | None = None,
) -> list[Path]:
    """Return the ``enhanced_frs_<year>.h5`` path for each requested year.

    No side effects, no filesystem lookup — this is the pure mapping from
    year-list to paths, so callers can preview what will be uploaded
    without triggering anything.
    """
    folder = Path(storage_folder) if storage_folder is not None else STORAGE_FOLDER
    return [folder / SNAPSHOT_FILENAME_TEMPLATE.format(year=int(y)) for y in years]


def upload_yearly_snapshots(
    years: Iterable[int],
    storage_folder: Path | None = None,
) -> list[Path]:
    """Upload one ``enhanced_frs_<year>.h5`` per year to the private repo.

    Args:
        years: calendar years to upload.
        storage_folder: override the source directory. Defaults to the
            repo's ``STORAGE_FOLDER``.

    Returns:
        The list of paths that were uploaded, in the order the years
        were supplied.

    Raises:
        FileNotFoundError: if any of the expected per-year files is
            missing. No partial upload is attempted.
        ValueError: if ``years`` is empty.
    """
    year_list = list(years)
    if not year_list:
        raise ValueError("upload_yearly_snapshots requires at least one year.")

    paths = yearly_snapshot_paths(year_list, storage_folder=storage_folder)
    missing = [p for p in paths if not p.exists()]
    if missing:
        joined = ", ".join(str(p.name) for p in missing)
        raise FileNotFoundError(
            f"Cannot upload panel snapshots — missing file(s): {joined}."
        )

    upload_data_files(
        files=paths,
        hf_repo_name=PRIVATE_HF_REPO,
        hf_repo_type=PRIVATE_HF_REPO_TYPE,
        gcs_bucket_name=PRIVATE_GCS_BUCKET,
    )
    return paths


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Upload per-year panel snapshots to the private repo."
    )
    parser.add_argument(
        "years",
        nargs="+",
        type=int,
        help="Calendar years to upload, e.g. 2023 2024 2025.",
    )
    args = parser.parse_args()
    upload_yearly_snapshots(args.years)
