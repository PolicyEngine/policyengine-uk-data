"""
Per-year snapshot production for the panel pipeline (step 2 of #345).

Takes a single already-imputed base dataset and produces one output file per
requested year by uprating monetary variables to that year's values. The
person, benefit unit and household identifiers are preserved byte-for-byte
across every saved snapshot, so downstream consumers can join rows across
years using the panel keys documented in ``utils/panel_ids``.

This module is deliberately narrow:

- It does not apply demographic ageing (age, mortality, births) — that is
  step 3 of #345.
- It does not run per-year calibration — that depends on step 4 providing
  year-specific targets. Each snapshot therefore carries the same household
  weights as the input base dataset.
- It does not touch the existing ``create_datasets.py`` single-year flow.
  Callers that want panel output opt in explicitly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from policyengine_uk.data import UKSingleYearDataset

from policyengine_uk_data.utils.panel_ids import assert_panel_id_consistency
from policyengine_uk_data.utils.uprating import uprate_dataset


DEFAULT_FILENAME_TEMPLATE = "enhanced_frs_{year}.h5"


def create_yearly_snapshots(
    base: UKSingleYearDataset,
    years: Iterable[int],
    output_dir: str | Path,
    *,
    filename_template: str = DEFAULT_FILENAME_TEMPLATE,
) -> list[Path]:
    """Produce and save one uprated snapshot per requested year.

    Args:
        base: a fully-imputed (and optionally calibrated) base dataset. Its
            ``time_period`` defines the source year of the uprating step.
        years: the target years to produce snapshots for. May include the
            base year, in which case that snapshot is written as a straight
            copy with no uprating applied.
        output_dir: directory to write ``.h5`` snapshots into. Must exist.
        filename_template: ``str.format`` template taking a ``year`` field.

    Returns:
        The list of written file paths, in the order the years were given.

    Raises:
        AssertionError: if any produced snapshot's panel IDs diverge from
            the base. This cannot happen with the current pipeline (uprating
            does not add or drop rows) but the check is there so that step 3
            — which _will_ mutate the person table — cannot silently break
            the panel contract.
        FileNotFoundError: if ``output_dir`` does not exist.
    """
    output_dir = Path(output_dir)
    if not output_dir.is_dir():
        raise FileNotFoundError(f"Output directory {output_dir} does not exist.")

    base_year = int(base.time_period)
    written: list[Path] = []

    for year in years:
        if int(year) == base_year:
            snapshot = base.copy()
        else:
            snapshot = uprate_dataset(base, int(year))

        assert_panel_id_consistency(
            base,
            snapshot,
            label_base=f"base_{base_year}",
            label_other=f"snapshot_{int(year)}",
        )

        file_path = output_dir / filename_template.format(year=int(year))
        snapshot.save(file_path)
        written.append(file_path)

    return written
