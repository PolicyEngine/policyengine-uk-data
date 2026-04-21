"""
Per-year snapshot production for the panel pipeline.

Takes a single already-imputed base dataset and produces one output
file per requested year.

Two modes:

- ``mode="uprate_only"`` (default): each snapshot is the base dataset
  with monetary variables uprated to that year. Person, benefit unit
  and household identifiers are preserved byte-for-byte so downstream
  consumers can join rows across years on the panel keys documented
  in ``utils/panel_ids``. No demographic ageing, no transitions.
- ``mode="full_panel"``: each year is produced by rolling forward one
  year at a time from the base via :func:`utils.advance_year.advance_year`,
  which applies migration, separation, leaving-home, marriage,
  employment and income-decile transitions, demographic ageing, and
  uprating in a deterministic, seeded sequence. Panel IDs evolve
  (deaths remove IDs, births add IDs, migration both).

Per-year calibration is not run here — each snapshot carries the
same household weights as the base dataset. Calibration belongs in a
separate step that reads these snapshots.

Callers that want panel output opt in explicitly; the default build
pipeline in ``create_datasets.py`` is unchanged.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping

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
    mode: str = "uprate_only",
    seed: int = 0,
    mortality_rates: Any = None,
    fertility_rates: Any = None,
    marriage_rates: Any = None,
    separation_rates: Any = None,
    leaving_home_rates: Any = None,
    net_migration_rates: Any = None,
    ukhls_employment_rates: Mapping | None = None,
    ukhls_decile_rates: Mapping | None = None,
) -> list[Path]:
    """Produce and save one snapshot per requested year.

    Args:
        base: a fully-imputed (and optionally calibrated) base dataset.
            Its ``time_period`` defines the source year.
        years: the target years to produce snapshots for. May include
            the base year (straight copy, no uprating / no ageing).
        output_dir: directory to write ``.h5`` snapshots into. Must
            exist.
        filename_template: ``str.format`` template taking a ``year``
            field.
        mode: ``"uprate_only"`` (default) or ``"full_panel"``. See the
            module docstring for semantics.
        seed: base RNG seed used in full-panel mode. Per-year seeds
            are derived deterministically as ``seed + (year - base_year)``
            so a reproducible run is tagged by the scalar ``seed``.
        mortality_rates, fertility_rates, marriage_rates,
        separation_rates, leaving_home_rates, net_migration_rates,
        ukhls_employment_rates, ukhls_decile_rates:
            passed through to :func:`utils.advance_year.advance_year`
            in full-panel mode; ignored in uprate-only mode.

    Returns:
        The list of written file paths, in the order the years were
        given.

    Raises:
        FileNotFoundError: if ``output_dir`` does not exist.
        ValueError: if ``mode`` is not recognised.
        AssertionError: uprate-only mode enforces byte-for-byte panel
            ID consistency. Full-panel mode does not — panel IDs
            evolve by design there.
    """
    if mode not in ("uprate_only", "full_panel"):
        raise ValueError(f"mode must be 'uprate_only' or 'full_panel', got {mode!r}")

    output_dir = Path(output_dir)
    if not output_dir.is_dir():
        raise FileNotFoundError(f"Output directory {output_dir} does not exist.")

    base_year = int(base.time_period)
    sorted_years = sorted({int(y) for y in years})
    written: list[Path] = []

    if mode == "uprate_only":
        for year in sorted_years:
            if year == base_year:
                snapshot = base.copy()
            else:
                snapshot = uprate_dataset(base, year)
            assert_panel_id_consistency(
                base,
                snapshot,
                label_base=f"base_{base_year}",
                label_other=f"snapshot_{year}",
            )
            file_path = output_dir / filename_template.format(year=year)
            snapshot.save(file_path)
            written.append(file_path)
        return written

    # Full-panel mode: roll forward year-by-year.
    from policyengine_uk_data.utils.advance_year import advance_year

    current = base.copy()
    # Cache of cumulative snapshots keyed on year, so callers can ask
    # for an arbitrary (possibly non-contiguous) target set.
    cache: dict[int, UKSingleYearDataset] = {base_year: current}
    max_year = max(sorted_years + [base_year])
    running = current
    for step_year in range(base_year + 1, max_year + 1):
        running = advance_year(
            running,
            target_year=step_year,
            seed=seed + (step_year - base_year),
            mortality_rates=mortality_rates,
            fertility_rates=fertility_rates,
            marriage_rates=marriage_rates,
            separation_rates=separation_rates,
            leaving_home_rates=leaving_home_rates,
            net_migration_rates=net_migration_rates,
            ukhls_employment_rates=ukhls_employment_rates,
            ukhls_decile_rates=ukhls_decile_rates,
            uprate=True,
        )
        cache[step_year] = running

    for year in sorted_years:
        if year < base_year:
            # Requesting a past year under full_panel mode falls back
            # to a straight uprate/downrate of the base; we don't run
            # transitions backwards in time.
            snapshot = uprate_dataset(base, year)
        else:
            snapshot = cache[year]
        file_path = output_dir / filename_template.format(year=year)
        snapshot.save(file_path)
        written.append(file_path)

    return written
