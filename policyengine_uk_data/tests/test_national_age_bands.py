"""National age-band granularity tests.

The national gender x age targets previously used 15-year bands (0-14,
15-29, ...). A band that wide pins only its own total: the calibration could
satisfy 15-29 while distributing those people any way it liked inside the
band, and it settled on one that starved 18-24 down to ~3.4M against ONS's
~5.4M. Regional ten-year age bands constrain the same household weights,
but provide less age detail than national five-year sex bands.

These tests pin the five-year granularity and the no-overlap property, so a
future edit cannot quietly widen the bands or reintroduce the superseded CSV
rows alongside them.
"""

import re
from pathlib import Path

import pandas as pd

from policyengine_uk_data.targets.sources import ons_demographics
from policyengine_uk_data.targets.sources.ons_demographics import (
    _GENDER_BANDS,
    get_targets,
)

_GENDER_AGE = re.compile(r"^ons/(female|male)_(\d+)_(\d+|plus)$")


def _gender_age_targets():
    return [t for t in get_targets() if _GENDER_AGE.match(t.name)]


def test_bands_are_at_most_five_years_wide():
    """Only the terminal top-coded band may be wider than five years."""
    assert _GENDER_BANDS[-1] == (80, None)
    for low, high in _GENDER_BANDS[:-1]:
        assert high is not None, "only the terminal band may be open-ended"
        assert 0 <= high - low < 5, f"band {low}-{high} is too wide to pin composition"


def test_bands_are_contiguous_and_non_overlapping():
    """Overlapping bands would double-count the population."""
    ordered = sorted(_GENDER_BANDS)
    for (_, prev_high), (next_low, _) in zip(ordered, ordered[1:]):
        assert next_low == prev_high + 1, f"gap or overlap at {prev_high}/{next_low}"


def test_young_adult_ages_are_covered_by_narrow_bands():
    """18-24 must not sit inside one wide band, which is the failure mode."""
    covering = [b for b in _GENDER_BANDS if b[0] <= 24 and (b[1] is None or b[1] >= 18)]
    assert len(covering) >= 2, (
        "18-24 is covered by a single band; its composition would be free"
    )


def test_national_gender_age_targets_are_emitted_once_per_band():
    """The superseded 15-year CSV rows must not be emitted alongside these."""
    targets = _gender_age_targets()
    assert len(targets) == len(_GENDER_BANDS) * 2
    assert len({t.name for t in targets}) == len(targets)


def test_national_gender_age_targets_sum_to_uk_population(monkeypatch):
    """Every fixture age must be counted exactly once across both sexes."""
    projection = pd.read_csv(
        Path(__file__).parent / "fixtures" / "ons_uk_population_2025.csv"
    )
    monkeypatch.setattr(ons_demographics, "_download_uk_projection", lambda: projection)
    targets = _gender_age_targets()
    year = 2025
    total = sum(t.values[year] for t in targets)
    assert total == projection[str(year)].sum()
