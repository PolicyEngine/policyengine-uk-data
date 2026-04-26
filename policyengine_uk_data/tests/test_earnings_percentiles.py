"""Tests for missing earnings percentile imputation (#68)."""

import numpy as np
import pandas as pd

from policyengine_uk_data.datasets.local_areas.earnings_percentiles import (
    PERCENTILE_COLUMNS,
    REFERENCE_VALUES,
    fill_missing_percentiles,
)


def _make_row(**overrides):
    """Create a row with NaN for all percentiles, then apply overrides."""
    data = {col: np.nan for col in PERCENTILE_COLUMNS}
    for k, v in overrides.items():
        col = f"{k} percentile"
        data[col] = v
    return pd.Series(data)


class TestFillMissingPercentiles:
    def test_known_values_preserved(self):
        """Known percentile values should not be overwritten."""
        row = _make_row(**{str(p): float(v) for p, v in REFERENCE_VALUES.items()})
        filled = fill_missing_percentiles(row.copy())
        for col in PERCENTILE_COLUMNS:
            assert filled[col] == row[col]

    def test_missing_p95_filled(self):
        """A missing P95 should be imputed when P90 is known."""
        row = _make_row(**{"90": 50000.0})
        filled = fill_missing_percentiles(row.copy())
        assert pd.notna(filled["95 percentile"])

    def test_all_missing_returns_unchanged(self):
        """If all percentiles are NaN, the row should be returned unchanged."""
        row = _make_row()
        filled = fill_missing_percentiles(row.copy())
        assert all(pd.isna(filled[col]) for col in PERCENTILE_COLUMNS)

    def test_averaging_pulls_toward_national(self):
        """With national_weight=0.5, imputed value should be between
        the pure ratio estimate and the national reference value."""
        row = _make_row(**{"90": 80000.0})  # higher than national P90 (62000)

        pure_ratio = fill_missing_percentiles(row.copy(), national_weight=0.0)
        blended = fill_missing_percentiles(row.copy(), national_weight=0.5)

        p95_ratio = pure_ratio["95 percentile"]
        p95_national = REFERENCE_VALUES[95]
        p95_blended = blended["95 percentile"]

        # Blended should be between ratio estimate and national value
        assert (
            min(p95_ratio, p95_national) <= p95_blended <= max(p95_ratio, p95_national)
        )

    def test_national_weight_zero_matches_old_behaviour(self):
        """national_weight=0.0 should reproduce the original ratio-only approach."""
        row = _make_row(**{"90": 50000.0})
        filled = fill_missing_percentiles(row.copy(), national_weight=0.0)
        expected_p95 = 50000.0 * (REFERENCE_VALUES[95] / REFERENCE_VALUES[90])
        assert abs(filled["95 percentile"] - expected_p95) < 0.01

    def test_national_weight_one_gives_national_values(self):
        """national_weight=1.0 should return the national reference values."""
        row = _make_row(**{"90": 50000.0})
        filled = fill_missing_percentiles(row.copy(), national_weight=1.0)
        assert abs(filled["95 percentile"] - REFERENCE_VALUES[95]) < 0.01

    def test_monotonicity_preserved(self):
        """Imputed percentiles should be monotonically increasing."""
        row = _make_row(**{"10": 12000.0, "50": 25000.0, "90": 55000.0})
        filled = fill_missing_percentiles(row.copy())
        values = [filled[col] for col in PERCENTILE_COLUMNS if pd.notna(filled[col])]
        for i in range(len(values) - 1):
            assert values[i] <= values[i + 1], (
                f"Non-monotonic at index {i}: {values[i]} > {values[i + 1]}"
            )

    def test_extrapolate_downward_from_upper(self):
        """If only an upper percentile is known, lower ones should be extrapolated."""
        row = _make_row(**{"50": 30000.0})
        filled = fill_missing_percentiles(row.copy())
        assert pd.notna(filled["10 percentile"])
        assert filled["10 percentile"] < filled["50 percentile"]
