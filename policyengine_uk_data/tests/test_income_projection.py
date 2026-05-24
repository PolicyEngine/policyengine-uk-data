"""Tests for income projection accuracy (issues #218 and #393).

These tests verify that projected incomes are not inflated beyond
reasonable bounds after projection. They require incomes_projection.csv
to have been generated and will be skipped otherwise.
"""

import pytest
import pandas as pd
from policyengine_uk_data.storage import STORAGE_FOLDER
from policyengine_uk_data.targets.sources.hmrc_spi import _SPI_YEAR

PROJECTION_PATH = STORAGE_FOLDER / "incomes_projection.csv"
BASE_PATH = STORAGE_FOLDER / "incomes.csv"


def _is_aggregate_row(df):
    return (df["total_income_lower_bound"] == 12_570) & (
        df["total_income_upper_bound"] == float("inf")
    )


def _without_aggregate(df):
    return df[~_is_aggregate_row(df)]


@pytest.fixture
def projections():
    if not PROJECTION_PATH.exists():
        pytest.skip(
            "incomes_projection.csv not available (run create_income_projections first)"
        )
    return pd.read_csv(PROJECTION_PATH)


@pytest.fixture
def base_targets():
    if not BASE_PATH.exists():
        pytest.skip("incomes.csv not available")
    return pd.read_csv(BASE_PATH)


# Maximum tolerable ratio of projected total to uprated SPI baseline.
# Uprating from the current SPI year to 2029 should not exceed ~1.6x
# even with generous growth assumptions. A 2x cap gives ample
# headroom while catching the ~2.5x inflation documented in #218.
MAX_RATIO = 2.0


@pytest.mark.parametrize(
    "variable",
    [
        "dividend_income",
        "property_income",
        "private_pension_income",
        "state_pension",
    ],
)
def test_projected_totals_not_inflated(projections, base_targets, variable):
    """No income type should be inflated >2x relative to the SPI baseline."""
    spi_total = _without_aggregate(base_targets)[f"{variable}_amount"].sum()
    for year in projections["year"].unique():
        year_df = _without_aggregate(projections[projections["year"] == year])
        projected_total = year_df[f"{variable}_amount"].sum()
        ratio = projected_total / spi_total
        assert ratio < MAX_RATIO, (
            f"{variable} in {year}: projected £{projected_total / 1e9:.1f}bn "
            f"is {ratio:.2f}x the SPI baseline £{spi_total / 1e9:.1f}bn "
            f"(max allowed {MAX_RATIO}x)"
        )


def test_employment_income_still_calibrated(projections, base_targets):
    """Base-year employment income should match the official SPI table."""
    spi_total = _without_aggregate(base_targets)["employment_income_amount"].sum()
    base_projection = _without_aggregate(projections[projections["year"] == _SPI_YEAR])
    projected_total = base_projection["employment_income_amount"].sum()
    ratio = projected_total / spi_total
    assert 0.999 < ratio < 1.001, (
        f"Employment income in {_SPI_YEAR}: projected £{projected_total / 1e9:.1f}bn "
        f"vs SPI baseline £{spi_total / 1e9:.1f}bn (ratio {ratio:.2f})"
    )


def test_high_income_band_not_extreme(projections):
    """The highest income bands should not show extreme overestimation.

    Issue #218 found the £500k-£1M band at 12.6x target — this test
    guards against that regression.
    """
    for year in projections["year"].unique():
        year_df = _without_aggregate(projections[projections["year"] == year])
        total_all_bands = year_df["dividend_income_amount"].sum()
        # The top two bands should not dominate total dividends
        # (sorted by income band, last rows are highest)
        top_bands = year_df.tail(2)
        top_band_total = top_bands["dividend_income_amount"].sum()
        if total_all_bands > 0:
            top_share = top_band_total / total_all_bands
            assert top_share < 0.85, (
                f"In {year}, top 2 income bands hold {top_share:.0%} of "
                f"all dividend income — suggests weight inflation in "
                f"high-income bands"
            )


def test_projection_starts_at_current_spi_target_year(projections):
    assert projections["year"].min() == _SPI_YEAR


def test_projection_aggregate_rows_match_detailed_totals(projections):
    """Aggregate rows should be present only as exact sum-check diagnostics."""
    variables = [
        "employment_income",
        "self_employment_income",
        "state_pension",
        "private_pension_income",
        "property_income",
        "dividend_income",
    ]
    for year in projections["year"].unique():
        year_df = projections[projections["year"] == year]
        aggregate_rows = year_df[_is_aggregate_row(year_df)]
        assert len(aggregate_rows) == 1
        aggregate = aggregate_rows.iloc[0]
        detailed = _without_aggregate(year_df)

        for variable in variables:
            for suffix in ["amount", "count"]:
                column = f"{variable}_{suffix}"
                assert aggregate[column] == pytest.approx(
                    detailed[column].sum(),
                    abs=len(detailed),
                )


def test_projection_keeps_top_open_ended_band(projections):
    """The real top band also has upper=inf and must not be treated as aggregate."""
    future = projections[projections["year"] == _SPI_YEAR + 1]
    top_band = future[
        (future["total_income_lower_bound"] == 1_000_000)
        & (future["total_income_upper_bound"] == float("inf"))
    ]
    assert len(top_band) == 1
    assert top_band.iloc[0]["dividend_income_amount"] > 0
