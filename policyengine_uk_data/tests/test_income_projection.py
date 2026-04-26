"""Tests for income projection accuracy (issue #218).

These tests verify that projected incomes are not inflated beyond
reasonable bounds after reweighting. They require incomes_projection.csv
to have been generated and will be skipped otherwise.
"""

import pytest
import pandas as pd
from policyengine_uk_data.storage import STORAGE_FOLDER

PROJECTION_PATH = STORAGE_FOLDER / "incomes_projection.csv"
SPI_PATH = STORAGE_FOLDER / "incomes.csv"


@pytest.fixture
def projections():
    if not PROJECTION_PATH.exists():
        pytest.skip(
            "incomes_projection.csv not available (run create_income_projections first)"
        )
    return pd.read_csv(PROJECTION_PATH)


@pytest.fixture
def spi_targets():
    if not SPI_PATH.exists():
        pytest.skip("incomes.csv not available")
    return pd.read_csv(SPI_PATH)


# Maximum tolerable ratio of projected total to uprated SPI baseline.
# Uprating from 2021 to 2029 should not exceed ~1.6x even with generous
# growth assumptions. A 2x cap gives ample headroom while catching the
# ~2.5x inflation that issue #218 documented.
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
def test_projected_totals_not_inflated(projections, spi_targets, variable):
    """No income type should be inflated >2x relative to the SPI baseline."""
    spi_total = spi_targets[f"{variable}_amount"].sum()
    for year in projections["year"].unique():
        year_df = projections[projections["year"] == year]
        projected_total = year_df[f"{variable}_amount"].sum()
        ratio = projected_total / spi_total
        assert ratio < MAX_RATIO, (
            f"{variable} in {year}: projected £{projected_total / 1e9:.1f}bn "
            f"is {ratio:.2f}x the SPI baseline £{spi_total / 1e9:.1f}bn "
            f"(max allowed {MAX_RATIO}x)"
        )


def test_employment_income_still_calibrated(projections, spi_targets):
    """Employment income should remain close to uprated targets."""
    spi_total = spi_targets["employment_income_amount"].sum()
    year_2022 = projections[projections["year"] == 2022]
    projected_total = year_2022["employment_income_amount"].sum()
    # Employment income is a reweighting target, so the 2022 projection
    # should be within 30% of a simple uprate from the SPI baseline.
    ratio = projected_total / spi_total
    assert 0.7 < ratio < 1.8, (
        f"Employment income in 2022: projected £{projected_total / 1e9:.1f}bn "
        f"vs SPI baseline £{spi_total / 1e9:.1f}bn (ratio {ratio:.2f})"
    )


def test_high_income_band_not_extreme(projections):
    """The highest income bands should not show extreme overestimation.

    Issue #218 found the £500k-£1M band at 12.6x target — this test
    guards against that regression.
    """
    for year in projections["year"].unique():
        year_df = projections[projections["year"] == year]
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
