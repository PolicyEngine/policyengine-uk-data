"""Tests for the ONS Public Sector Employment calibration target.

The target constrains the simulated count of public-sector workers
(`employment_sector == PUBLIC`) towards the official ONS Public Sector
Employment (PSE) headcount. A 20% relative tolerance is accepted: the
FRS self-reported sector over-counts public employment, so calibration
only needs to bring the figure within a fifth of the official total.
"""

import pytest

from policyengine_uk_data.datasets.frs_release import CURRENT_FRS_RELEASE
from policyengine_uk_data.targets import get_all_targets
from policyengine_uk_data.targets.build_loss_matrix import _resolve_value
from policyengine_uk_data.targets.sources.ons_public_sector_employment import (
    get_targets,
)

# Accepted relative error between the (target and, after data generation,
# the simulated) public-sector headcount and the official ONS PSE figure.
ACCEPTED_RELATIVE_ERROR = 0.20

# Official ONS Public Sector Employment, UK (headcount), by year. Held
# independently of the source module so a wrong target value is caught.
ONS_PSE_HEADCOUNT = {
    2023: 5_900_000.0,
    2024: 5_940_000.0,
}

# Years the enhanced FRS fixture can represent (mirrors land value tests).
MODEL_CHECK_YEARS = sorted(
    {
        CURRENT_FRS_RELEASE.base_year,
        CURRENT_FRS_RELEASE.calibration_year,
    }
)


# ── Target structure ─────────────────────────────────────────────────


def test_get_targets_returns_one():
    """get_targets() should return the single public sector target."""
    assert len(get_targets()) == 1


def test_target_variable_and_metadata():
    """Target should count employment_sector from ONS."""
    target = get_targets()[0]
    assert target.name == "ons/public_sector_employment"
    assert target.variable == "employment_sector"
    assert target.source == "ons"
    assert target.is_count


def test_targets_in_registry():
    """The target should appear in the global registry."""
    names = {t.name for t in get_all_targets()}
    assert "ons/public_sector_employment" in names


# ── Target values ────────────────────────────────────────────────────


def test_target_values_within_20pct_of_ons():
    """Each target value is within the accepted 20% of the ONS PSE figure."""
    values = get_targets()[0].values
    for year, official in ONS_PSE_HEADCOUNT.items():
        assert year in values, f"missing target for {year}"
        rel_error = abs(values[year] / official - 1)
        assert rel_error <= ACCEPTED_RELATIVE_ERROR, (
            f"{year} target {values[year]:,.0f} differs from ONS PSE "
            f"{official:,.0f} by {rel_error:.1%} (>20%)."
        )


# ── Simulated total after data generation ────────────────────────────


@pytest.mark.parametrize("year", MODEL_CHECK_YEARS, ids=map(str, MODEL_CHECK_YEARS))
def test_public_sector_employment_total(enhanced_frs, baseline, year):
    """Weighted public-sector total is within 20% of the ONS PSE target.

    Runs against the generated enhanced FRS, whose national calibration
    now includes the public sector employment target. Skipped if the
    dataset predates the variable (rebuild with ``make data``).
    """
    if "employment_sector" not in enhanced_frs.person.columns:
        pytest.skip("dataset predates employment_sector; rebuild with `make data`")

    target = _resolve_value(get_targets()[0], year)
    assert target is not None, f"no target value resolvable for {year}"

    weights = baseline.calculate("household_weight", period=year).values
    sector = baseline.calculate("employment_sector", period=year).values
    is_public = (sector == "PUBLIC").astype(float)
    estimate = (baseline.map_result(is_public, "person", "household") * weights).sum()

    rel_error = abs(estimate / target - 1)
    assert rel_error < ACCEPTED_RELATIVE_ERROR, (
        f"public sector employment ({year}): expected {target:,.0f}, "
        f"got {estimate:,.0f} (relative error = {rel_error:.1%}, "
        f"tolerance = {ACCEPTED_RELATIVE_ERROR:.0%})"
    )
