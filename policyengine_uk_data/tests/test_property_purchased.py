"""
Test that property_purchased is set correctly in the enhanced FRS dataset.

The property_purchased variable should be stochastically set based on
UK housing transaction rates (~3.85% of households per year).

Sources:
- Transactions: HMRC 2024 - 1.1m/year
  https://www.gov.uk/government/statistics/monthly-property-transactions-completed-in-the-uk-with-value-40000-or-above
- Households: ONS 2024 - 28.6m
  https://www.ons.gov.uk/peoplepopulationandcommunity/birthsdeathsandmarriages/families/bulletins/familiesandhouseholds/2024
- Rate: 1.1m / 28.6m = 3.85%

Households' SDLT is checked against the OBR's SDLT forecast less the SDLT
policyengine-uk attributes to companies and other non-natural persons
(`corporate_sdlt`), the figure calibration targets (#564). Before the 3.85%
rate, every household was charged SDLT and the total was about £370bn.
"""

import os

import pytest

from policyengine_uk_data.targets.obr_sources import load_workbook
from policyengine_uk_data.targets.sources import obr

PROPERTY_PURCHASE_RATE = 0.0385


def test_property_purchased_rate(baseline):
    """Test that property_purchased rate is approximately 3.85%."""
    property_purchased = baseline.calculate("property_purchased", 2025).values

    n_households = len(property_purchased)
    true_count = property_purchased.sum()
    actual_rate = true_count / n_households

    # Rate should be approximately 3.85% (allow for random variation)
    target_rate = PROPERTY_PURCHASE_RATE
    tolerance = 0.02

    assert abs(actual_rate - target_rate) < tolerance, (
        f"property_purchased rate {actual_rate:.2%} is not close to target {target_rate:.2%}"
    )


def test_property_purchased_not_all_true(baseline):
    """Test that not all households have property_purchased = True."""
    property_purchased = baseline.calculate("property_purchased", 2025).values

    true_count = property_purchased.sum()
    n_households = len(property_purchased)

    # Should NOT be 100% True (the bug we fixed)
    assert true_count < n_households * 0.1, (
        f"Too many households have property_purchased=True ({true_count}/{n_households})"
    )


def test_property_purchased_has_some_true(baseline):
    """Test that some households have property_purchased = True."""
    property_purchased = baseline.calculate("property_purchased", 2025).values

    true_count = property_purchased.sum()

    # Should have some True values (realistic purchasing rate)
    assert true_count > 0, "No households have property_purchased=True"


# PR CI builds with TESTING=1 (32 calibration epochs instead of 512), so its
# bound only catches order-of-magnitude errors. Full builds get the strict one.
_SDLT_BOUNDS = (0.25, 4.0) if os.environ.get("TESTING") == "1" else (0.5, 1.5)


def _household_sdlt_target(year: int) -> float:
    """The OBR SDLT calibration target: the OBR's SDLT less corporate SDLT."""
    targets = obr._parse_receipts(load_workbook("receipts"))
    return {t.name: t for t in targets}["obr/sdlt"].values[year]


def test_household_sdlt_near_target(baseline):
    """Households' SDLT is near the OBR's SDLT less corporate SDLT.

    Compared with `stamp_duty_land_tax`, households' own SDLT: `expected_sdlt`
    also included `corporate_sdlt` before PolicyEngine/policyengine-uk#2235.
    """
    weight = baseline.calculate("household_weight", 2025).values
    household = (baseline.calculate("stamp_duty_land_tax", 2025).values * weight).sum()
    target = _household_sdlt_target(2025)
    low, high = _SDLT_BOUNDS

    assert low < household / target < high, (
        f"Households' SDLT is £{household / 1e9:.2f}bn against a target of "
        f"£{target / 1e9:.2f}bn (the OBR's SDLT less corporate SDLT); the "
        f"ratio must lie between {low} and {high}."
    )


def test_corporate_sdlt_is_the_statistic_the_target_subtracts(baseline):
    """With households on target, total SDLT is then the OBR's figure."""
    weight = baseline.calculate("household_weight", 2025).values
    corporate = (baseline.calculate("corporate_sdlt", 2025).values * weight).sum()
    assert corporate == pytest.approx(
        obr.corporate_sdlt_receipts([2025])[2025], rel=1e-6
    )
