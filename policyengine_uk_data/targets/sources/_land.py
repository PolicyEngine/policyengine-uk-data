"""Shared ONS land target series used across uk-data targets and tests.

This stays in uk-data for now because uk-data CI consumes the released
policyengine-uk package, not the in-flight repository branch. Once the
country package release includes the refreshed land series, this helper
can switch to importing that series directly.
"""

from __future__ import annotations

_ONS_2020_HOUSEHOLD = 4_309_138_000_000
_ONS_2020_CORPORATE = 1_757_818_000_000
_ONS_2020_TOTAL = _ONS_2020_HOUSEHOLD + _ONS_2020_CORPORATE
_HOUSEHOLD_SHARE = _ONS_2020_HOUSEHOLD / _ONS_2020_TOTAL
_CORPORATE_SHARE = _ONS_2020_CORPORATE / _ONS_2020_TOTAL

_OBSERVED_TOTAL_LAND_VALUES = {
    2021: 7_106_785_000_000,
    2022: 7_138_696_000_000,
    2023: 6_756_315_000_000,
    2024: 7_100_000_000_000,
}

_REF_URL = (
    "https://www.ons.gov.uk/economy/nationalaccounts/"
    "uksectoraccounts/bulletins/nationalbalancesheet/2025"
)


def _split_total_land(total_land: float) -> tuple[float, float]:
    """Split aggregate land by the latest direct household/corporate shares."""
    return (
        total_land * _HOUSEHOLD_SHARE,
        total_land * _CORPORATE_SHARE,
    )


HOUSEHOLD_LAND_VALUES = {
    year: _split_total_land(total_land)[0]
    for year, total_land in _OBSERVED_TOTAL_LAND_VALUES.items()
}

CORPORATE_LAND_VALUES = {
    year: _split_total_land(total_land)[1]
    for year, total_land in _OBSERVED_TOTAL_LAND_VALUES.items()
}

TOTAL_LAND_VALUES = {
    **_OBSERVED_TOTAL_LAND_VALUES,
    2025: _OBSERVED_TOTAL_LAND_VALUES[2024],
    2026: _OBSERVED_TOTAL_LAND_VALUES[2024],
}

HOUSEHOLD_LAND_VALUES.update(
    {
        2025: HOUSEHOLD_LAND_VALUES[2024],
        2026: HOUSEHOLD_LAND_VALUES[2024],
    }
)
CORPORATE_LAND_VALUES.update(
    {
        2025: CORPORATE_LAND_VALUES[2024],
        2026: CORPORATE_LAND_VALUES[2024],
    }
)
