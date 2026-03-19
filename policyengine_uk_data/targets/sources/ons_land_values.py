"""ONS National Balance Sheet land value targets.

Aggregate land values from the ONS National Balance Sheet 2025.
The ONS directly measured total UK land at £7.1 trillion for 2024,
broken down into household land (£4.31tn in 2020) and corporate
land (£1.76tn in 2020).

Source: https://www.ons.gov.uk/economy/nationalaccounts/uksectoraccounts/bulletins/nationalbalancesheet/2025
"""

from policyengine_uk_data.targets.schema import Target, Unit

# ONS National Balance Sheet 2025
# 2020 breakdown: household £4.31tn, corporate £1.76tn, total £6.07tn
# 2024 measured total: £7.1tn
# We scale the 2020 household/corporate split proportionally to match
# the 2024 measured total, then hold constant for 2025-2026 (no newer
# ONS measurement available).

_ONS_2020_HOUSEHOLD = 4.31e12
_ONS_2020_CORPORATE = 1.76e12
_ONS_2020_TOTAL = _ONS_2020_HOUSEHOLD + _ONS_2020_CORPORATE
_ONS_2024_TOTAL = 7.1e12

# Scale 2020 split to 2024 measured total
_SCALE = _ONS_2024_TOTAL / _ONS_2020_TOTAL
_ONS_2024_HOUSEHOLD = _ONS_2020_HOUSEHOLD * _SCALE
_ONS_2024_CORPORATE = _ONS_2020_CORPORATE * _SCALE

_REF_URL = "https://www.ons.gov.uk/economy/nationalaccounts/uksectoraccounts/bulletins/nationalbalancesheet/2025"


def get_targets() -> list[Target]:
    return [
        Target(
            name="ons/household_land_value",
            variable="household_land_value",
            source="ons",
            unit=Unit.GBP,
            values={
                2024: _ONS_2024_HOUSEHOLD,
                2025: _ONS_2024_HOUSEHOLD,
                2026: _ONS_2024_HOUSEHOLD,
            },
            reference_url=_REF_URL,
        ),
        Target(
            name="ons/corporate_land_value",
            variable="corporate_land_value",
            source="ons",
            unit=Unit.GBP,
            values={
                2024: _ONS_2024_CORPORATE,
                2025: _ONS_2024_CORPORATE,
                2026: _ONS_2024_CORPORATE,
            },
            reference_url=_REF_URL,
        ),
        Target(
            name="ons/land_value",
            variable="land_value",
            source="ons",
            unit=Unit.GBP,
            values={
                2024: _ONS_2024_TOTAL,
                2025: _ONS_2024_TOTAL,
                2026: _ONS_2024_TOTAL,
            },
            reference_url=_REF_URL,
        ),
    ]
