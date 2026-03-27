"""Housing affordability targets.

Total mortgage payments, private rent, and social rent.

Sources:
- ONS PRHI Feb 2026: UK avg private rent £1,374/month
  https://www.ons.gov.uk/economy/inflationandpriceindices/bulletins/privaterentandhousepricesuk/march2026
- English Housing Survey 2023-24: avg social rent £118/week
  https://www.gov.uk/government/statistics/english-housing-survey-2023-to-2024-rented-sectors
- EHS + devolved stats: ~5.4m private renters, ~5.0m social renters UK-wide
"""

from policyengine_uk_data.targets.schema import Target, Unit

# Private rent: ONS PRHI UK avg £1,374/month × 5.4m UK private renters
_PRIVATE_RENT_TOTAL = 1_374 * 12 * 5.4e6  # ~£89bn

# Mortgage: avg £1,100/month × 7.5m owner-occupiers with mortgage
_MORTGAGE_TOTAL = 1_100 * 12 * 7.5e6  # ~£99bn

# Social rent: EHS 2023-24 mean £118/week × 52 × 5.0m UK social renters
_SOCIAL_RENT_TOTAL = 118 * 52 * 5.0e6  # ~£30.7bn

_EHS_REF = (
    "https://www.gov.uk/government/statistics/"
    "english-housing-survey-2023-to-2024-rented-sectors"
)
_ONS_RENT_REF = (
    "https://www.ons.gov.uk/economy/inflationandpriceindices/"
    "bulletins/privaterentandhousepricesuk/march2026"
)


def get_targets() -> list[Target]:
    return [
        Target(
            name="housing/total_mortgage",
            variable="mortgage_capital_repayment",
            source="ons",
            unit=Unit.GBP,
            values={2025: _MORTGAGE_TOTAL},
            reference_url=_ONS_RENT_REF,
        ),
        Target(
            name="housing/rent_private",
            variable="rent",
            source="ehs",
            unit=Unit.GBP,
            values={2025: _PRIVATE_RENT_TOTAL},
            reference_url=_ONS_RENT_REF,
        ),
        Target(
            name="housing/rent_social",
            variable="rent",
            source="ehs",
            unit=Unit.GBP,
            values={2025: _SOCIAL_RENT_TOTAL},
            reference_url=_EHS_REF,
        ),
    ]
