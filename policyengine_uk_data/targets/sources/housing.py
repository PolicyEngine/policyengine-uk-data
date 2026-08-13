"""Housing affordability targets.

Total mortgage payments, private rent, and social rent.

All three targets are set for 2025 and should be anchored to comparable
vintages. The social rent basis previously used EHS 2023-24 while the
private basis used ONS PRHI from February 2026, understating the social
target by roughly 10% relative to its own year.

Sources:
- ONS PRHI Feb 2026: UK avg private rent £1,374/month
  https://www.ons.gov.uk/economy/inflationandpriceindices/bulletins/privaterentandhousepricesuk/march2026
- English Housing Survey 2024-25: avg social rent £129/week (local authority
  £120, housing association £134)
  https://www.gov.uk/government/statistics/chapters-for-english-housing-survey-2024-to-2025-headline-findings-on-demographics-and-household-resilience/chapter-2-housing-costs-and-affordability
- EHS + devolved stats: ~5.4m private renters, ~5.0m social renters UK-wide

Known limitation: these are totals only, so an error in the household count
and an offsetting error in the mean cannot be detected. At the 1.40.3
dataset the private target is met (1.04x) with 6.99m renters at £256/week
against a basis of 5.4m at £317/week — the total matches by cancellation.
See PolicyEngine/policyengine-uk-data#454.
"""

from policyengine_uk_data.targets.schema import Target, Unit

# Private rent: ONS PRHI UK avg £1,374/month × 5.4m UK private renters
_PRIVATE_RENT_TOTAL = 1_374 * 12 * 5.4e6  # ~£89bn

# Mortgage: avg £1,100/month × 7.5m owner-occupiers with mortgage
_MORTGAGE_TOTAL = 1_100 * 12 * 7.5e6  # ~£99bn

# Social rent: EHS 2024-25 mean £129/week × 52 × 5.0m UK social renters
_SOCIAL_RENT_TOTAL = 129 * 52 * 5.0e6  # ~£33.5bn

_EHS_REF = (
    "https://www.gov.uk/government/statistics/"
    "chapters-for-english-housing-survey-2024-to-2025-headline-findings-on-"
    "demographics-and-household-resilience/chapter-2-housing-costs-and-"
    "affordability"
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
