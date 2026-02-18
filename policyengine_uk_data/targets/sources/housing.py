"""Housing affordability targets.

Total mortgage payments and private rent from ONS/English Housing Survey.

Sources:
- ONS PRHI: https://www.ons.gov.uk/economy/inflationandpriceindices/bulletins/privaterentandhousepricesuk/january2025
- English Housing Survey mortgage data
"""

from policyengine_uk_data.targets.schema import Target, Unit

# Estimated total annual housing costs (£)
# Private rent: avg £1,400/month × 12 × 4.7m private renters
# Mortgage: avg £1,100/month × 12 × 7.5m owner-occupiers with mortgage
_PRIVATE_RENT_TOTAL = 1_400 * 12 * 4.7e6
_MORTGAGE_TOTAL = 1_100 * 12 * 7.5e6


def get_targets() -> list[Target]:
    return [
        Target(
            name="housing/total_mortgage",
            variable="mortgage_capital_repayment",
            source="ons",
            unit=Unit.GBP,
            values={2025: _MORTGAGE_TOTAL},
            reference_url="https://www.ons.gov.uk/economy/inflationandpriceindices/bulletins/privaterentandhousepricesuk/january2025",
        ),
        Target(
            name="housing/rent_private",
            variable="rent",
            source="ons",
            unit=Unit.GBP,
            values={2025: _PRIVATE_RENT_TOTAL},
            reference_url="https://www.ons.gov.uk/economy/inflationandpriceindices/bulletins/privaterentandhousepricesuk/january2025",
        ),
    ]
