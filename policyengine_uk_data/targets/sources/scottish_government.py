"""Scottish Government targets.

Scottish Child Payment spend from Scottish Budget.
Source: https://www.gov.scot/publications/scottish-budget-2026-2027/pages/6/
"""

from policyengine_uk_data.targets.schema import (
    GeographicLevel,
    Target,
    Unit,
)


def get_targets() -> list[Target]:
    # Scottish Child Payment from Scottish Budget 2026-27 Table 5.08
    scp_spend = {
        2024: 455.8e6,
        2025: 471.0e6,
        2026: 484.8e6,
    }
    # Extrapolate other years at 3% growth
    for y in range(2027, 2030):
        scp_spend[y] = 471.0e6 * (1.03 ** (y - 2025))

    return [
        Target(
            name="sss/scottish_child_payment",
            variable="scottish_child_payment",
            source="scottish_government",
            unit=Unit.GBP,
            geographic_level=GeographicLevel.COUNTRY,
            geo_code="S",
            geo_name="Scotland",
            values=scp_spend,
            reference_url="https://www.gov.scot/publications/scottish-budget-2026-2027/pages/6/",
        )
    ]
