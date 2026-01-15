"""Test Scotland babies under 1 calibration target.

Source: NRS Vital Events Reference Tables 2024
https://www.nrscotland.gov.uk/publications/vital-events-reference-tables-2024/
Scotland had 45,763 live births in 2024.
"""

import pytest


@pytest.mark.xfail(
    reason="Will pass after recalibration with new scotland_babies_under_1 target"
)
def test_scotland_babies_under_1(baseline):
    """Test that babies under 1 in Scotland matches NRS birth statistics.

    Target: ~46,000 babies under 1 (based on ~46k annual births)
    Source: NRS Vital Events 2024 reports 45,763 births
    """
    region = baseline.calculate("region", map_to="person", period=2025)
    age = baseline.calculate("age", map_to="person", period=2025).values
    person_weight = baseline.calculate(
        "person_weight", map_to="person", period=2025
    ).values

    scotland_babies = (region.values == "SCOTLAND") & (age < 1)
    total_babies = (person_weight * scotland_babies).sum()

    TARGET = 46_000  # NRS Vital Events 2024: 45,763 births
    TOLERANCE = 0.15  # 15% tolerance

    assert abs(total_babies / TARGET - 1) < TOLERANCE, (
        f"Expected ~{TARGET/1000:.0f}k babies under 1 in Scotland, "
        f"got {total_babies/1000:.0f}k ({total_babies/TARGET*100:.0f}% of target)"
    )
