"""NTS vehicle ownership targets.

From the National Travel Survey 2024.
Source: https://www.gov.uk/government/statistics/national-travel-survey-2024
"""

from policyengine_uk_data.targets.schema import Target, Unit

_REF = "https://www.gov.uk/government/statistics/national-travel-survey-2024"

# NTS 2024: 22% no car, 44% one car, 34% two+ cars
NTS_NO_VEHICLE_RATE = 0.22
NTS_ONE_VEHICLE_RATE = 0.44
NTS_TWO_PLUS_VEHICLE_RATE = 0.34

# ~29.6m total UK households (from VOA/ONS council tax stock 2024)
_TOTAL_HOUSEHOLDS = 29.6e6


def get_targets() -> list[Target]:
    return [
        Target(
            name="nts/households_no_vehicle",
            variable="num_vehicles",
            source="nts",
            unit=Unit.COUNT,
            values={2024: _TOTAL_HOUSEHOLDS * NTS_NO_VEHICLE_RATE},
            is_count=True,
            reference_url=_REF,
        ),
        Target(
            name="nts/households_one_vehicle",
            variable="num_vehicles",
            source="nts",
            unit=Unit.COUNT,
            values={2024: _TOTAL_HOUSEHOLDS * NTS_ONE_VEHICLE_RATE},
            is_count=True,
            reference_url=_REF,
        ),
        Target(
            name="nts/households_two_plus_vehicles",
            variable="num_vehicles",
            source="nts",
            unit=Unit.COUNT,
            values={2024: _TOTAL_HOUSEHOLDS * NTS_TWO_PLUS_VEHICLE_RATE},
            is_count=True,
            reference_url=_REF,
        ),
    ]
