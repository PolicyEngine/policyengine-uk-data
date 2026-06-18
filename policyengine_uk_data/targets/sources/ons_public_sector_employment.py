"""ONS Public Sector Employment (PSE) target.

The FRS self-reported employer sector (`mjobsect` -> `employment_sector`)
over-counts public-sector employment relative to the official ONS PSE
headcount, so this adds a national calibration target for the number of
people whose main job is in the public sector
(`employment_sector == PUBLIC`).

PSE measures the institutional public sector (central government, local
government and public corporations) - i.e. NHS, state schools, councils,
civil service and the armed forces - so it is the right official total for
the whole-public-sector `employment_sector` flag, not the much narrower
SIC division 84 ("public administration and defence").

Source: ONS Public Sector Employment, UK (headcount, not seasonally
adjusted). Headline UK totals: ~5.90m (2023), ~5.94m (2024).
"""

from policyengine_uk_data.targets.schema import (
    GeographicLevel,
    Target,
    Unit,
)

_REF = (
    "https://www.ons.gov.uk/employmentandlabourmarket/peopleinwork/"
    "publicsectorpersonnel/bulletins/publicsectoremployment/latest"
)

# ONS PSE UK total headcount (people), by calendar year.
_VALUES = {
    2023: 5_900_000.0,
    2024: 5_940_000.0,
}


def get_targets() -> list[Target]:
    return [
        Target(
            name="ons/public_sector_employment",
            variable="employment_sector",
            source="ons",
            unit=Unit.COUNT,
            geographic_level=GeographicLevel.NATIONAL,
            geo_code="K02000001",
            geo_name="United Kingdom",
            values=dict(_VALUES),
            is_count=True,
            reference_url=_REF,
        )
    ]
