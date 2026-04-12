"""Country-level housing targets for Scotland and Wales.

Adds private-rented stock and private-rent spend anchors for the two
countries that are currently most weakly identified in constituency
calibration.

Sources:
- Wales dwelling stock by tenure, 31 March 2024:
  https://www.gov.wales/dwelling-stock-estimates-31-march-2024-html
- Scotland stock by tenure workbook, 2023:
  https://www.gov.scot/binaries/content/documents/govscot/publications/statistics/2018/09/housing-statistics-stock-by-tenure/documents/stock-by-tenure-2017/stock-by-tenure-2017/govscot%3Adocument/Stock%2Bby%2Btenure.xlsx
- ONS private rents bulletin, May 2025:
  https://www.ons.gov.uk/economy/inflationandpriceindices/bulletins/privaterentandhousepricesuk/may2025
"""

from policyengine_uk_data.targets.schema import GeographicLevel, Target, Unit

_ONS_RENT_REF = (
    "https://www.ons.gov.uk/economy/inflationandpriceindices/"
    "bulletins/privaterentandhousepricesuk/may2025"
)
_WALES_STOCK_REF = (
    "https://www.gov.wales/dwelling-stock-estimates-31-march-2024-html"
)
_SCOTLAND_STOCK_REF = (
    "https://www.gov.scot/binaries/content/documents/govscot/publications/"
    "statistics/2018/09/housing-statistics-stock-by-tenure/documents/"
    "stock-by-tenure-2017/stock-by-tenure-2017/govscot%3Adocument/"
    "Stock%2Bby%2Btenure.xlsx"
)

_WALES_PRIVATE_RENTED_STOCK_2025 = 200_700
_SCOTLAND_PRIVATE_RENTED_STOCK_2025 = 357_706

_WALES_AVG_MONTHLY_RENT_2025 = 795
_SCOTLAND_AVG_MONTHLY_RENT_2025 = 999

_WALES_PRIVATE_RENT_TOTAL_2025 = (
    _WALES_PRIVATE_RENTED_STOCK_2025 * _WALES_AVG_MONTHLY_RENT_2025 * 12
)
_SCOTLAND_PRIVATE_RENT_TOTAL_2025 = (
    _SCOTLAND_PRIVATE_RENTED_STOCK_2025
    * _SCOTLAND_AVG_MONTHLY_RENT_2025
    * 12
)


def get_targets() -> list[Target]:
    return [
        Target(
            name="gov_wales/tenure_wales_rented_privately",
            variable="tenure_type",
            source="welsh_government",
            unit=Unit.COUNT,
            geographic_level=GeographicLevel.COUNTRY,
            geo_code="W",
            geo_name="Wales",
            values={2025: _WALES_PRIVATE_RENTED_STOCK_2025},
            is_count=True,
            reference_url=_WALES_STOCK_REF,
        ),
        Target(
            name="gov_scot/tenure_scotland_rented_privately",
            variable="tenure_type",
            source="scottish_government",
            unit=Unit.COUNT,
            geographic_level=GeographicLevel.COUNTRY,
            geo_code="S",
            geo_name="Scotland",
            values={2025: _SCOTLAND_PRIVATE_RENTED_STOCK_2025},
            is_count=True,
            reference_url=_SCOTLAND_STOCK_REF,
        ),
        Target(
            name="housing/rent_private/wales",
            variable="rent",
            source="ons",
            unit=Unit.GBP,
            geographic_level=GeographicLevel.COUNTRY,
            geo_code="W",
            geo_name="Wales",
            values={2025: _WALES_PRIVATE_RENT_TOTAL_2025},
            reference_url=_ONS_RENT_REF,
        ),
        Target(
            name="housing/rent_private/scotland",
            variable="rent",
            source="ons",
            unit=Unit.GBP,
            geographic_level=GeographicLevel.COUNTRY,
            geo_code="S",
            geo_name="Scotland",
            values={2025: _SCOTLAND_PRIVATE_RENT_TOTAL_2025},
            reference_url=_ONS_RENT_REF,
        ),
    ]
