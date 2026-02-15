"""ONS savings interest income targets.

Downloads the HAXV timeseries from the ONS National Accounts:
D.41g — Households (S.14): Interest resources.

SPI significantly underestimates savings income because it only
captures taxable interest, not tax-free ISAs/NS&I.

Source: https://www.ons.gov.uk/economy/grossdomesticproductgdp/timeseries/haxv/ukea
"""

import logging

import requests

from policyengine_uk_data.targets.schema import Target, Unit

logger = logging.getLogger(__name__)

_API_URL = "https://www.ons.gov.uk/economy/grossdomesticproductgdp/timeseries/haxv/ukea/data"
_REF = "https://www.ons.gov.uk/economy/grossdomesticproductgdp/timeseries/haxv/ukea"
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
    ),
}


def get_targets() -> list[Target]:
    try:
        r = requests.get(
            _API_URL, headers=_HEADERS, allow_redirects=True, timeout=30
        )
        r.raise_for_status()
        data = r.json()

        values = {}
        for item in data.get("years", []):
            year = int(item["year"])
            if 2018 <= year <= 2029:
                values[year] = float(item["value"]) * 1e6

        # Hold flat from last actual year for projections
        if values:
            last_year = max(values.keys())
            last_val = values[last_year]
            for y in range(last_year + 1, 2030):
                values[y] = last_val

        if values:
            return [
                Target(
                    name="ons/savings_interest_income",
                    variable="savings_interest_income",
                    source="ons",
                    unit=Unit.GBP,
                    values=values,
                    reference_url=_REF,
                )
            ]

    except Exception as e:
        logger.error("Failed to download ONS savings timeseries: %s", e)

    return []
