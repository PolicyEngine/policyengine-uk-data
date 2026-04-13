"""Country-level private-rent anchors for constituency calibration.

These are used directly by the constituency loss builder rather than the
general target registry, because constituency calibration uses a bespoke
matrix constructor.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


_PRIVATE_RENT_TARGETS = {
    "WALES": {
        "private_renter_households": 200_700,
        "annual_private_rent": 795 * 12,
    },
    "SCOTLAND": {
        "private_renter_households": 357_706,
        "annual_private_rent": 999 * 12,
    },
}

_CODE_TO_COUNTRY = {
    "W": "WALES",
    "S": "SCOTLAND",
}


def add_private_rent_targets(
    matrix: pd.DataFrame,
    y: pd.DataFrame,
    age_targets: pd.DataFrame,
    *,
    country: np.ndarray,
    tenure_type: np.ndarray,
    rent: np.ndarray,
) -> None:
    """Append Wales/Scotland private-rent count and amount targets.

    Country totals are allocated across 2010 constituencies in proportion to
    their official age-target population shares within each country.
    """

    constituency_population = age_targets.filter(like="age/").sum(axis=1)
    constituency_country = age_targets["code"].str[0].map(_CODE_TO_COUNTRY)
    private_renter = tenure_type == "RENT_PRIVATELY"

    for country_name, target in _PRIVATE_RENT_TARGETS.items():
        area_mask = constituency_country == country_name
        country_population = constituency_population.where(area_mask, 0).sum()
        if country_population <= 0:
            raise ValueError(
                f"No constituency population available for {country_name} housing targets"
            )

        share = np.where(area_mask, constituency_population / country_population, 0.0)
        in_country_private_rent = (country == country_name) & private_renter
        prefix = country_name.lower()

        matrix[f"housing/{prefix}_private_renter_households"] = (
            in_country_private_rent
        ).astype(float)
        matrix[f"housing/{prefix}_private_rent_amount"] = np.where(
            in_country_private_rent,
            rent,
            0.0,
        )

        y[f"housing/{prefix}_private_renter_households"] = (
            share * target["private_renter_households"]
        )
        y[f"housing/{prefix}_private_rent_amount"] = (
            share * target["private_renter_households"] * target["annual_private_rent"]
        )
