"""Regional household land value targets.

Historical behaviour — still the default — sets each region's share of
the national household land total proportional to its property wealth
(``dwellings × avg_house_price``). That is systematically wrong for LVT
analysis because the land-to-property ratio is not constant across
regions: London dwellings are overwhelmingly land value, while rural
and northern dwellings are a much higher proportion building value.
See issue #357.

This module now supports passing a per-region land-to-property ratio so
the targets can reflect that variation once the ratios are sourced. The
plumbing + tests land here; sourcing the real numbers (VOA dwelling
value minus ONS reconstruction cost, or Savills residential land value
estimates) and supplying them to callers is deliberately a separate
follow-up — the ratios are load-bearing assumptions that need reviewer
sign-off from the modelling team.
"""

from __future__ import annotations

import pandas as pd

from policyengine_uk_data.targets.schema import (
    GeographicLevel,
    Target,
    Unit,
)
from policyengine_uk_data.targets.sources._land import (
    HOUSEHOLD_LAND_VALUES,
    _REF_URL,
)
from policyengine_uk_data.targets.sources._common import STORAGE


def _regional_shares_from_frame(
    df: pd.DataFrame,
    land_to_property_ratio: dict[str, float] | None = None,
) -> dict[str, float]:
    """Pure computation of region → share-of-national-household-land.

    Args:
        df: must contain ``region``, ``dwellings`` and ``avg_house_price``.
        land_to_property_ratio: optional ``region → ratio`` mapping. When
            supplied, each region's contribution to the national total is
            weighted by this ratio, so a region with a higher land share
            of property value takes a larger slice of the national total
            than its property-wealth share alone. When ``None`` (default)
            the shares are proportional to property wealth, reproducing
            the pre-#357 behaviour exactly.

    Returns:
        Dict of region → share. Shares sum to 1 by construction.

    Raises:
        KeyError: if a supplied ratio mapping is missing any region in
            ``df``. A silently-zeroed region would quietly misallocate
            the national total elsewhere, which is exactly the class of
            bug #357 is about.
        ValueError: if the weighted land-value totals come out to zero
            (all ratios zero, or empty DataFrame).
    """
    property_wealth = df["dwellings"] * df["avg_house_price"]
    if land_to_property_ratio is None:
        ratios = pd.Series(1.0, index=df.index)
    else:
        regions_in_df = set(df["region"])
        missing = regions_in_df - set(land_to_property_ratio)
        if missing:
            raise KeyError(
                f"land_to_property_ratio is missing regions: {sorted(missing)}"
            )
        ratios = df["region"].map(land_to_property_ratio)
    land_value_estimate = property_wealth * ratios.values
    total = land_value_estimate.sum()
    if total <= 0:
        raise ValueError(
            "Regional land-value estimates sum to zero; check that "
            "``dwellings``, ``avg_house_price`` and any supplied ratios "
            "are strictly positive."
        )
    return dict(zip(df["region"], land_value_estimate / total))


def _compute_regional_shares(
    land_to_property_ratio: dict[str, float] | None = None,
) -> dict[str, float]:
    """Split household land totals across regions using fixed 2025 shares.

    See ``_regional_shares_from_frame`` for the ratio semantics.
    """
    csv_path = STORAGE / "regional_land_values.csv"
    df = pd.read_csv(csv_path)
    return _regional_shares_from_frame(df, land_to_property_ratio)


def _compute_regional_targets(
    land_to_property_ratio: dict[str, float] | None = None,
) -> dict[str, dict[int, float]]:
    """Scale regional shares by the national household-land series."""
    shares = _compute_regional_shares(land_to_property_ratio)
    return {
        region: {
            year: share * HOUSEHOLD_LAND_VALUES[year] for year in HOUSEHOLD_LAND_VALUES
        }
        for region, share in shares.items()
    }


def get_targets(
    land_to_property_ratio: dict[str, float] | None = None,
) -> list[Target]:
    csv_path = STORAGE / "regional_land_values.csv"
    if not csv_path.exists():
        return []

    regional = _compute_regional_targets(land_to_property_ratio)
    targets = []

    for region, value in regional.items():
        targets.append(
            Target(
                name=f"ons/household_land_value/{region}",
                variable="household_land_value",
                source="ons",
                unit=Unit.GBP,
                geographic_level=GeographicLevel.REGION,
                geo_name=region,
                values=value,
                reference_url=_REF_URL,
            )
        )

    return targets
