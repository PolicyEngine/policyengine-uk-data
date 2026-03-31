"""Shared logic for imputing missing earnings percentiles in local areas.

ASHE data provides percentiles 10–90 for each area. Percentiles 91–98
and 100 are missing and must be imputed. The original approach used
ratio scaling from national reference values. This module adds
averaging (shrinkage) toward the national value to improve stability,
per issue #68.
"""

import pandas as pd

# National percentile points (total income before tax).
# Source: https://www.gov.uk/government/statistics/percentile-points-from-1-to-99-for-total-income-before-and-after-tax
REFERENCE_VALUES = {
    10: 15300,
    20: 18000,
    30: 20800,
    40: 23700,
    50: 27200,
    60: 31600,
    70: 37500,
    80: 46100,
    90: 62000,
    91: 65300,
    92: 69200,
    93: 74000,
    94: 79800,
    95: 87400,
    96: 97200,
    97: 111000,
    98: 137000,
    100: 199000,
}

PERCENTILE_COLUMNS = [
    "10 percentile",
    "20 percentile",
    "30 percentile",
    "40 percentile",
    "50 percentile",
    "60 percentile",
    "70 percentile",
    "80 percentile",
    "90 percentile",
    "91 percentile",
    "92 percentile",
    "93 percentile",
    "94 percentile",
    "95 percentile",
    "96 percentile",
    "97 percentile",
    "98 percentile",
    "100 percentile",
]


def fill_missing_percentiles(
    row,
    percentile_columns=PERCENTILE_COLUMNS,
    reference_values=REFERENCE_VALUES,
    national_weight: float = 0.5,
):
    """Fill missing percentile values using ratio scaling averaged with national values.

    For each missing percentile, the imputed value is a weighted average of:
    - the ratio-scaled estimate from the nearest known local percentile
    - the national reference value for that percentile

    Args:
        row: A pandas Series representing one area's percentile data.
        percentile_columns: List of column names for percentiles.
        reference_values: Dict mapping percentile number to national value.
        national_weight: Weight given to the national reference value
            (0.0 = pure ratio scaling as before, 0.5 = equal blend,
            1.0 = pure national value). Default 0.5.
    """
    known_values = {
        int(col.split()[0]): row[col]
        for col in percentile_columns
        if pd.notna(row[col])
    }

    if not known_values:
        return row

    known_percentiles = sorted(known_values.keys())

    for col in percentile_columns:
        percentile = int(col.split()[0])

        if pd.isna(row[col]):
            lower = max([p for p in known_percentiles if p < percentile], default=None)
            upper = min([p for p in known_percentiles if p > percentile], default=None)

            ratio_estimate = None
            if lower is not None:
                lower_ratio = reference_values[percentile] / reference_values[lower]
                ratio_estimate = row[f"{lower} percentile"] * lower_ratio
            elif upper is not None:
                upper_ratio = reference_values[percentile] / reference_values[upper]
                ratio_estimate = row[f"{upper} percentile"] * upper_ratio

            if ratio_estimate is not None:
                national_value = reference_values[percentile]
                row[col] = (
                    1 - national_weight
                ) * ratio_estimate + national_weight * national_value

    return row
