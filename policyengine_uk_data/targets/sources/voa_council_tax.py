"""VOA council tax band targets.

Council tax band counts (A-H + total) by region from VOA stock of
properties data.

Source: https://www.gov.uk/government/statistics/council-tax-stock-of-properties-2024
Scotland: https://www.gov.scot/publications/council-tax-datasets/
"""

import pandas as pd
from pathlib import Path

from policyengine_uk_data.targets.schema import (
    GeographicLevel,
    Target,
    Unit,
)

_STORAGE = Path(__file__).parents[2] / "storage"
_REF = "https://www.gov.uk/government/statistics/council-tax-stock-of-properties-2024"


def get_targets() -> list[Target]:
    """Build council tax band targets from the CSV."""
    csv_path = _STORAGE / "council_tax_bands_2024.csv"
    if not csv_path.exists():
        return []

    ct_data = pd.read_csv(csv_path)
    targets = []

    for _, row in ct_data.iterrows():
        region = row["Region"]
        for band in ["A", "B", "C", "D", "E", "F", "G", "H"]:
            targets.append(
                Target(
                    name=f"voa/council_tax/{region}/{band}",
                    variable="council_tax_band",
                    source="voa",
                    unit=Unit.COUNT,
                    geographic_level=GeographicLevel.REGION,
                    geo_name=region,
                    values={2024: float(row[band])},
                    is_count=True,
                    reference_url=_REF,
                )
            )
        # Total row
        targets.append(
            Target(
                name=f"voa/council_tax/{region}/total",
                variable="council_tax_band",
                source="voa",
                unit=Unit.COUNT,
                geographic_level=GeographicLevel.REGION,
                geo_name=region,
                values={2024: float(row["Total"])},
                is_count=True,
                reference_url=_REF,
            )
        )

    return targets
