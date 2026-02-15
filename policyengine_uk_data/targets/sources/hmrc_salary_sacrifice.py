"""HMRC salary sacrifice income tax and NICs relief targets.

Downloads Table 6.2 CSV from HMRC to get salary sacrifice IT relief
by tax rate band and NICs relief (employee + employer).

Source: https://assets.publishing.service.gov.uk/media/687a294e312ee8a5f0806b6d/Tables_6_1_and_6_2.csv
"""

import io
import logging
from pathlib import Path

import pandas as pd
import requests
import yaml

from policyengine_uk_data.targets.schema import Target, Unit

logger = logging.getLogger(__name__)

_SOURCES_YAML = Path(__file__).parent.parent / "sources.yaml"
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
    ),
}

# Uprate 3% pa for wage growth from the base year
_GROWTH = 1.03
_BASE_YEAR = 2024  # 2023-24 tax year → calendar 2024


def _load_config():
    with open(_SOURCES_YAML) as f:
        return yaml.safe_load(f)


def _to_float(val) -> float:
    """Convert CSV value to float, handling suppressed '[z]' etc."""
    if isinstance(val, (int, float)):
        return float(val)
    try:
        return float(val)
    except (ValueError, TypeError):
        return 0.0


def get_targets() -> list[Target]:
    config = _load_config()
    ref = config["hmrc"]["salary_sacrifice_table_6"]
    targets = []

    try:
        r = requests.get(
            ref, headers=_HEADERS, allow_redirects=True, timeout=30
        )
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.content.decode("utf-8-sig")))

        ss = df[df["contribution_type"] == "Salary sacrificed contributions"]

        # IT relief by tax band
        ss_it = ss[
            (ss["income_tax_nics"] == "Income Tax")
            & (ss["sector_scheme"] == "Total")
            & (ss["scheme_type"] == "Total")
        ]
        for _, row in ss_it.iterrows():
            rate = row["tax_rate"]
            val = _to_float(row["value_of_relief"])
            if val <= 0:
                continue
            rate_key = rate.lower().replace(" ", "_")
            base = val * 1e6
            targets.append(
                Target(
                    name=f"hmrc/salary_sacrifice_it_relief_{rate_key}",
                    variable="income_tax",
                    source="hmrc",
                    unit=Unit.GBP,
                    values={
                        y: base * _GROWTH ** max(0, y - _BASE_YEAR)
                        for y in range(_BASE_YEAR, 2032)
                    },
                    reference_url=ref,
                )
            )

        # NICs relief (employee + employer)
        ss_nics = ss[
            (ss["income_tax_nics"] == "NICs")
            & (ss["sector_scheme"] == "Total")
            & (ss["scheme_type"] == "Total")
        ]
        for _, row in ss_nics.iterrows():
            nics_class = row["nics_relief_class"]
            val = _to_float(row["value_of_relief"])
            if val <= 0:
                continue
            if "employee" in str(nics_class).lower():
                name = "hmrc/salary_sacrifice_employee_nics_relief"
                variable = "ni_employee"
            elif "employer" in str(nics_class).lower():
                name = "hmrc/salary_sacrifice_employer_nics_relief"
                variable = "ni_employer"
            else:
                continue

            # Only take the first (Total scheme) row for each class
            existing = {t.name for t in targets}
            if name in existing:
                continue

            base = val * 1e6
            targets.append(
                Target(
                    name=name,
                    variable=variable,
                    source="hmrc",
                    unit=Unit.GBP,
                    values={
                        y: base * _GROWTH ** max(0, y - _BASE_YEAR)
                        for y in range(_BASE_YEAR, 2032)
                    },
                    reference_url=ref,
                )
            )

    except Exception as e:
        logger.error(
            "Failed to download/parse HMRC salary sacrifice CSV: %s", e
        )

    return targets
