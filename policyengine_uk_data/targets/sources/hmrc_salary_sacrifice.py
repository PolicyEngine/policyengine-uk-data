"""HMRC salary sacrifice income tax and NICs relief targets.

Downloads Table 6.2 CSV from HMRC to get salary sacrifice IT relief
by tax rate band and NICs relief (employee + employer).

Source: https://assets.publishing.service.gov.uk/media/687a294e312ee8a5f0806b6d/Tables_6_1_and_6_2.csv
"""

import io
import logging

import pandas as pd
import requests

from policyengine_uk_data.targets.schema import Target, Unit
from policyengine_uk_data.targets.sources._common import (
    HEADERS,
    load_config,
    to_float,
)

logger = logging.getLogger(__name__)

# Uprate 3% pa for wage growth from the base year
_GROWTH = 1.03
_BASE_YEAR = 2024  # 2023-24 tax year → calendar 2024


def get_targets() -> list[Target]:
    config = load_config()
    ref = config["hmrc"]["salary_sacrifice_table_6"]
    targets = []

    try:
        r = requests.get(
            ref, headers=HEADERS, allow_redirects=True, timeout=30
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
            val = to_float(row["value_of_relief"])
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
            val = to_float(row["value_of_relief"])
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

    # Total salary sacrifice contributions (SPP Review 2025: £24bn base)
    _SS_CONTRIBUTIONS = {
        y: 24e9 * _GROWTH ** max(0, y - _BASE_YEAR)
        for y in range(_BASE_YEAR, 2030)
    }
    targets.append(
        Target(
            name="hmrc/salary_sacrifice_contributions",
            variable="pension_contributions_via_salary_sacrifice",
            source="hmrc",
            unit=Unit.GBP,
            values=_SS_CONTRIBUTIONS,
            reference_url=(
                "https://assets.publishing.service.gov.uk/media/"
                "67ce0e7c08e764d17a5d3c21/2025_SPP_Review.pdf"
            ),
        )
    )

    return targets
