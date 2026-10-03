"""HMRC salary sacrifice income tax and NICs relief targets.

Reads HMRC private pension statistics Tables 6.1 and 6.2 (tidy CSV): income
tax relief on salary-sacrificed pension contributions by marginal rate, and
the Class 1 primary (employee) and secondary (employer) NICs relief on them.

GOV.UK withdraws a release's assets when HMRC publishes the next one: the
July 2025 CSV has returned 410 Gone since the July 2026 release, and the
broad ``except`` this module used to have turned that into calibration
builds that silently lacked these targets. A failed download now falls back
to the copy committed in storage (the same release as ``sources.yaml``), and
a table that no longer has the expected rows raises instead of returning
fewer targets.

Source: https://www.gov.uk/government/statistics/personal-and-stakeholder-pensions-statistics
"""

import io
import logging
import re

import pandas as pd
import requests

from policyengine_uk_data.targets.schema import Target, Unit
from policyengine_uk_data.targets.sources._common import (
    HEADERS,
    STORAGE,
    load_config,
    to_float,
)

logger = logging.getLogger(__name__)

# Tables 6.1 and 6.2 from the July 2026 release (tax year 2024-25), as
# downloaded from the sources.yaml URL. Replace both together.
FALLBACK_CSV = STORAGE / "hmrc_pension_relief_tables_6_1_6_2_2024_25.csv"

# Uprate 3% pa for wage growth from the base year
_GROWTH = 1.03
_LAST_YEAR = 2031

# Table 6.1 rate rows → target name suffix. HMRC counts Scottish starter
# and intermediate rate relief as basic rate.
_IT_RATES = {
    "Basic Rate": "basic_rate",
    "Higher Rate": "higher_rate",
    "Additional Rate": "additional_rate",
    "Total": "total",
}
# Table 6.2 NICs classes → (target name, PolicyEngine variable)
_NICS_CLASSES = {
    "Class 1 Primary (employee)": (
        "hmrc/salary_sacrifice_employee_nics_relief",
        "ni_employee",
    ),
    "Class 1 Secondary (employer)": (
        "hmrc/salary_sacrifice_employer_nics_relief",
        "ni_employer",
    ),
}

# Total salary sacrifice contributions (SPP Review 2025: £24bn base)
_SS_CONTRIBUTIONS_BASE_YEAR = 2024


def _read_table(url: str) -> pd.DataFrame:
    try:
        r = requests.get(url, headers=HEADERS, allow_redirects=True, timeout=30)
        r.raise_for_status()
        text = r.content.decode("utf-8-sig")
    except requests.RequestException as e:
        logger.warning(
            "HMRC Tables 6.1 and 6.2 download failed (%s); using %s",
            e,
            FALLBACK_CSV.name,
        )
        text = FALLBACK_CSV.read_text(encoding="utf-8-sig")
    return pd.read_csv(io.StringIO(text), dtype=str)


def _base_year(df: pd.DataFrame) -> int:
    """Map the table's tax year to a PolicyEngine year.

    PolicyEngine UK's year N is tax year N to N+1 (a year-N simulation reads
    the parameter values in force from 6 April N), so "2024 to 2025" is 2024.
    """
    years = df["tax_year"].unique()
    match = re.fullmatch(r"(\d{4}) to (\d{4})", years[0]) if len(years) == 1 else None
    if match is None or int(match[2]) != int(match[1]) + 1:
        raise ValueError(f"Expected one tax year in Tables 6.1/6.2, got {years}")
    return int(match[1])


def _relief(rows: pd.DataFrame, column: str, label: str) -> float:
    """The single positive £m value in ``rows`` whose ``column`` is ``label``."""
    values = rows.loc[rows[column] == label, "value_of_relief"].map(to_float)
    if len(values) != 1 or values.iloc[0] <= 0:
        raise ValueError(
            f"HMRC salary sacrifice relief: expected one positive value for "
            f"{label!r}, got {values.tolist()}"
        )
    return values.iloc[0] * 1e6


def _relief_targets(df: pd.DataFrame, reference_url: str) -> list[Target]:
    base_year = _base_year(df)
    ss = df[
        (df["contribution_type"] == "Salary sacrificed contributions")
        & (df["sector_scheme"] == "Total")
        & (df["scheme_type"] == "Total")
    ]
    income_tax = ss[ss["income_tax_nics"] == "Income Tax"]
    nics = ss[(ss["income_tax_nics"] == "NICs") & (ss["tax_rate"] == "Total")]

    specs = [
        (
            f"hmrc/salary_sacrifice_it_relief_{suffix}",
            "income_tax",
            _relief(income_tax, "tax_rate", rate),
        )
        for rate, suffix in _IT_RATES.items()
    ] + [
        (name, variable, _relief(nics, "nics_relief_class", nics_class))
        for nics_class, (name, variable) in _NICS_CLASSES.items()
    ]
    return [
        Target(
            name=name,
            variable=variable,
            source="hmrc",
            unit=Unit.GBP,
            values={
                y: base * _GROWTH ** (y - base_year)
                for y in range(base_year, _LAST_YEAR + 1)
            },
            reference_url=reference_url,
        )
        for name, variable, base in specs
    ]


def get_targets() -> list[Target]:
    ref = load_config()["hmrc"]["salary_sacrifice_table_6"]
    targets = _relief_targets(_read_table(ref), ref)

    targets.append(
        Target(
            name="hmrc/salary_sacrifice_contributions",
            variable="pension_contributions_via_salary_sacrifice",
            source="hmrc",
            unit=Unit.GBP,
            values={
                y: 24e9 * _GROWTH ** (y - _SS_CONTRIBUTIONS_BASE_YEAR)
                for y in range(_SS_CONTRIBUTIONS_BASE_YEAR, 2030)
            },
            reference_url=(
                "https://assets.publishing.service.gov.uk/media/"
                "67ce0e7c08e764d17a5d3c21/2025_SPP_Review.pdf"
            ),
        )
    )

    return targets
