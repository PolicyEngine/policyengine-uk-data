"""HMRC salary sacrifice income tax and NICs relief targets.

Reads HMRC private pension statistics Tables 6.1 and 6.2 (tidy CSV): income
tax relief on salary-sacrificed pension contributions by marginal rate, and
the Class 1 primary (employee) and secondary (employer) NICs relief on them.

GOV.UK withdraws a release's assets when HMRC publishes the next one: the
July 2025 CSV has returned 410 Gone since the July 2026 release, and the
broad ``except`` this module used to have turned that into calibration
builds that silently lacked these targets. A failed download now falls back
to the copy committed in storage (the same release as ``sources.yaml``), and
a table for another tax year, or without the expected rows, raises instead
of returning fewer targets.

Targets grow 3% a year from the table's year. The NICs relief targets also
follow the Class 1 rates in PolicyEngine's parameters, so the April 2025
rise in the employer rate from 13.8% to 15% raises the employer target in
the years the simulation charges 15%.

Source: https://www.gov.uk/government/statistics/personal-and-stakeholder-pensions-statistics
"""

import io
import logging

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

# Tax year of the Tables 6.1 and 6.2 release in sources.yaml (July 2026,
# 2024-25) and of its committed copy. Refresh the URL, the copy and this
# together: a table for any other year raises.
_TAX_YEAR = 2024
FALLBACK_CSV = (
    STORAGE
    / f"hmrc_pension_relief_tables_6_1_6_2_{_TAX_YEAR}_{(_TAX_YEAR + 1) % 100:02d}.csv"
)

# Uprate 3% pa for wage growth from the base year
_GROWTH = 1.03
_LAST_YEAR = 2031

# Table 6.1 rate rows → target name suffix. HMRC counts Scottish starter
# and intermediate rate relief as basic rate. The Total row is not a target:
# it is the sum of the bands, which HMRC rounds separately (2024-25: bands
# £8.9bn, total £8.8bn), so targeting both would ask for two values.
_IT_RATES = {
    "Basic Rate": "basic_rate",
    "Higher Rate": "higher_rate",
    "Additional Rate": "additional_rate",
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

# Table 6.2 (class, rate row) → the PolicyEngine Class 1 rate it was relieved at
_NICS_RATE_PARAMETERS = {
    ("Class 1 Primary (employee)", "Main Rate"): "employee.main",
    ("Class 1 Primary (employee)", "Additional Rate"): "employee.additional",
    ("Class 1 Secondary (employer)", "Main Rate"): "employer",
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


def _check_tax_year(df: pd.DataFrame) -> int:
    """The table's PolicyEngine year, which must be ``_TAX_YEAR``.

    PolicyEngine UK's year N is tax year N to N+1 (a year-N simulation reads
    the parameter values in force from 6 April N), so "2024 to 2025" is 2024.
    """
    years = list(df["tax_year"].unique())
    expected = f"{_TAX_YEAR} to {_TAX_YEAR + 1}"
    if years != [expected]:
        raise ValueError(
            f"HMRC Tables 6.1/6.2 cover {years}, but {FALLBACK_CSV.name} and "
            f"_TAX_YEAR are {expected!r}: refresh the sources.yaml URL, the "
            "committed copy and _TAX_YEAR together."
        )
    return _TAX_YEAR


def _nics_rate_factors(rows: pd.DataFrame, nics_class: str, base_year: int) -> dict:
    """Class 1 rate in each year relative to the table's year, by year.

    Weighted by HMRC's split of the class's relief between the main and
    additional rates (rows suppressed as [z] weigh nothing).
    """
    from policyengine_uk import CountryTaxBenefitSystem

    parameters = CountryTaxBenefitSystem().parameters
    weights = {}
    for rate_row in ("Main Rate", "Additional Rate"):
        values = rows.loc[
            (rows["nics_relief_class"] == nics_class) & (rows["tax_rate"] == rate_row),
            "value_of_relief",
        ].map(to_float)
        if values.sum() > 0:
            path = _NICS_RATE_PARAMETERS[(nics_class, rate_row)]
            weights[path] = values.sum()
    if not weights:
        raise ValueError(f"HMRC Table 6.2: no rate split for {nics_class!r}")

    def rate(path, year):
        return parameters.get_child(
            f"gov.hmrc.national_insurance.class_1.rates.{path}"
        )(str(year))

    return {
        year: sum(w * rate(p, year) / rate(p, base_year) for p, w in weights.items())
        / sum(weights.values())
        for year in range(base_year, _LAST_YEAR + 1)
    }


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
    base_year = _check_tax_year(df)
    ss = df[
        (df["contribution_type"] == "Salary sacrificed contributions")
        & (df["sector_scheme"] == "Total")
        & (df["scheme_type"] == "Total")
    ]
    income_tax = ss[ss["income_tax_nics"] == "Income Tax"]
    nics = ss[ss["income_tax_nics"] == "NICs"]
    nics_total = nics[nics["tax_rate"] == "Total"]
    unchanged = {y: 1.0 for y in range(base_year, _LAST_YEAR + 1)}

    specs = [
        (
            f"hmrc/salary_sacrifice_it_relief_{suffix}",
            "income_tax",
            _relief(income_tax, "tax_rate", rate),
            unchanged,
        )
        for rate, suffix in _IT_RATES.items()
    ] + [
        (
            name,
            variable,
            _relief(nics_total, "nics_relief_class", nics_class),
            _nics_rate_factors(nics, nics_class, base_year),
        )
        for nics_class, (name, variable) in _NICS_CLASSES.items()
    ]
    return [
        Target(
            name=name,
            variable=variable,
            source="hmrc",
            unit=Unit.GBP,
            values={
                y: base * _GROWTH ** (y - base_year) * factor
                for y, factor in factors.items()
            },
            reference_url=reference_url,
        )
        for name, variable, base, factors in specs
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
