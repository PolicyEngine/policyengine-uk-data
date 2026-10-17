"""Dataset-side disability benefit category mapping.

PolicyEngine UK models PIP, DLA, and Attendance Allowance from category
inputs. The FRS observes reported amounts, so the data pipeline keeps those
amounts as internal build intermediates and converts them to model inputs
before datasets are published.

Conventions shared by the category and flag derivations:

- ``year`` is the survey year (fiscal year ``year``/``year + 1``, the
  dataset's ``time_period``). Reported amounts are thresholded against the
  DWP rates in force during that fiscal year, read from the fiscal-converted
  ``gov`` parameter tree.
- Reported amounts are weekly survey responses annualised in ``frs.py`` with
  ``365.25 / 7``; both derivations convert back with the same factor.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd
from policyengine_uk import CountryTaxBenefitSystem
from policyengine_uk.data import UKSingleYearDataset


DISABILITY_REPORTED_AMOUNT_COLUMNS = (
    "attendance_allowance_reported",
    "dla_sc_reported",
    "dla_m_reported",
    "pip_m_reported",
    "pip_dl_reported",
)

DISABILITY_CATEGORY_COLUMNS = (
    "aa_category",
    "dla_sc_category",
    "dla_m_category",
    "pip_m_category",
    "pip_dl_category",
)

BASE_DISABILITY_FLAG_REPORTED_AMOUNT_COLUMNS = (
    "attendance_allowance_reported",
    "dla_sc_reported",
    "dla_m_reported",
    "pip_m_reported",
    "pip_dl_reported",
    "sda_reported",
    "incapacity_benefit_reported",
    "iidb_reported",
    "afcs_reported",
    "esa_contrib_reported",
    "esa_income_reported",
)

CATEGORY_THRESHOLD_WEEKLY_TOLERANCE = 1.0
# The factor `frs.py` annualises weekly FRS amounts with. Converting back
# with the model's 52-week constant would inflate weekly amounts by 0.34% and
# make the GBP 1/week tolerance below mean GBP 1.37 (uk-data#476).
SURVEY_REPORTED_AMOUNT_WEEKS_IN_YEAR = 365.25 / 7


@lru_cache(maxsize=None)
def _dwp_rate_parameters(year: int):
    """DWP weekly rates in force during the survey's fiscal year.

    policyengine-uk rewrites ``parameters.gov`` onto fiscal years at load, so
    ``gov`` at ``year`` carries the rates paid from April of that year. The
    ``baseline`` clone is taken before that rewrite and stays on calendar
    instants, so ``baseline`` at ``year`` is the previous fiscal year's
    table; categories and flags must read the same tree (uk-data#475).
    """
    return CountryTaxBenefitSystem().parameters(year).gov.dwp


def _reported_amount(person: pd.DataFrame, column: str) -> pd.Series:
    if column not in person.columns:
        return pd.Series(0.0, index=person.index)
    return pd.to_numeric(person[column], errors="coerce").fillna(0.0)


def _reported_amount_sum(
    person: pd.DataFrame,
    columns: tuple[str, ...],
) -> pd.Series:
    total = pd.Series(0.0, index=person.index)
    for column in columns:
        total += _reported_amount(person, column)
    return total


def _category_from_reported_amount(
    reported_amount: pd.Series,
    thresholds: tuple[tuple[str, float], ...],
) -> np.ndarray:
    weekly_amount = pd.to_numeric(reported_amount, errors="coerce").fillna(0)
    weekly_amount = (
        weekly_amount.to_numpy(dtype=float) / SURVEY_REPORTED_AMOUNT_WEEKS_IN_YEAR
    )
    category = np.full(len(weekly_amount), "NONE", dtype=object)
    for category_name, weekly_rate in thresholds:
        # FRS benefit amounts are weekly survey responses annualised upstream;
        # allow GBP 1/week of rounding noise without discounting rates by a
        # percentage that can promote people into higher award categories.
        threshold = max(
            0.0,
            float(weekly_rate) - CATEGORY_THRESHOLD_WEEKLY_TOLERANCE,
        )
        category[weekly_amount >= threshold] = category_name
    return category


def _reaches_weekly_rate(reported_amount: pd.Series, weekly_rate: float) -> pd.Series:
    """Whether reported amounts reach a weekly rate under the category rule.

    Uses the category derivation itself, so a flag built from this agrees with
    the categories by construction, including at the tolerance boundary.
    """
    category = _category_from_reported_amount(
        reported_amount, (("REACHED", weekly_rate),)
    )
    return pd.Series(category == "REACHED", index=reported_amount.index)


def add_disability_benefit_categories_from_reported_amounts(
    person: pd.DataFrame,
    year: int,
    *,
    inplace: bool = False,
) -> pd.DataFrame:
    """Convert reported disability benefit amounts into category inputs."""

    if not inplace:
        person = person.copy()

    dwp = _dwp_rate_parameters(int(year))
    mappings = (
        (
            "attendance_allowance_reported",
            "aa_category",
            (
                ("LOWER", dwp.attendance_allowance.lower),
                ("HIGHER", dwp.attendance_allowance.higher),
            ),
        ),
        (
            "dla_sc_reported",
            "dla_sc_category",
            (
                ("LOWER", dwp.dla.self_care.lower),
                ("MIDDLE", dwp.dla.self_care.middle),
                ("HIGHER", dwp.dla.self_care.higher),
            ),
        ),
        (
            "dla_m_reported",
            "dla_m_category",
            (
                ("LOWER", dwp.dla.mobility.lower),
                ("HIGHER", dwp.dla.mobility.higher),
            ),
        ),
        (
            "pip_m_reported",
            "pip_m_category",
            (
                ("STANDARD", dwp.pip.mobility.standard),
                ("ENHANCED", dwp.pip.mobility.enhanced),
            ),
        ),
        (
            "pip_dl_reported",
            "pip_dl_category",
            (
                ("STANDARD", dwp.pip.daily_living.standard),
                ("ENHANCED", dwp.pip.daily_living.enhanced),
            ),
        ),
    )

    for reported_column, category_column, thresholds in mappings:
        if reported_column in person.columns:
            person[category_column] = _category_from_reported_amount(
                person[reported_column],
                thresholds,
            )

    return person


def add_disability_benefit_flags_from_reported_amounts(
    person: pd.DataFrame,
    year: int,
    *,
    inplace: bool = False,
) -> pd.DataFrame:
    """Recompute disability flags derived from reported benefit amounts."""

    if not inplace:
        person = person.copy()

    dwp = _dwp_rate_parameters(int(year))
    attendance_allowance = _reported_amount(person, "attendance_allowance_reported")
    dla_sc = _reported_amount(person, "dla_sc_reported")
    pip_dl = _reported_amount(person, "pip_dl_reported")

    person["is_disabled_for_benefits"] = (
        _reported_amount_sum(person, BASE_DISABILITY_FLAG_REPORTED_AMOUNT_COLUMNS) > 0
    )

    threshold_safety_gap = 1 * SURVEY_REPORTED_AMOUNT_WEEKS_IN_YEAR
    aa_higher = (
        dwp.attendance_allowance.higher * SURVEY_REPORTED_AMOUNT_WEEKS_IN_YEAR
        - threshold_safety_gap
    )
    dla_sc_higher = (
        dwp.dla.self_care.higher * SURVEY_REPORTED_AMOUNT_WEEKS_IN_YEAR
        - threshold_safety_gap
    )
    pip_dl_enhanced = (
        dwp.pip.daily_living.enhanced * SURVEY_REPORTED_AMOUNT_WEEKS_IN_YEAR
        - threshold_safety_gap
    )

    person["is_enhanced_disabled_for_benefits"] = (
        (attendance_allowance >= aa_higher)
        | (dla_sc > dla_sc_higher)
        | (pip_dl >= pip_dl_enhanced)
    )
    # The tax credit severe disability condition (CTC Regs 2002 reg 8(3)-(5);
    # WTC Regs 2002 reg 17(2)-(4)): DLA care at the highest rate, PIP daily
    # living at the enhanced rate, higher-rate Attendance Allowance (a WTC
    # condition; children cannot receive it), or armed forces independence
    # payment. policyengine-uk also keys the Universal Credit higher disabled
    # child addition on this flag, although UC Regs 2013 reg 24(2)(b) differs
    # (it adds blindness and has no AFIP limb). FRS code 8 (`afcs_reported`)
    # covers every Armed Forces Compensation Scheme and war disablement pension
    # payment, including the guaranteed income payment every AFIP recipient
    # also receives; AFIP has no code of its own and cannot be separated out,
    # so AFIP recipients are not flagged (a known under-count). The legacy
    # severe disability premium's wider list (any Attendance Allowance, DLA
    # care at the middle rate, PIP daily living at the standard rate) is read
    # by policyengine-uk from the benefit categories, not from this flag. The
    # flag uses the category rule, so it agrees with the stored categories by
    # construction.
    person["is_severely_disabled_for_benefits"] = (
        _reaches_weekly_rate(attendance_allowance, dwp.attendance_allowance.higher)
        | _reaches_weekly_rate(dla_sc, dwp.dla.self_care.higher)
        | _reaches_weekly_rate(pip_dl, dwp.pip.daily_living.enhanced)
    )

    return person


def drop_internal_disability_reported_amounts(
    person: pd.DataFrame,
    *,
    inplace: bool = False,
) -> pd.DataFrame:
    """Drop disability amount intermediates that are not PE-UK inputs."""

    if inplace:
        person.drop(
            columns=list(DISABILITY_REPORTED_AMOUNT_COLUMNS),
            errors="ignore",
            inplace=True,
        )
        return person
    return person.drop(
        columns=list(DISABILITY_REPORTED_AMOUNT_COLUMNS),
        errors="ignore",
    )


def strip_internal_disability_reported_amounts(
    dataset: UKSingleYearDataset,
) -> UKSingleYearDataset:
    """Return ``dataset`` without internal disability amount intermediates."""

    dataset = dataset.copy()
    dataset.person = drop_internal_disability_reported_amounts(dataset.person)
    return dataset
