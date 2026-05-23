"""Dataset-side disability benefit category mapping.

PolicyEngine UK models PIP, DLA, and Attendance Allowance from category
inputs. The FRS observes reported amounts, so the data pipeline keeps those
amounts as internal build intermediates and converts them to model inputs
before datasets are published.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd
from policyengine_uk import CountryTaxBenefitSystem
from policyengine_uk.data import UKSingleYearDataset
from policyengine_uk.model_api import WEEKS_IN_YEAR as MODEL_WEEKS_IN_YEAR


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

PIP_POINT_COLUMNS = (
    "pip_m_points",
    "pip_dl_points",
)

SAFETY_MARGIN = 0.1
SURVEY_REPORTED_AMOUNT_WEEKS_IN_YEAR = 365.25 / 7
PIP_STANDARD_POINTS = 8
PIP_ENHANCED_POINTS = 12


@lru_cache(maxsize=None)
def _dwp_category_threshold_parameters(year: int):
    # Match the category formulas removed from policyengine-uk. Those formulas
    # thresholded reported amounts against the baseline DWP rates.
    return CountryTaxBenefitSystem().parameters(year).baseline.gov.dwp


@lru_cache(maxsize=None)
def _dwp_flag_parameters(year: int):
    # Match the FRS disability flag derivation that already lived in uk-data.
    return CountryTaxBenefitSystem().parameters(year).gov.dwp


@lru_cache(maxsize=None)
def _model_supports_pip_point_inputs() -> bool:
    variables = CountryTaxBenefitSystem().variables
    return all(column in variables for column in PIP_POINT_COLUMNS)


def _reported_amount(person: pd.DataFrame, column: str) -> pd.Series:
    if column not in person.columns:
        return pd.Series(0.0, index=person.index)
    return pd.to_numeric(person[column], errors="coerce").fillna(0.0)


def _category_from_reported_amount(
    reported_amount: pd.Series,
    thresholds: tuple[tuple[str, float], ...],
) -> np.ndarray:
    weekly_amount = pd.to_numeric(reported_amount, errors="coerce").fillna(0)
    weekly_amount = weekly_amount.to_numpy(dtype=float) / MODEL_WEEKS_IN_YEAR
    category = np.full(len(weekly_amount), "NONE", dtype=object)
    for category_name, weekly_rate in thresholds:
        category[weekly_amount >= float(weekly_rate) * (1 - SAFETY_MARGIN)] = (
            category_name
        )
    return category


def _minimum_pip_points_from_category(category: pd.Series) -> np.ndarray:
    category = category.fillna("NONE").astype(str)
    return np.select(
        [
            category == "ENHANCED",
            category == "STANDARD",
        ],
        [
            PIP_ENHANCED_POINTS,
            PIP_STANDARD_POINTS,
        ],
        default=0,
    ).astype(int)


def add_pip_points_from_categories(
    person: pd.DataFrame,
    *,
    inplace: bool = False,
) -> pd.DataFrame:
    """Assign the minimum PIP points consistent with observed categories."""

    if not inplace:
        person = person.copy()

    mappings = (
        ("pip_m_category", "pip_m_points"),
        ("pip_dl_category", "pip_dl_points"),
    )
    for category_column, points_column in mappings:
        if category_column in person.columns:
            person[points_column] = _minimum_pip_points_from_category(
                person[category_column]
            )

    return person


def add_disability_benefit_categories_from_reported_amounts(
    person: pd.DataFrame,
    year: int,
    *,
    inplace: bool = False,
) -> pd.DataFrame:
    """Convert reported disability benefit amounts into category inputs."""

    if not inplace:
        person = person.copy()

    dwp = _dwp_category_threshold_parameters(int(year))
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

    if _model_supports_pip_point_inputs():
        add_pip_points_from_categories(person, inplace=True)

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

    dwp = _dwp_flag_parameters(int(year))
    dla_sc = _reported_amount(person, "dla_sc_reported")
    dla_m = _reported_amount(person, "dla_m_reported")
    pip_m = _reported_amount(person, "pip_m_reported")
    pip_dl = _reported_amount(person, "pip_dl_reported")
    afcs = _reported_amount(person, "afcs_reported")

    person["is_disabled_for_benefits"] = (dla_sc + dla_m + pip_m + pip_dl) > 0

    threshold_safety_gap = 1 * SURVEY_REPORTED_AMOUNT_WEEKS_IN_YEAR
    dla_sc_higher = (
        dwp.dla.self_care.higher * SURVEY_REPORTED_AMOUNT_WEEKS_IN_YEAR
        - threshold_safety_gap
    )
    pip_dl_enhanced = (
        dwp.pip.daily_living.enhanced * SURVEY_REPORTED_AMOUNT_WEEKS_IN_YEAR
        - threshold_safety_gap
    )

    person["is_enhanced_disabled_for_benefits"] = dla_sc > dla_sc_higher
    person["is_severely_disabled_for_benefits"] = (
        (dla_sc >= dla_sc_higher) | (pip_dl >= pip_dl_enhanced) | (afcs > 0)
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
