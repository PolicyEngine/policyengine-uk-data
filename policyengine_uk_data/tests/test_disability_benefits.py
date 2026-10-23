from __future__ import annotations

import pandas as pd
from policyengine_uk import CountryTaxBenefitSystem
from policyengine_uk.data import UKSingleYearDataset

from policyengine_uk_data.datasets.disability_benefits import (
    SURVEY_REPORTED_AMOUNT_WEEKS_IN_YEAR,
    add_disability_benefit_categories_from_reported_amounts,
    add_disability_benefit_flags_from_reported_amounts,
    drop_internal_disability_reported_amounts,
    strip_internal_disability_reported_amounts,
)


def test_reported_amounts_map_to_disability_categories():
    year = 2025
    dwp = CountryTaxBenefitSystem().parameters(year).gov.dwp

    def annual_near_threshold(weekly_rate):
        return (weekly_rate - 0.5) * SURVEY_REPORTED_AMOUNT_WEEKS_IN_YEAR

    person = pd.DataFrame(
        {
            "attendance_allowance_reported": [
                0,
                annual_near_threshold(dwp.attendance_allowance.lower),
                annual_near_threshold(dwp.attendance_allowance.higher),
            ],
            "dla_sc_reported": [
                0,
                annual_near_threshold(dwp.dla.self_care.lower),
                annual_near_threshold(dwp.dla.self_care.middle),
            ],
            "dla_m_reported": [
                0,
                annual_near_threshold(dwp.dla.mobility.lower),
                annual_near_threshold(dwp.dla.mobility.higher),
            ],
            "pip_m_reported": [
                0,
                annual_near_threshold(dwp.pip.mobility.standard),
                annual_near_threshold(dwp.pip.mobility.enhanced),
            ],
            "pip_dl_reported": [
                0,
                annual_near_threshold(dwp.pip.daily_living.standard),
                annual_near_threshold(dwp.pip.daily_living.enhanced),
            ],
        }
    )

    result = add_disability_benefit_categories_from_reported_amounts(person, year)

    assert result["aa_category"].tolist() == ["NONE", "LOWER", "HIGHER"]
    assert result["dla_sc_category"].tolist() == ["NONE", "LOWER", "MIDDLE"]
    assert result["dla_m_category"].tolist() == ["NONE", "LOWER", "HIGHER"]
    assert result["pip_m_category"].tolist() == ["NONE", "STANDARD", "ENHANCED"]
    assert result["pip_dl_category"].tolist() == ["NONE", "STANDARD", "ENHANCED"]


def test_reported_amounts_do_not_use_percentage_category_margin():
    year = 2025
    dwp = CountryTaxBenefitSystem().parameters(year).gov.dwp
    person = pd.DataFrame(
        {
            "attendance_allowance_reported": [
                dwp.attendance_allowance.higher
                * SURVEY_REPORTED_AMOUNT_WEEKS_IN_YEAR
                * 0.91,
            ],
            "pip_dl_reported": [
                dwp.pip.daily_living.standard
                * SURVEY_REPORTED_AMOUNT_WEEKS_IN_YEAR
                * 0.91,
            ],
            "pip_m_reported": [
                dwp.pip.mobility.enhanced * SURVEY_REPORTED_AMOUNT_WEEKS_IN_YEAR * 0.91,
            ],
        }
    )

    result = add_disability_benefit_categories_from_reported_amounts(person, year)

    assert result["aa_category"].tolist() == ["LOWER"]
    assert result["pip_dl_category"].tolist() == ["NONE"]
    assert result["pip_m_category"].tolist() == ["STANDARD"]


def test_reported_amounts_recompute_disability_flags():
    year = 2025
    dwp = CountryTaxBenefitSystem().parameters(year).gov.dwp
    person = pd.DataFrame(
        {
            "attendance_allowance_reported": [0.0, 0.0, 0.0],
            "dla_sc_reported": [
                0.0,
                dwp.dla.self_care.higher * SURVEY_REPORTED_AMOUNT_WEEKS_IN_YEAR,
                0.0,
            ],
            "dla_m_reported": [0.0, 0.0, 0.0],
            "pip_m_reported": [0.0, 0.0, 0.0],
            "pip_dl_reported": [
                0.0,
                0.0,
                dwp.pip.daily_living.enhanced * SURVEY_REPORTED_AMOUNT_WEEKS_IN_YEAR,
            ],
            "afcs_reported": [0.0, 0.0, 0.0],
            "is_disabled_for_benefits": [True, False, False],
            "is_enhanced_disabled_for_benefits": [True, False, False],
            "is_severely_disabled_for_benefits": [True, False, False],
        }
    )

    result = add_disability_benefit_flags_from_reported_amounts(person, year)

    assert result["is_disabled_for_benefits"].tolist() == [False, True, True]
    assert result["is_enhanced_disabled_for_benefits"].tolist() == [
        False,
        True,
        True,
    ]
    assert result["is_severely_disabled_for_benefits"].tolist() == [
        False,
        True,
        True,
    ]


def test_reported_amounts_widen_base_disability_flag():
    year = 2025
    person = pd.DataFrame(
        {
            "attendance_allowance_reported": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "sda_reported": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "incapacity_benefit_reported": [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            "iidb_reported": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            "afcs_reported": [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            "esa_contrib_reported": [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            "esa_income_reported": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        }
    )

    result = add_disability_benefit_flags_from_reported_amounts(person, year)

    assert result["is_disabled_for_benefits"].all()


def test_attendance_allowance_feeds_stronger_disability_flags():
    year = 2025
    dwp = CountryTaxBenefitSystem().parameters(year).gov.dwp
    weeks = SURVEY_REPORTED_AMOUNT_WEEKS_IN_YEAR
    person = pd.DataFrame(
        {
            "attendance_allowance_reported": [
                dwp.attendance_allowance.lower * weeks,
                dwp.attendance_allowance.higher * weeks,
            ],
        }
    )

    result = add_disability_benefit_flags_from_reported_amounts(person, year)

    assert result["is_disabled_for_benefits"].tolist() == [True, True]
    assert result["is_enhanced_disabled_for_benefits"].tolist() == [False, True]
    assert result["is_severely_disabled_for_benefits"].tolist() == [True, True]


def test_categories_and_flags_share_the_survey_fiscal_year_rates():
    # FRS 2024-25 respondents were paid the rates in force from April 2024,
    # which the fiscal-converted `gov` tree carries at 2024. Reading the
    # `baseline` clone at 2024 thresholded categories against the 2023-24
    # table, a year stale relative to the flags (uk-data#475).
    year = 2024
    system = CountryTaxBenefitSystem()
    current = system.parameters(year).gov.dwp.attendance_allowance.higher
    previous = system.parameters(year - 1).gov.dwp.attendance_allowance.higher
    assert previous < current

    person = pd.DataFrame(
        {
            "attendance_allowance_reported": [
                current * SURVEY_REPORTED_AMOUNT_WEEKS_IN_YEAR,
                previous * SURVEY_REPORTED_AMOUNT_WEEKS_IN_YEAR,
            ],
        }
    )

    categories = add_disability_benefit_categories_from_reported_amounts(person, year)
    flags = add_disability_benefit_flags_from_reported_amounts(person, year)

    assert categories["aa_category"].tolist() == ["HIGHER", "LOWER"]
    assert flags["is_enhanced_disabled_for_benefits"].tolist() == [True, False]


def test_category_tolerance_is_one_pound_a_week_in_survey_weeks():
    # frs.py annualises weekly amounts with 365.25/7; converting back with
    # the model's 52 stretched the GBP 1/week tolerance to GBP 1.37
    # (uk-data#476).
    from policyengine_uk_data.datasets.frs import WEEKS_IN_YEAR as FRS_WEEKS_IN_YEAR

    assert SURVEY_REPORTED_AMOUNT_WEEKS_IN_YEAR == FRS_WEEKS_IN_YEAR

    year = 2024
    higher = (
        CountryTaxBenefitSystem().parameters(year).gov.dwp.attendance_allowance.higher
    )
    person = pd.DataFrame(
        {
            "attendance_allowance_reported": [
                (higher - 0.99) * FRS_WEEKS_IN_YEAR,
                (higher - 1.01) * FRS_WEEKS_IN_YEAR,
            ],
        }
    )

    result = add_disability_benefit_categories_from_reported_amounts(person, year)

    assert result["aa_category"].tolist() == ["HIGHER", "LOWER"]


def test_drop_internal_disability_reported_amounts_keeps_categories():
    person = pd.DataFrame(
        {
            "person_id": [1],
            "pip_dl_reported": [1_000.0],
            "pip_dl_category": ["STANDARD"],
        }
    )

    result = drop_internal_disability_reported_amounts(person)

    assert "pip_dl_reported" not in result.columns
    assert result["pip_dl_category"].tolist() == ["STANDARD"]
    assert "pip_dl_reported" in person.columns


def test_strip_internal_disability_reported_amounts_cleans_dataset_person_frame():
    dataset = UKSingleYearDataset(
        person=pd.DataFrame(
            {
                "person_id": [0],
                "person_benunit_id": [0],
                "person_household_id": [0],
                "pip_dl_reported": [1_000.0],
                "pip_dl_category": ["STANDARD"],
            }
        ),
        benunit=pd.DataFrame({"benunit_id": [0]}),
        household=pd.DataFrame({"household_id": [0]}),
        fiscal_year=2025,
    )

    result = strip_internal_disability_reported_amounts(dataset)

    assert "pip_dl_reported" not in result.person.columns
    assert "pip_dl_reported" in dataset.person.columns
