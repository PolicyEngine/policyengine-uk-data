from __future__ import annotations

import pandas as pd
from policyengine_uk import CountryTaxBenefitSystem
from policyengine_uk.data import UKSingleYearDataset
from policyengine_uk.model_api import WEEKS_IN_YEAR

from policyengine_uk_data.datasets.disability_benefits import (
    add_disability_benefit_categories_from_reported_amounts,
    add_disability_benefit_flags_from_reported_amounts,
    drop_internal_disability_reported_amounts,
    strip_internal_disability_reported_amounts,
)


def test_reported_amounts_map_to_disability_categories():
    year = 2025
    dwp = CountryTaxBenefitSystem().parameters(year).baseline.gov.dwp
    person = pd.DataFrame(
        {
            "attendance_allowance_reported": [
                0,
                dwp.attendance_allowance.lower * WEEKS_IN_YEAR * 0.91,
                dwp.attendance_allowance.higher * WEEKS_IN_YEAR * 0.91,
            ],
            "dla_sc_reported": [
                0,
                dwp.dla.self_care.lower * WEEKS_IN_YEAR * 0.91,
                dwp.dla.self_care.middle * WEEKS_IN_YEAR * 0.91,
            ],
            "dla_m_reported": [
                0,
                dwp.dla.mobility.lower * WEEKS_IN_YEAR * 0.91,
                dwp.dla.mobility.higher * WEEKS_IN_YEAR * 0.91,
            ],
            "pip_m_reported": [
                0,
                dwp.pip.mobility.standard * WEEKS_IN_YEAR * 0.91,
                dwp.pip.mobility.enhanced * WEEKS_IN_YEAR * 0.91,
            ],
            "pip_dl_reported": [
                0,
                dwp.pip.daily_living.standard * WEEKS_IN_YEAR * 0.91,
                dwp.pip.daily_living.enhanced * WEEKS_IN_YEAR * 0.91,
            ],
        }
    )

    result = add_disability_benefit_categories_from_reported_amounts(person, year)

    assert result["aa_category"].tolist() == ["NONE", "LOWER", "HIGHER"]
    assert result["dla_sc_category"].tolist() == ["NONE", "LOWER", "MIDDLE"]
    assert result["dla_m_category"].tolist() == ["NONE", "LOWER", "HIGHER"]
    assert result["pip_m_category"].tolist() == ["NONE", "STANDARD", "ENHANCED"]
    assert result["pip_dl_category"].tolist() == ["NONE", "STANDARD", "ENHANCED"]


def test_reported_amounts_recompute_disability_flags():
    year = 2025
    dwp = CountryTaxBenefitSystem().parameters(year).gov.dwp
    person = pd.DataFrame(
        {
            "dla_sc_reported": [
                0.0,
                dwp.dla.self_care.higher * (365.25 / 7),
                0.0,
            ],
            "dla_m_reported": [0.0, 0.0, 0.0],
            "pip_m_reported": [0.0, 0.0, 0.0],
            "pip_dl_reported": [
                0.0,
                0.0,
                dwp.pip.daily_living.enhanced * (365.25 / 7),
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
        False,
    ]
    assert result["is_severely_disabled_for_benefits"].tolist() == [
        False,
        True,
        True,
    ]


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
