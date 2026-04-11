import inspect

import numpy as np
import pandas as pd

from policyengine_uk_data.datasets.frs import (
    add_legacy_benefit_proxies,
    create_frs,
    derive_esa_health_condition_proxy,
    derive_esa_support_group_proxy,
    derive_legacy_jobseeker_proxy,
)


def test_legacy_jobseeker_proxy_tracks_unemployed_working_age_non_workers():
    result = derive_legacy_jobseeker_proxy(
        age=np.array([18, 30, 66, 17, 25, 25, 66]),
        employment_status=np.array(
            [
                "UNEMPLOYED",
                "UNEMPLOYED",
                "UNEMPLOYED",
                "UNEMPLOYED",
                "STUDENT",
                "CARER",
                "UNEMPLOYED",
            ]
        ),
        hours_worked=np.array([0, 12, 0, 0, 0, 0, 0]),
        state_pension_age=np.array([66, 66, 66, 66, 66, 66, 67]),
    )

    assert result.tolist() == [True, False, False, False, False, False, True]


def test_esa_health_condition_proxy_uses_disabled_employment_states():
    result = derive_esa_health_condition_proxy(
        age=np.array([16, 45, 45, 66]),
        employment_status=np.array(
            [
                "LONG_TERM_DISABLED",
                "SHORT_TERM_DISABLED",
                "FT_EMPLOYED",
                "LONG_TERM_DISABLED",
            ]
        ),
        state_pension_age=np.array([66, 66, 66, 66]),
    )

    assert result.tolist() == [True, True, False, False]


def test_esa_support_group_proxy_is_stricter_subset_of_health_proxy():
    health_proxy = np.array([True, True, True, False])
    result = derive_esa_support_group_proxy(
        age=np.array([16, 45, 45, 66]),
        employment_status=np.array(
            [
                "LONG_TERM_DISABLED",
                "SHORT_TERM_DISABLED",
                "LONG_TERM_DISABLED",
                "FT_EMPLOYED",
            ]
        ),
        hours_worked=np.array([0, 0, 12, 0]),
        esa_health_condition_proxy=health_proxy,
        state_pension_age=np.array([66, 66, 66, 66]),
    )

    assert result.tolist() == [True, False, False, False]


def test_add_legacy_benefit_proxies_wires_all_three_columns():
    pe_person = pd.DataFrame(
        {
            "age": [18, 45, 45, 66],
            "employment_status": [
                "UNEMPLOYED",
                "LONG_TERM_DISABLED",
                "SHORT_TERM_DISABLED",
                "LONG_TERM_DISABLED",
            ],
            "hours_worked": [0, 0, 12, 0],
            "is_disabled_for_benefits": [False, True, False, True],
            "is_severely_disabled_for_benefits": [False, False, True, True],
            "esa_income_reported": [0.0, 0.0, 100.0, 0.0],
            "esa_contrib_reported": [0.0, 0.0, 0.0, 0.0],
            "incapacity_benefit_reported": [0.0, 0.0, 0.0, 0.0],
            "sda_reported": [0.0, 0.0, 0.0, 0.0],
        }
    )

    result = add_legacy_benefit_proxies(
        pe_person.copy(), state_pension_age=np.array([66, 66, 66, 66])
    )

    assert result["legacy_jobseeker_proxy"].tolist() == [True, False, False, False]
    assert result["esa_health_condition_proxy"].tolist() == [False, True, True, False]
    assert result["esa_support_group_proxy"].tolist() == [False, True, False, False]


def test_create_frs_calls_add_legacy_benefit_proxies():
    source = inspect.getsource(create_frs)

    assert (
        'state_pension_age = sim.calculate("state_pension_age", year).values' in source
    )
    assert "add_legacy_benefit_proxies(pe_person, state_pension_age)" in source
