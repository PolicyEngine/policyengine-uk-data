"""NHS usage bands are whole years of age, and infants carry an age in months.

frs.impute_infant_age_in_months gives children recorded as aged 0 an age such
as 0.375. The NHS imputation must band them as 0-year-olds: before it banded on
the raw age, so the "0 years" band counted nobody and its per-person spending
divided by zero.
"""

import numpy as np
import pandas as pd

from policyengine_uk_data.datasets.imputations.services.nhs import (
    create_nhs_usage_data,
    impute_nhs_usage,
)

SPENDING = [
    "nhs_a_and_e_spending",
    "nhs_admitted_patient_spending",
    "nhs_outpatient_spending",
]


def people(infant_ages):
    """Every NHS band populated: the infants given, then each age 1 to 100."""
    ages = list(infant_ages) + list(range(1, 101))
    n = len(ages)
    return pd.DataFrame(
        {
            "age": np.repeat(np.array(ages, dtype=float), 2),
            "gender": ["MALE", "FEMALE"] * n,
            "household_weight": np.full(2 * n, 1_000.0),
        }
    )


def test_months_and_whole_years_give_the_same_infant_averages():
    in_months = create_nhs_usage_data(people([0.375, 0.875]))
    whole_years = create_nhs_usage_data(people([0, 0]))
    numeric = in_months.select_dtypes("number")
    assert np.isfinite(numeric.values).all()
    pd.testing.assert_frame_equal(in_months, whole_years)


def test_infants_in_months_get_finite_spending():
    efrs = impute_nhs_usage(people([0.0417, 0.375, 0.875, 0.9583]))
    infants = efrs[efrs.age < 1]
    assert np.isfinite(infants[SPENDING].values).all()
    assert (infants[SPENDING].sum(axis=1) > 0).all()
    # An infant in months gets the same as a whole-year 0-year-old.
    reference = impute_nhs_usage(people([0, 0, 0, 0]))
    assert np.allclose(
        infants[SPENDING].values, reference[reference.age < 1][SPENDING].values
    )
