"""
Service imputations for UK households.

This module coordinates the imputation of various public services including
NHS usage, education spending, and transport subsidies to UK households.
"""

from policyengine_uk.data import UKSingleYearDataset
from policyengine_uk.system import system
from .nhs import impute_nhs_usage
from .etb import impute_public_services, create_efrs_input_dataset

# ETB survey year (most recent year in ETB data)
ETB_SURVEY_YEAR = 2021

# Fallback fare index for 2021 if parameter not yet available in policyengine-uk
# This is the cumulative fare index from base year 2020 (+1.0% from 2020)
FALLBACK_FARE_INDEX_2021 = 1.010


def get_fare_index_survey_year() -> float:
    """
    Get the rail fare index for the ETB survey year.

    Attempts to read from policyengine-uk parameters, falls back to
    hardcoded value if parameter not yet available.
    """
    try:
        return system.parameters.gov.dft.rail.fare_index(ETB_SURVEY_YEAR)
    except AttributeError:
        # Parameter not yet available in policyengine-uk
        return FALLBACK_FARE_INDEX_2021


def impute_services(
    dataset: UKSingleYearDataset,
) -> UKSingleYearDataset:
    """
    Impute public service usage and spending for households.

    This function combines NHS usage imputations with other public services
    (education, rail, and bus subsidies) to create a comprehensive dataset
    of household public service consumption.

    Args:
        dataset: A PolicyEngine UK dataset containing household and person data.

    Returns:
        Updated dataset with imputed service usage and spending variables.
    """
    dataset = dataset.copy()
    input_data = create_efrs_input_dataset(dataset)

    input_data = impute_nhs_usage(input_data)
    input_data = impute_public_services(input_data)

    for household_imputations in [
        "dfe_education_spending",
        "rail_subsidy_spending",
        "bus_subsidy_spending",
    ]:
        dataset.household[household_imputations] = (
            input_data[household_imputations]
            .groupby(input_data.household_id)
            .sum()
            .values
        )

    # Derive rail_usage (quantity at base year prices) from rail_subsidy_spending
    # rail_usage = rail_subsidy_spending / fare_index at survey year
    # This allows reforms to modify fare_index independently of usage quantity
    fare_index_survey_year = get_fare_index_survey_year()
    dataset.household["rail_usage"] = (
        dataset.household["rail_subsidy_spending"] / fare_index_survey_year
    )

    visit_variables = [
        "a_and_e_visits",
        "admitted_patient_visits",
        "outpatient_visits",
    ]
    spending_variables = [
        "nhs_a_and_e_spending",
        "nhs_admitted_patient_spending",
        "nhs_outpatient_spending",
    ]

    for person_imputations in visit_variables + spending_variables:
        dataset.person[person_imputations] = input_data[
            person_imputations
        ].values

    return dataset
