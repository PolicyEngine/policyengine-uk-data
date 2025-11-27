"""
Service imputations for UK households.

This module coordinates the imputation of various public services including
NHS usage, education spending, and transport subsidies to UK households.
"""

from policyengine_uk.data import UKSingleYearDataset
from .nhs import impute_nhs_usage
from .etb import impute_public_services, create_efrs_input_dataset

# Rail fare index for the survey year (2021, from ETB data)
# This is the cumulative fare index from base year 2020
# Used to derive rail_usage (quantity) from rail_subsidy_spending
# See: policyengine-uk gov.dft.rail.fare_index parameter
RAIL_FARE_INDEX_SURVEY_YEAR = 1.010  # 2021 value (+1.0% from 2020)


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
    # rail_usage = rail_subsidy_spending / fare_index
    # This allows reforms to modify fare_index independently of usage quantity
    dataset.household["rail_usage"] = (
        dataset.household["rail_subsidy_spending"]
        / RAIL_FARE_INDEX_SURVEY_YEAR
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
