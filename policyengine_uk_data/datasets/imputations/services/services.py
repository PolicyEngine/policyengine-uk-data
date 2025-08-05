from policyengine_uk.data import UKSingleYearDataset
from .nhs import impute_nhs_usage
from .etb import impute_public_services, create_efrs_input_dataset


def impute_services(
    dataset: UKSingleYearDataset,
) -> UKSingleYearDataset:
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
