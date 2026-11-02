from policyengine_uk.data import UKSingleYearDataset
import pandas as pd

# Benefit-unit roles taken from the survey, which policyengine-uk otherwise
# infers from ages. A stacked table without them would leave missing roles.
SURVEY_ROLE_COLUMNS = ("is_claimant_or_partner",)


def stack_datasets(
    data_1: UKSingleYearDataset, data_2: UKSingleYearDataset
) -> UKSingleYearDataset:
    for column in SURVEY_ROLE_COLUMNS:
        if (column in data_1.person.columns) != (column in data_2.person.columns):
            raise ValueError(f"Only one of the stacked datasets has {column}.")
    # Copies share their survey household's provenance even after ID remapping.
    for dataset in (data_1, data_2):
        if "source_household_id" not in dataset.household.columns:
            dataset.household["source_household_id"] = dataset.household["household_id"]
    person_id_offset = data_1.person.person_id.max() + 1
    benunit_id_offset = data_1.benunit.benunit_id.max() + 1
    household_id_offset = data_1.household.household_id.max() + 1
    data_2.person.person_id += person_id_offset
    data_2.person.person_benunit_id += benunit_id_offset
    data_2.person.person_household_id += household_id_offset
    data_2.benunit.benunit_id += benunit_id_offset
    data_2.household.household_id += household_id_offset

    return UKSingleYearDataset(
        person=pd.concat([data_1.person, data_2.person], ignore_index=True),
        benunit=pd.concat([data_1.benunit, data_2.benunit], ignore_index=True),
        household=pd.concat([data_1.household, data_2.household], ignore_index=True),
        fiscal_year=data_1.time_period,
    )
