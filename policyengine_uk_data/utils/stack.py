from policyengine_uk.data import UKSingleYearDataset
import pandas as pd


def stack_datasets(
    data_1: UKSingleYearDataset, data_2: UKSingleYearDataset
) -> UKSingleYearDataset:
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
        household=pd.concat(
            [data_1.household, data_2.household], ignore_index=True
        ),
        fiscal_year=data_1.time_period,
    )
