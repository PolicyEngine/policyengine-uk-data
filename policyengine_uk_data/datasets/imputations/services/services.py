"""
Service imputations for UK households.

This module coordinates the imputation of various public services including
NHS usage, education spending, and transport subsidies to UK households.
"""

from policyengine_uk.data import UKSingleYearDataset
from policyengine_uk import Microsimulation
from policyengine_uk.system import system
from policyengine_uk_data.datasets.private_releases import CURRENT_ETB_RELEASE
from .nhs import impute_nhs_usage
from .etb import impute_public_services, create_efrs_input_dataset

# ETB survey year used by the current training data.
ETB_SURVEY_YEAR = CURRENT_ETB_RELEASE.default_training_year

RAIL_SUBSIDY_TARGETS = {
    # ORR/GOV.UK rail finance statistics report GBP 21.6bn of government
    # support to the rail industry in 2024-25.
    2025: 21.6e9,
}

BUS_SUBSIDY_TARGETS = {
    # DfT Annual Bus Statistics, year ending March 2025 (England), table
    # BUS05bii: total net government support for local bus services was
    # GBP 3.0bn (of which GBP 0.8bn concessionary travel reimbursement).
    # https://www.gov.uk/government/statistics/annual-bus-statistics-year-ending-march-2025/annual-bus-statistics-year-ending-march-2025
    # England-coverage figure used as the UK anchor: DfT publishes no single
    # GB/UK total and GB/UK would be ~10-20% higher, but this is far better
    # than the unanchored aggregate, which drifts well below the true total.
    2025: 3.0e9,
}


def get_fare_index_survey_year() -> float:
    """
    Get the rail fare index for the ETB survey year.

    Attempts to read from policyengine-uk parameters, falls back to
    hardcoded value if parameter not yet available.
    """
    try:
        return system.parameters.gov.dft.rail.fare_index(ETB_SURVEY_YEAR)
    except AttributeError:
        return 1.0


def calibrate_rail_subsidy_spending(
    dataset: UKSingleYearDataset,
    time_period: int,
) -> float | None:
    target = RAIL_SUBSIDY_TARGETS.get(time_period)
    if target is None:
        return None

    original_time_period = dataset.time_period
    dataset.time_period = str(original_time_period)
    try:
        simulation = Microsimulation(dataset=dataset)
        actual = simulation.calculate(
            "rail_subsidy_spending",
            period=time_period,
            map_to="household",
        ).sum()
    finally:
        dataset.time_period = original_time_period
    if actual <= 0:
        raise ValueError(
            f"Cannot calibrate rail_subsidy_spending: aggregate is {actual}."
        )

    scale = target / actual
    dataset.household["rail_usage"] *= scale
    if "rail_subsidy_spending" in dataset.household:
        dataset.household["rail_subsidy_spending"] *= scale
    return scale


def calibrate_bus_subsidy_spending(
    dataset: UKSingleYearDataset,
    time_period: int,
) -> float | None:
    """Scale bus_subsidy_spending to the DfT net-support total (BUS_SUBSIDY_TARGETS)."""
    target = BUS_SUBSIDY_TARGETS.get(time_period)
    if target is None or "bus_subsidy_spending" not in dataset.household:
        return None

    original_time_period = dataset.time_period
    dataset.time_period = str(original_time_period)
    try:
        simulation = Microsimulation(dataset=dataset)
        actual = simulation.calculate(
            "bus_subsidy_spending",
            period=time_period,
            map_to="household",
        ).sum()
    finally:
        dataset.time_period = original_time_period
    if actual <= 0:
        raise ValueError(
            f"Cannot calibrate bus_subsidy_spending: aggregate is {actual}."
        )

    scale = target / actual
    dataset.household["bus_subsidy_spending"] *= scale
    return scale


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
        dataset.person[person_imputations] = input_data[person_imputations].values

    return dataset
