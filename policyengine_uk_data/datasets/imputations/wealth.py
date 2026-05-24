"""
Household wealth imputation using Wealth and Assets Survey data.

This module imputes various types of household wealth (property, financial,
corporate) using machine learning models trained on the UK Wealth and Assets
Survey (WAS) data.
"""

import numpy as np
import pandas as pd
from policyengine_uk_data.datasets.private_releases import CURRENT_WAS_RELEASE
from policyengine_uk_data.storage import STORAGE_FOLDER
from policyengine_uk.data import UKSingleYearDataset
from policyengine_uk import Microsimulation
from policyengine_uk_data.utils.qrf import QRF

WAS_TAB_FOLDER = STORAGE_FOLDER / CURRENT_WAS_RELEASE.name
WEALTH_MODEL_FILENAME = f"wealth_{CURRENT_WAS_RELEASE.name}.pkl"

REGIONS = {
    1: "NORTH_EAST",
    2: "NORTH_WEST",
    4: "YORKSHIRE",
    5: "EAST_MIDLANDS",
    6: "WEST_MIDLANDS",
    7: "EAST_OF_ENGLAND",
    8: "LONDON",
    9: "SOUTH_EAST",
    10: "SOUTH_WEST",
    11: "WALES",
    12: "SCOTLAND",
}

PREDICTOR_VARIABLES = [
    "household_net_income",
    "num_adults",
    "num_children",
    "private_pension_income",
    "employment_income",
    "self_employment_income",
    "capital_income",
    "num_bedrooms",
    "council_tax",
    "is_renting",
    "region",
]

IMPUTE_VARIABLES = [
    "owned_land",
    "property_wealth",
    "corporate_wealth",
    "gross_financial_wealth",
    "net_financial_wealth",
    "main_residence_value",
    "other_residential_property_value",
    "non_residential_property_value",
    "savings",
    "num_vehicles",
    "student_loan_balance",
]

WAS_RENAMES = {
    "R8xshhwgt": "household_weight",
    # Components for estimating land holdings.
    "DVLUKValR8_sum": "owned_land",  # In the UK.
    "DVPropertyR8": "property_wealth",
    "DVFESHARESR8_aggr": "emp_shares_options",
    "DVFShUKVR8_aggr": "uk_shares",
    "DVIISAVR8_aggr": "investment_isas",
    "DVFCollVR8_aggr": "unit_investment_trusts",
    "totalpenr8_aggr": "pensions",
    "dvvaldbt_scaper8_aggr": "db_pensions",
    # Predictors for fusing to FRS.
    "dvtotgirR8": "gross_income",
    "NumAdultR8": "num_adults",
    "NumCh18R8": "num_children",
    # Household Gross Annual income from occupational or private pensions
    "DVGIPPENR8_AGGR": "private_pension_income",
    "DVGISER8_AGGR": "self_employment_income",
    # Household Gross annual income from investments
    "DVGIINVR8_aggr": "capital_income",
    # Household Total Annual Gross employee income
    "DVGIEMPR8_AGGR": "employment_income",
    "HBedRmR8": "num_bedrooms",
    "GORR8": "region",
    "DVPriRntR8": "is_renter",  # {1, 2} TODO: Get codebook values.
    "CTAmtR8": "council_tax",
    # Other columns for reference.
    "DVLOSValR8_sum": "non_uk_land",
    "HFINWNTR8_Sum": "net_financial_wealth",
    "DVLUKDebtR8_sum": "uk_land_debt",
    "HFINWR8_SUM": "gross_financial_wealth",
    "TotalWlthR8": "wealth",
    "DVhvalueR8": "main_residence_value",
    "DVHseValR8_sum": "other_residential_property_value",
    "DVBlDValR8_sum": "non_residential_property_value",
    "DVTotinc_bhcR8": "household_net_income",
    "DVSaValR8_aggr": "savings",
    "vcarnr8": "num_vehicles",
    "Tot_LosR8_aggr": "total_loans",
    "Tot_los_exc_SLCR8_aggr": "total_loans_exc_slc",
}


def generate_was_table(was: pd.DataFrame):
    """
    Clean and transform WAS data for model training.

    Args:
        was: Raw WAS survey data DataFrame.

    Returns:
        Cleaned DataFrame with renamed columns and computed variables.
    """
    was = was.rename(columns={col: col.lower() for col in was.columns})

    to_remove = []
    to_add = {}

    RENAMES = {x.lower(): y for x, y in WAS_RENAMES.items()}

    for key in RENAMES:
        key = key.lower()
        old_key = str(key)
        if key not in was.columns:
            key = key.replace("r", "w")
        if key not in was.columns:
            key = key.replace("w", "r")
        if key not in was.columns:
            raise ValueError(f"Could not find column {key}")
        else:
            to_add[key] = RENAMES[old_key]
            to_remove.append(old_key)

    for key in to_remove:
        del RENAMES[key]

    for key in to_add:
        RENAMES[key] = to_add[key]

    was = was.rename(columns=RENAMES).fillna(0)[list(RENAMES.values())]

    was["is_renting"] = was["is_renter"] == 1

    was["non_db_pensions"] = was.pensions - was.db_pensions
    was["corporate_wealth"] = was[
        [
            "non_db_pensions",
            "emp_shares_options",
            "uk_shares",
            "investment_isas",
            "unit_investment_trusts",
        ]
    ].sum(axis=1)
    was["student_loan_balance"] = was["total_loans"] - was["total_loans_exc_slc"]
    was["region"] = was["region"].map(REGIONS)
    return was


WEALTH_MODEL_METADATA = {
    "was_release_name": CURRENT_WAS_RELEASE.name,
    "was_household_tab_filename": CURRENT_WAS_RELEASE.household_tab_filename,
    "predictor_variables": tuple(PREDICTOR_VARIABLES),
    "impute_variables": tuple(IMPUTE_VARIABLES),
}


def get_wealth_model_metadata() -> dict:
    return dict(WEALTH_MODEL_METADATA)


def get_wealth_model_path():
    return STORAGE_FOLDER / WEALTH_MODEL_FILENAME


def _wealth_model_matches_current_release(model: QRF) -> bool:
    """Check whether a cached wealth model was trained with current inputs."""
    if getattr(model, "metadata", {}) != get_wealth_model_metadata():
        return False

    trained_outputs = getattr(model.model, "imputed_variables", None)
    return list(trained_outputs) == IMPUTE_VARIABLES


def _person_column(person: pd.DataFrame, name: str, default) -> pd.Series:
    if name in person:
        return person[name]
    return pd.Series(default, index=person.index)


def _allocate_student_loan_balance_to_people(
    household_balances: pd.Series,
    person: pd.DataFrame,
) -> np.ndarray:
    """
    Allocate household-imputed student loan balances to plausible holders.

    The WAS target is household-level, but `student_loan_balance` is a person-
    level input in `policyengine-uk`. We therefore allocate each household's
    imputed balance to the most plausible holder set in priority order:
    current repayers, reported borrowers, tertiary-qualified adults, current
    tertiary students, then working-age adults as a final fallback.
    """
    balances = np.zeros(len(person), dtype=float)
    if len(person) == 0:
        return balances

    age = (
        pd.to_numeric(_person_column(person, "age", 0), errors="coerce")
        .fillna(0)
        .to_numpy()
    )
    repayments = (
        pd.to_numeric(
            _person_column(person, "student_loan_repayments", 0), errors="coerce"
        )
        .fillna(0)
        .to_numpy()
    )
    reported_loans = (
        pd.to_numeric(_person_column(person, "student_loans", 0), errors="coerce")
        .fillna(0)
        .to_numpy()
    )
    current_education = (
        _person_column(person, "current_education", "NOT_IN_EDUCATION")
        .fillna("NOT_IN_EDUCATION")
        .astype(str)
        .to_numpy()
    )
    highest_education = (
        _person_column(person, "highest_education", "UPPER_SECONDARY")
        .fillna("UPPER_SECONDARY")
        .astype(str)
        .to_numpy()
    )

    group_indices = person.groupby("person_household_id").indices

    for household_id, household_balance in household_balances.items():
        if household_balance <= 0 or household_id not in group_indices:
            continue

        idx = np.asarray(group_indices[household_id], dtype=int)
        repayer_mask = repayments[idx] > 0
        borrower_mask = reported_loans[idx] > 0
        tertiary_grad_mask = highest_education[idx] == "TERTIARY"
        current_student_mask = current_education[idx] == "TERTIARY"
        working_age_mask = (age[idx] >= 18) & (age[idx] <= 55)

        for mask in (
            repayer_mask,
            borrower_mask,
            tertiary_grad_mask,
            current_student_mask,
            working_age_mask,
            np.ones(len(idx), dtype=bool),
        ):
            if mask.any():
                chosen = idx[mask]
                break

        if repayer_mask.any() and np.sum(repayments[idx][repayer_mask]) > 0:
            weights = repayments[idx][repayer_mask]
            balances[idx[repayer_mask]] += household_balance * (weights / weights.sum())
        else:
            balances[chosen] += household_balance / len(chosen)

    return balances


def save_imputation_models():
    """
    Train and save wealth imputation model.

    Returns:
        Trained QRF model.
    """
    was = pd.read_csv(
        WAS_TAB_FOLDER / CURRENT_WAS_RELEASE.household_tab_filename,
        sep="\t",
        low_memory=False,
    )
    was = generate_was_table(was)

    wealth = QRF()
    wealth.metadata = get_wealth_model_metadata()

    wealth.fit(
        was[PREDICTOR_VARIABLES],
        was[IMPUTE_VARIABLES],
    )
    wealth.save(get_wealth_model_path())
    return wealth


def create_wealth_model(overwrite_existing: bool = False):
    """
    Create or load wealth imputation model.

    Args:
        overwrite_existing: Whether to retrain model if it exists.

    Returns:
        QRF model for wealth imputation.
    """
    model_path = get_wealth_model_path()
    if model_path.exists() and not overwrite_existing:
        wealth = QRF(file_path=model_path)
        if _wealth_model_matches_current_release(wealth):
            return wealth
    return save_imputation_models()


def impute_wealth(dataset: UKSingleYearDataset) -> UKSingleYearDataset:
    """
    Impute household wealth variables using trained model.

    Uses WAS-trained models to predict various wealth components for
    households based on income, demographics, and housing characteristics.
    Vehicle ownership is calibrated to NTS 2024 targets.

    Args:
        dataset: PolicyEngine UK dataset to augment with wealth data.

    Returns:
        Dataset with household wealth variables added to the household table and
        `student_loan_balance` allocated to people.
    """
    dataset = dataset.copy()

    model = create_wealth_model()
    sim = Microsimulation(dataset=dataset)
    predictors = model.input_columns

    input_df = sim.calculate_dataframe(predictors, map_to="household")

    input_df["region"] = input_df["region"].replace(
        "NORTHERN_IRELAND", "WALES"
    )  # WAS doesn't sample NI -> put NI households in Wales (closest aggregate)
    output_df = model.predict(input_df)

    for column in output_df.columns:
        if column == "student_loan_balance":
            dataset.person[column] = _allocate_student_loan_balance_to_people(
                household_balances=output_df[column].clip(lower=0),
                person=dataset.person,
            )
            continue
        dataset.household[column] = output_df[column].values

    dataset.validate()

    return dataset
