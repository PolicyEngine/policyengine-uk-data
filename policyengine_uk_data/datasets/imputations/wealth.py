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

# Chain order matters: each target is fitted on the predictors plus every
# earlier target. Private pension wealth takes the slot the old pension-laden
# corporate_wealth held, the share-like components follow it, and each
# secured debt comes after the asset it is secured on.
IMPUTE_VARIABLES = [
    "owned_land",
    "property_wealth",
    "private_pension_wealth",
    "directly_held_shares",
    "unit_and_investment_trusts",
    "stocks_and_shares_isa",
    "gross_financial_wealth",
    "net_financial_wealth",
    "main_residence_value",
    "other_residential_property_value",
    "non_residential_property_value",
    "savings",
    "num_vehicles",
    "student_loan_balance",
    "cash_isa",
    "other_residential_property_secured_debt",
    "non_residential_property_secured_debt",
    "owned_land_secured_debt",
    # Append new outputs: retain #501's chain order and quantile seeds.
    "overseas_property_value",
    "other_property_value",
]

# corporate_wealth is not imputed: it is the sum of its imputed components,
# so the identity holds on every household.
CORPORATE_WEALTH_COMPONENTS = (
    "directly_held_shares",
    "unit_and_investment_trusts",
    "stocks_and_shares_isa",
)

# Debt secured on each capital asset (UC Regs 2013 reg. 49(1)(b) and its
# legacy equivalents deduct it from that asset's value only).
SECURED_DEBT_ASSETS = {
    "other_residential_property_secured_debt": "other_residential_property_value",
    "non_residential_property_secured_debt": "non_residential_property_value",
    "owned_land_secured_debt": "owned_land",
}

# microimpute draws each target's quantile from the same seed, so a household
# gets the same quantile for every target. Chained on earlier draws, that makes
# sparse targets near-certain for households drawn high for a related asset
# (nearly every imputed land holder had debt secured on the land). Each target
# gets its own seed so the draws are independent.
QUANTILE_DRAW_SEEDS = {
    variable: 1_000 + index for index, variable in enumerate(IMPUTE_VARIABLES)
}

# WAS round 8 sources of the targets built from more than one column. Part of
# the model metadata, so a change of definition retrains a cached model.
DERIVED_TARGETS = {
    "private_pension_wealth": ("totalpenr8_aggr", "-dvvaldbt_scaper8_aggr"),
    "directly_held_shares": ("DVFShUKVR8_aggr", "DVFESHARESR8_aggr"),
    "other_residential_property_value": ("DVHseValR8_sum", "DVBltValR8_sum"),
    "other_residential_property_secured_debt": (
        "DVHseDebtR8_sum",
        "DVBLtDebtR8_sum",
    ),
    "student_loan_balance": ("Tot_LosR8_aggr", "-Tot_los_exc_SLCR8_aggr"),
}

# Seven WAS source terms become six model inputs: #501 combines second
# homes and buy-to-let. Keep direct UK land and overseas property distinct.
PROPERTY_COMPONENTS = (
    "main_residence_value",
    "other_residential_property_value",
    "non_residential_property_value",
    "owned_land",
    "overseas_property_value",
    "other_property_value",
)

# Keep the source total as a sequential predictor, but let the companion
# model sum the six uprated inputs, covering all seven WAS source terms.
# Independent predictions need not reconcile to the conditioning draw.
CONDITIONING_ONLY_VARIABLES = ("property_wealth",)

WAS_RENAMES = {
    "R8xshhwgt": "household_weight",
    # Components for estimating land holdings.
    "DVLUKValR8_sum": "owned_land",  # In the UK.
    "DVLUKDebtR8_sum": "owned_land_secured_debt",
    "DVPropertyR8": "property_wealth",
    # UK shares (listed or not) and employee shares and options, held outside
    # ISAs and pooled funds.
    "DVFESHARESR8_aggr": "emp_shares_options",
    "DVFShUKVR8_aggr": "uk_shares",
    # Investment ISAs: the survey's stocks and shares ISA question.
    "DVIISAVR8_aggr": "stocks_and_shares_isa",
    "DVCISAVR8_aggr": "cash_isa",
    # Unit trusts and investment trusts: one survey question.
    "DVFCollVR8_aggr": "unit_and_investment_trusts",
    # Total private pension wealth and its current-employment defined benefit
    # part, both valued on the SCAPE basis.
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
    # Overseas land AND property (including dwellings/time-shares), not land only.
    "DVLOSValR8_sum": "overseas_property_value",
    "HFINWNTR8_Sum": "net_financial_wealth",
    "HFINWR8_SUM": "gross_financial_wealth",
    "TotalWlthR8": "wealth",
    "DVhvalueR8": "main_residence_value",
    # Gross values of property other than the main residence, and the
    # mortgages and loans secured on each class.
    "DVHseValR8_sum": "second_homes_value",
    "DVHseDebtR8_sum": "second_homes_debt",
    "DVBltValR8_sum": "buy_to_let_value",
    "DVBLtDebtR8_sum": "buy_to_let_debt",
    "DVBlDValR8_sum": "non_residential_property_value",
    "DVBldDebtR8_sum": "non_residential_property_secured_debt",
    "DVOprValR8_sum": "other_property_value",
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

    # WAS gross property fields use -9 (not applicable) and -8 (unknown/refused).
    # Treat these as missing, like NaN. Gross asset values cannot be negative;
    # reject undocumented codes without exposing any household records.
    # Validate the seven raw terms before combining residential values: a
    # missing/negative buy-to-let value must not reduce or hide in second homes.
    gross_columns = [
        "property_wealth",
        "main_residence_value",
        "second_homes_value",
        "buy_to_let_value",
        "non_residential_property_value",
        "owned_land",
        "overseas_property_value",
        "other_property_value",
    ]
    was[gross_columns] = was[gross_columns].replace({-9: 0, -8: 0})
    negative = was[gross_columns].lt(0).sum()
    if negative.any():
        raise ValueError(
            "Unexpected negative WAS gross property values (household counts): "
            + ", ".join(f"{name}={count}" for name, count in negative.items() if count)
        )

    was["is_renting"] = was["is_renter"] == 1

    # Private pension wealth other than current-employment defined benefit
    # rights: pension rights are disregarded capital in every means test, so it
    # is kept out of corporate_wealth.
    was["private_pension_wealth"] = was.pensions - was.db_pensions
    was["directly_held_shares"] = was.uk_shares + was.emp_shares_options
    was["corporate_wealth"] = was[list(CORPORATE_WEALTH_COMPONENTS)].sum(axis=1)
    was["other_residential_property_value"] = (
        was.second_homes_value + was.buy_to_let_value
    )
    was["other_residential_property_secured_debt"] = (
        was.second_homes_debt + was.buy_to_let_debt
    )
    was["student_loan_balance"] = was["total_loans"] - was["total_loans_exc_slc"]
    was["region"] = was["region"].map(REGIONS)
    return was


WEALTH_MODEL_METADATA = {
    "was_release_name": CURRENT_WAS_RELEASE.name,
    "was_household_tab_filename": CURRENT_WAS_RELEASE.household_tab_filename,
    "predictor_variables": tuple(PREDICTOR_VARIABLES),
    "impute_variables": tuple(IMPUTE_VARIABLES),
    "derived_targets": tuple(DERIVED_TARGETS.items()),
    "quantile_draw_seeds": tuple(QUANTILE_DRAW_SEEDS.items()),
    "property_mapping_version": 2,
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


def derive_wealth_outputs(output_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add the outputs derived from imputed targets.

    The share-like components are clipped at zero and summed into
    corporate_wealth, so the identity holds on every household. A secured debt
    is kept only where the household holds the asset it is secured on.
    """
    output_df = output_df.copy()
    for column in (*CORPORATE_WEALTH_COMPONENTS, *SECURED_DEBT_ASSETS):
        output_df[column] = output_df[column].clip(lower=0)
    output_df["corporate_wealth"] = output_df[list(CORPORATE_WEALTH_COMPONENTS)].sum(
        axis=1
    )
    for debt, asset in SECURED_DEBT_ASSETS.items():
        output_df[debt] = output_df[debt].where(output_df[asset] > 0, 0.0)
    return output_df


def use_independent_quantile_draws(model: QRF) -> QRF:
    """Give each imputed target its own quantile-draw seed."""
    for variable, seed in QUANTILE_DRAW_SEEDS.items():
        model.model.models[variable].seed = seed
    return model


def store_wealth_predictions(
    dataset: UKSingleYearDataset, predictions: pd.DataFrame
) -> UKSingleYearDataset:
    """Write imputed wealth to the dataset, skipping conditioning-only outputs.

    `student_loan_balance` is allocated to people; every other saved output is
    a household column.
    """
    # Also remove an inherited saved total when augmenting an existing dataset.
    dataset.household.drop(
        columns=list(CONDITIONING_ONLY_VARIABLES), errors="ignore", inplace=True
    )
    for column in predictions.columns:
        if column in CONDITIONING_ONLY_VARIABLES:
            continue
        if column == "student_loan_balance":
            dataset.person[column] = _allocate_student_loan_balance_to_people(
                household_balances=predictions[column].clip(lower=0),
                person=dataset.person,
            )
            continue
        dataset.household[column] = predictions[column].values
    return dataset


def require_property_component_model():
    """Reject a model that would silently omit the newly retained categories.

    The companion model changes, including the UK-only land formula, are
    specified in docs/property-wealth-model-changes.md. They must land before
    building a release with this mapping.
    """
    from policyengine_uk.system import system

    new_components = {
        "overseas_property_value",
        "other_property_value",
    }
    missing = new_components - system.variables.keys()
    if missing:
        raise ValueError(
            "Property imputation requires the companion policyengine-uk change "
            "in docs/property-wealth-model-changes.md; missing variables: "
            + ", ".join(sorted(missing))
        )
    property_adds = sorted(system.variables["property_wealth"].adds or ())
    residential_adds = sorted(system.variables["residential_property_value"].adds or ())
    if property_adds != sorted(
        [
            "residential_property_value",
            "non_residential_property_value",
            "owned_land",
            "overseas_property_value",
            "other_property_value",
        ]
    ) or residential_adds != sorted(
        [
            "main_residence_value",
            "other_residential_property_value",
        ]
    ):
        raise ValueError(
            "policyengine-uk property aggregates must include all seven WAS "
            "categories; apply docs/property-wealth-model-changes.md"
        )
    required_index = "gov.economic_assumptions.indices.obr.per_capita.gdp"
    if any(
        system.variables[name].uprating != required_index
        for name in PROPERTY_COMPONENTS
    ):
        raise ValueError(
            "Every WAS property component requires the GDP-per-capita model uprating index"
        )

    # The UK dataset loader materialises yearly inputs using this registry;
    # variable attributes alone do not uprate those explicitly supplied arrays.
    from pathlib import Path
    import yaml
    from policyengine_uk.data import economic_assumptions

    with open(
        Path(economic_assumptions.__file__).parent / "uprating_indices.yaml"
    ) as registry_file:
        registry = yaml.safe_load(registry_file)
    registered = [name for names in registry.values() for name in names]
    gdp_registered = registry.get(
        "gov.economic_assumptions.yoy_growth.obr.per_capita.gdp", ()
    )
    if any(
        registered.count(name) != 1 or name not in gdp_registered
        for name in PROPERTY_COMPONENTS
    ):
        raise ValueError(
            "Every WAS property component requires UK dataset uprating registration"
        )

    # Exercise the installed formula directly: adding the inputs and aggregate
    # without changing land estimation would still double-count UK land and
    # tax overseas assets. These are synthetic formula probes, not survey data.
    from policyengine_core.periods import period
    from policyengine_uk_data.datasets.frs_release import CURRENT_FRS_RELEASE

    probe_period = period(CURRENT_FRS_RELEASE.base_year)
    regions = system.variables["region"].possible_values
    inputs = {
        "region": regions.encode(np.array(["LONDON", "LONDON", "LONDON"])),
        "property_wealth": np.array([100.0, 100.0, 150.0]),
        "owned_land": np.array([100.0, 0.0, 20.0]),
        "overseas_property_value": np.array([0.0, 100.0, 30.0]),
        **{
            name: np.zeros(3)
            for name in PROPERTY_COMPONENTS
            if name not in ("owned_land", "overseas_property_value")
        },
        "main_residence_value": np.array([0.0, 0.0, 100.0]),
    }
    formula = system.variables["household_land_value"].get_formula(probe_period)
    land = formula(lambda name, _: inputs[name], probe_period, system.parameters)
    regional = system.parameters(
        probe_period
    ).household.wealth.land.intensity.property_wealth_by_region
    expected_land = [100.0, 0.0, 100.0 * regional.LONDON + 20.0]
    if not np.allclose(land, expected_land):
        raise ValueError(
            "policyengine-uk land formula must count UK land once and exclude "
            "overseas property; apply docs/property-wealth-model-changes.md"
        )


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
    use_independent_quantile_draws(wealth)
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

    require_property_component_model()
    model = create_wealth_model()
    sim = Microsimulation(dataset=dataset)
    predictors = model.input_columns

    input_df = sim.calculate_dataframe(predictors, map_to="household")

    input_df["region"] = input_df["region"].replace(
        "NORTHERN_IRELAND", "WALES"
    )  # WAS doesn't sample NI -> put NI households in Wales (closest aggregate)
    output_df = derive_wealth_outputs(model.predict(input_df))
    dataset = store_wealth_predictions(dataset, output_df)
    dataset.validate()

    return dataset
