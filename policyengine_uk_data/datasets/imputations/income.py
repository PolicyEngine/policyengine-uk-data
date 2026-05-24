"""
Income imputation using Survey of Personal Incomes data.

This module imputes detailed income components (employment, self-employment,
pensions, property, savings interest, dividends) using machine learning
models trained on HMRC Survey of Personal Incomes (SPI) data.
"""

import pandas as pd
import numpy as np
from policyengine_uk_data.storage import STORAGE_FOLDER
from policyengine_uk.data import UKSingleYearDataset
from policyengine_uk import Microsimulation
from policyengine_uk_data.datasets.spi import (
    AGE_RANGES,
    REGION_MAP,
    SPI_RELEASE_NAME,
    SPI_TAB_FILENAME,
)
from policyengine_uk_data.utils.stack import stack_datasets
from policyengine_uk_data.utils.subsample import subsample_dataset

SPI_TAB_FOLDER = STORAGE_FOLDER / SPI_RELEASE_NAME
SPI_RENAMES = dict(
    private_pension_income="PENSION",
    self_employment_income="PROFITS",
    property_income="INCPROP",
    savings_interest_income="INCBBS",
    dividend_income="DIVIDENDS",
    blind_persons_allowance="BPADUE",
    married_couples_allowance="MCAS",
    gift_aid="GIFTAID",
    capital_allowances="CAPALL",
    deficiency_relief="DEFICIEN",
    covenanted_payments="COVNTS",
    charitable_investment_gifts="GIFTINV",
    employment_expenses="EPB",
    other_deductions="MOTHDED",
    person_weight="FACT",
    benunit_weight="FACT",
    household_weight="FACT",
    state_pension="SRP",
)


def _spi_age_bounds(age_code) -> tuple[int, int]:
    try:
        return AGE_RANGES[int(age_code)]
    except (TypeError, ValueError, KeyError):
        return AGE_RANGES[-1]


def generate_spi_table(
    spi: pd.DataFrame,
    seed: int = 0,
    sample_size: int | None = 100_000,
):
    """
    Clean and transform SPI data for income imputation model training.

    Args:
        spi: Raw SPI survey data DataFrame.

    Returns:
        Cleaned DataFrame with age and region mappings applied.
    """
    rng = np.random.default_rng(seed)
    age_range = spi.AGERANGE
    bounds = np.array([_spi_age_bounds(age) for age in age_range])
    spi["age"] = bounds[:, 0] + rng.random(len(spi)) * (bounds[:, 1] - bounds[:, 0])

    spi["region"] = spi.GORCODE.map(REGION_MAP).fillna("UNKNOWN")

    spi["gender"] = np.where(spi.SEX == 1, "MALE", "FEMALE")

    for rename in SPI_RENAMES:
        spi[rename] = spi[SPI_RENAMES[rename]]

    spi["employment_income"] = spi[["PAY", "EPB", "TAXTERM"]].sum(axis=1)

    if sample_size is not None:
        spi = pd.concat(
            [
                spi.sample(
                    sample_size,
                    weights=spi.person_weight,
                    replace=True,
                    random_state=seed,
                ),
            ]
        )

    return spi


PREDICTORS = [
    "age",
    "gender",
    "region",
]

INCOME_COMPONENTS = [
    "employment_income",
    "self_employment_income",
    "savings_interest_income",
    "dividend_income",
    "private_pension_income",
    "property_income",
]

# Gift Aid (SPI GIFTAID) and charitable investment gifts (SPI GIFTINV) are
# separate reliefs on the UK side but both absent from the FRS — without them
# in the model outputs, the zero-weight SPI-donor rows carry a middle-income
# FRS donor's (always zero) charitable giving, missing the £1-1.5bn/yr Gift
# Aid higher-rate relief flow and an additional ~£0.1bn of qualifying-
# investment gifts. Including them here means the multi-output QRF draws
# them jointly with income components, so high-earner donors get plausibly
# non-zero values. Kept separate from INCOME_COMPONENTS because the
# rent/mortgage adjustment factor downstream is built from income sums, and
# these are expenditures, not income. The standalone SPI dataset in
# `datasets/spi.py` sums GIFTAID + GIFTINV into a single `gift_aid` column
# because that path doesn't carry a separate `charitable_investment_gifts`
# variable; the enhanced-FRS path here keeps them separate so each maps to
# its own policyengine-uk variable.
IMPUTATIONS = INCOME_COMPONENTS + ["gift_aid", "charitable_investment_gifts"]


INCOME_MODEL_METADATA = {
    "spi_release_name": SPI_RELEASE_NAME,
    "spi_tab_filename": SPI_TAB_FILENAME,
    "imputations": tuple(IMPUTATIONS),
}
INCOME_MODEL_PATH = STORAGE_FOLDER / f"income_{SPI_RELEASE_NAME}.pkl"


def _income_model_matches_current_release(model) -> bool:
    if getattr(model, "metadata", {}) != INCOME_MODEL_METADATA:
        return False

    cached_outputs = set(getattr(model.model, "imputed_variables", []))
    return cached_outputs == set(IMPUTATIONS)


def save_imputation_models():
    """
    Train and save income imputation model.

    Returns:
        Trained QRF model for income imputation.
    """
    from policyengine_uk_data.utils import QRF

    income = QRF()
    income.metadata = INCOME_MODEL_METADATA
    spi = pd.read_csv(SPI_TAB_FOLDER / SPI_TAB_FILENAME, delimiter="\t")
    spi = generate_spi_table(spi)
    spi = spi[PREDICTORS + IMPUTATIONS]
    income.fit(spi[PREDICTORS], spi[IMPUTATIONS])
    income.save(INCOME_MODEL_PATH)
    return income


def create_income_model(overwrite_existing: bool = False):
    """
    Create or load income imputation model.

    If a cached model exists and its training metadata or output columns don't
    match the current SPI release and ``IMPUTATIONS`` list, the cache is
    discarded and the model is retrained.

    Args:
        overwrite_existing: Whether to retrain model if it exists.

    Returns:
        QRF model for income imputation.
    """
    from policyengine_uk_data.utils.qrf import QRF

    if INCOME_MODEL_PATH.exists() and not overwrite_existing:
        cached = QRF(file_path=INCOME_MODEL_PATH)
        if _income_model_matches_current_release(cached):
            return cached
        # Cached model was trained against a different SPI release or output set.
    return save_imputation_models()


def impute_over_incomes(
    dataset: UKSingleYearDataset, model, output_variables: list[str]
) -> pd.DataFrame:
    """
    Impute specified income components using trained model.

    Args:
        dataset: PolicyEngine UK dataset to augment with income data.
        output_variables: List of income components to impute.

    Returns:
        DataFrame with imputed income components.
    """
    dataset = dataset.copy()
    sim = Microsimulation(dataset=dataset)
    input_df = sim.calculate_dataframe(["age", "gender", "region"])
    output_df = model.predict(input_df)

    for column in output_variables:
        dataset.person[column] = output_df[column].fillna(0).values

    # Housing costs (rent, mortgage interest, mortgage capital) used to be
    # rescaled here by new_income_total / original_income_total across
    # INCOME_COMPONENTS. Because FRS dividend_income is near-zero and the
    # SPI-trained QRF predicts materially larger dividends, the ratio
    # inflated rent/mortgage by ~2.5× uniformly in the built enhanced FRS
    # — pushing AHC poverty rates 10–18 pp above HBAI for non-pensioners
    # (see issue #367). Housing costs now pass through unchanged; their
    # year-on-year growth is handled by per-variable OBR uprating indices,
    # not by income-imputation side-effects.

    return dataset


def impute_income(dataset: UKSingleYearDataset) -> UKSingleYearDataset:
    """
    Impute detailed income components using trained model.

    Uses SPI-trained models to predict various income sources for individuals
    based on age, gender, and region. Creates a synthetic population with
    the imputed income data.

    Args:
        dataset: PolicyEngine UK dataset to augment with income data.

    Returns:
        Combined dataset with original data plus synthetic high-income individuals.
    """
    # Impute wealth, assuming same time period as trained data
    dataset = dataset.copy()
    # gift_aid and charitable_investment_gifts are in IMPUTATIONS but are not
    # columns on the raw FRS build, so initialise them to zero everywhere
    # before imputation. Without this, the full-FRS half stays NaN for these
    # columns (they're never touched by the dividend-only impute_over_incomes
    # call below), and the eventual stacked dataset fails validate().
    for column in ("gift_aid", "charitable_investment_gifts"):
        if column not in dataset.person.columns:
            dataset.person[column] = 0.0
    zero_weight_copy = dataset.copy()
    zero_weight_copy.household.household_weight = 0
    zero_weight_copy = subsample_dataset(zero_weight_copy, 10_000)

    model = create_income_model()

    # Impute just dividends on the original, full variable set on the copy

    zero_weight_copy = impute_over_incomes(
        zero_weight_copy,
        model,
        IMPUTATIONS,
    )

    # Second-stage QRF: rewrite FRS-only variables (benefit `_reported`
    # columns, pension contributions, savings, etc.) on the SPI-donor rows
    # so they correlate with the freshly-imputed incomes instead of staying
    # as whatever middle-income FRS donor was sampled. Without this the
    # £2M imputed earners keep their donor's £120 UC receipt, blowing up
    # benefit aggregates under calibration upweight.
    from policyengine_uk_data.datasets.imputations.frs_only import (
        impute_frs_only_variables,
    )
    from policyengine_uk_data.datasets.disability_benefits import (
        strip_internal_disability_reported_amounts,
    )

    zero_weight_copy = impute_frs_only_variables(
        train_dataset=dataset,
        target_dataset=zero_weight_copy,
    )

    dataset = impute_over_incomes(
        dataset,
        model,
        ["dividend_income"],
    )

    zero_weight_copy.validate()
    dataset.validate()

    data = stack_datasets(
        dataset,
        zero_weight_copy,
    )

    return strip_internal_disability_reported_amounts(data)
