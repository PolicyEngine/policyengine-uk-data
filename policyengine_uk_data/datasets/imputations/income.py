"""
Income imputation using Survey of Personal Incomes data.

This module imputes detailed income components (employment, self-employment,
pensions, property, savings interest, dividends) using machine learning
models trained on HMRC Survey of Personal Incomes (SPI) data.

The draw is conditioned on each person's earnings group (see
``EARNINGS_GROUPS``) as well as their age, gender and region, so the incomes
agree with the employment status the FRS donor row keeps.
"""

import pandas as pd
import numpy as np
import os
import pickle
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
from policyengine_uk_data.utils.employment_status import (
    CHILD_STATUS,
    EMPLOYEE_STATUSES,
    SELF_EMPLOYED_STATUSES,
)

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


# Earnings groups that both surveys identify. The SPI has no ILO employment
# status and no hours, but it records each taxpayer's income sources: pay, and
# whether they file the self-employment pages of a tax return for a trade or
# partnership (SEINC_NUM, "Indicator for self-employed cases"). PROFITS is
# floored at zero, so SEINC_NUM is what finds break-even and loss-making
# traders. The FRS gives the same split through the main job's ILO status
# (EMPSTATI) and any second-job earnings.
NO_EARNINGS = "NO_EARNINGS"
EMPLOYEE = "EMPLOYEE"
SELF_EMPLOYED = "SELF_EMPLOYED"
EMPLOYEE_AND_SELF_EMPLOYED = "EMPLOYEE_AND_SELF_EMPLOYED"
EARNINGS_GROUPS = (
    NO_EARNINGS,
    EMPLOYEE,
    SELF_EMPLOYED,
    EMPLOYEE_AND_SELF_EMPLOYED,
)
# FRS rows that take no SPI draw and keep their own values: the FRS child
# table (dependent children, including 16-19-year-olds in education), whose
# earnings this FRS build does not record, and anyone under 16. With no
# earnings or job information, there is nothing to tie a taxpayer's SPI
# incomes to.
NOT_IMPUTED = "NOT_IMPUTED"


# Every earnings group gets at least this share of the nominal training sample
# size, so the small self-employed groups are not fitted on a few thousand
# records. Floors are added on top, so the sample can exceed the nominal size.
MIN_GROUP_SAMPLE_SHARE = 0.1


def earnings_group(has_pay, has_trade) -> np.ndarray:
    """Earnings group from whether a person has pay and a trade."""
    has_pay = np.asarray(has_pay, dtype=bool)
    has_trade = np.asarray(has_trade, dtype=bool)
    return np.select(
        [has_pay & has_trade, has_trade, has_pay],
        [EMPLOYEE_AND_SELF_EMPLOYED, SELF_EMPLOYED, EMPLOYEE],
        NO_EARNINGS,
    ).astype(object)


def spi_earnings_group(
    employment_income, self_employment_income, self_employed_indicator
) -> np.ndarray:
    """Earnings group of SPI records.

    Pay is PAY + EPB + TAXTERM. A trade is SEINC_NUM = 1 (self-employment
    pages filed, whatever the profit) or any assessable profit.
    """
    has_trade = (np.asarray(self_employed_indicator) == 1) | (
        np.asarray(self_employment_income, dtype=float) > 0
    )
    return earnings_group(np.asarray(employment_income, dtype=float) > 0, has_trade)


def frs_earnings_group(
    employment_status, age, employment_income, self_employment_income
) -> np.ndarray:
    """Earnings group of FRS rows, or ``NOT_IMPUTED`` for children.

    An employee main job always draws pay and a self-employed main job always
    draws a trade, whatever the FRS recorded for the donor. Earnings the FRS
    records outside the main job (a second job or a side trade) add the other
    source. Everyone else, retired, unemployed or inactive, draws from SPI
    records with neither.
    """
    status = np.asarray(employment_status, dtype=object)
    has_pay = np.isin(status, EMPLOYEE_STATUSES) | (
        np.asarray(employment_income, dtype=float) > 0
    )
    has_trade = np.isin(status, SELF_EMPLOYED_STATUSES) | (
        np.asarray(self_employment_income, dtype=float) > 0
    )
    is_child = (status == CHILD_STATUS) | (np.asarray(age, dtype=float) < 16)
    return np.where(is_child, NOT_IMPUTED, earnings_group(has_pay, has_trade)).astype(
        object
    )


def earnings_group_sample_sizes(
    group_weights: dict[str, float], sample_size: int
) -> dict[str, int]:
    """Training records per earnings group: the group's weighted share of
    ``sample_size``, but never under ``MIN_GROUP_SAMPLE_SHARE`` of
    ``sample_size``. Groups with no weight get no records. Floored groups are
    not offset elsewhere, so the total can exceed ``sample_size`` (by at most
    one floor per group, plus rounding)."""
    weights = {group: float(w) for group, w in group_weights.items() if w > 0}
    total = sum(weights.values())
    floor = int(np.ceil(MIN_GROUP_SAMPLE_SHARE * sample_size))
    return {
        group: max(int(round(sample_size * w / total)), floor)
        for group, w in weights.items()
    }


def generate_spi_table(
    spi: pd.DataFrame,
    seed: int = 0,
    sample_size: int | None = 100_000,
):
    """
    Clean and transform SPI data for income imputation model training.

    Args:
        spi: Raw SPI survey data DataFrame.
        seed: Seed for the age draw and the resample.
        sample_size: If set, resample records with replacement, in proportion
            to their weight within each earnings group, with each group's
            record count from ``earnings_group_sample_sizes``.

    Returns:
        Cleaned DataFrame with age, region and earnings group.
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
    spi["earnings_group"] = spi_earnings_group(
        spi.employment_income, spi.self_employment_income, spi.SEINC_NUM
    )

    if sample_size is not None:
        sizes = earnings_group_sample_sizes(
            spi.groupby("earnings_group").person_weight.sum().to_dict(),
            sample_size,
        )
        spi = pd.concat(
            [
                spi[spi.earnings_group == group].sample(
                    sizes[group],
                    weights="person_weight",
                    replace=True,
                    random_state=seed + i,
                )
                for i, group in enumerate(EARNINGS_GROUPS)
                if group in sizes
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
    "earnings_groups": EARNINGS_GROUPS,
    "min_group_sample_share": MIN_GROUP_SAMPLE_SHARE,
}
INCOME_MODEL_PATH = STORAGE_FOLDER / f"income_{SPI_RELEASE_NAME}.pkl"
INCOME_MODEL_SAMPLE_SIZE = 100_000
TESTING_INCOME_MODEL_SAMPLE_SIZE = 10_000


def get_income_model_sample_size() -> int:
    if os.environ.get("TESTING", "0") == "1":
        return TESTING_INCOME_MODEL_SAMPLE_SIZE
    return INCOME_MODEL_SAMPLE_SIZE


def get_income_model_metadata() -> dict:
    return {
        **INCOME_MODEL_METADATA,
        "sample_size": get_income_model_sample_size(),
    }


class EarningsGroupIncomeModel:
    """One QRF per earnings group, each fitted on that group's SPI records.

    A person is drawn only from SPI records in their own earnings group, so an
    employee always draws pay, a self-employed person always draws a trade
    (whose profit can be zero), and someone with neither draws neither.
    ``predict`` returns NaN for rows in no group (``NOT_IMPUTED``).
    """

    def __init__(self, models: dict, metadata: dict | None = None):
        self.models = models
        self.metadata = metadata or {}

    @property
    def imputed_variables(self) -> list[str]:
        return list(IMPUTATIONS)

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        groups = np.asarray(X["earnings_group"], dtype=object)
        output = pd.DataFrame(np.nan, index=X.index, columns=IMPUTATIONS)
        for group, model in self.models.items():
            in_group = groups == group
            if in_group.any():
                inputs = pd.DataFrame(
                    {column: np.asarray(X[column])[in_group] for column in PREDICTORS}
                )
                draws = model.predict(inputs)
                output.loc[in_group, IMPUTATIONS] = draws[IMPUTATIONS].to_numpy()
        return output

    def save(self, file_path):
        with open(file_path, "wb") as f:
            pickle.dump(
                {
                    "models": {
                        group: model.model for group, model in self.models.items()
                    },
                    "input_columns": PREDICTORS,
                    "metadata": self.metadata,
                },
                f,
            )

    @classmethod
    def load(cls, file_path):
        """The cached model, or None if the file holds another format."""
        from policyengine_uk_data.utils.qrf import QRF

        with open(file_path, "rb") as f:
            data = pickle.load(f)
        if not isinstance(data, dict) or "models" not in data:
            return None
        models = {}
        for group, fitted in data["models"].items():
            model = QRF()
            model.model = fitted
            model.input_columns = data.get("input_columns", PREDICTORS)
            models[group] = model
        return cls(models, data.get("metadata", {}))


def _income_model_matches_current_release(model) -> bool:
    if model is None or getattr(model, "metadata", {}) != get_income_model_metadata():
        return False

    models = getattr(model, "models", {})
    if set(models) != set(EARNINGS_GROUPS):
        return False
    return all(
        set(getattr(group_model.model, "imputed_variables", [])) == set(IMPUTATIONS)
        for group_model in models.values()
    )


def save_imputation_models():
    """
    Train and save the income imputation model: one QRF per earnings group.

    Returns:
        Trained ``EarningsGroupIncomeModel``.
    """
    from policyengine_uk_data.utils import QRF

    spi = pd.read_csv(SPI_TAB_FOLDER / SPI_TAB_FILENAME, delimiter="\t")
    spi = generate_spi_table(spi, sample_size=get_income_model_sample_size())
    models = {}
    for group in EARNINGS_GROUPS:
        training = spi[spi.earnings_group == group]
        model = QRF()
        model.fit(training[PREDICTORS], training[IMPUTATIONS])
        models[group] = model
    income = EarningsGroupIncomeModel(models, get_income_model_metadata())
    income.save(INCOME_MODEL_PATH)
    return income


def create_income_model(overwrite_existing: bool = False):
    """
    Create or load income imputation model.

    If a cached model exists and its training metadata, earnings groups or
    output columns don't match the current SPI release and ``IMPUTATIONS``
    list, the cache is discarded and the model is retrained.

    Args:
        overwrite_existing: Whether to retrain model if it exists.

    Returns:
        ``EarningsGroupIncomeModel`` for income imputation.
    """
    if INCOME_MODEL_PATH.exists() and not overwrite_existing:
        cached = EarningsGroupIncomeModel.load(INCOME_MODEL_PATH)
        if _income_model_matches_current_release(cached):
            return cached
        # Cached model was trained against a different SPI release, output
        # set or grouping.
    return save_imputation_models()


def income_model_inputs(dataset: UKSingleYearDataset) -> pd.DataFrame:
    """Predictors (age, gender, region) and earnings group of each person."""
    sim = Microsimulation(dataset=dataset)
    frame = sim.calculate_dataframe(["age", "gender", "region"])
    inputs = pd.DataFrame({column: np.asarray(frame[column]) for column in PREDICTORS})
    person = dataset.person
    inputs["earnings_group"] = frs_earnings_group(
        person.employment_status,
        inputs.age,
        person.employment_income,
        person.self_employment_income,
    )
    return inputs


def apply_income_draws(
    person: pd.DataFrame, draws: pd.DataFrame, groups, output_variables
) -> pd.DataFrame:
    """Write ``draws`` over ``output_variables`` for every person in an
    earnings group (missing draws become zero). ``NOT_IMPUTED`` rows keep
    their own values."""
    person = person.copy()
    drawn = np.asarray(groups, dtype=object) != NOT_IMPUTED
    for column in output_variables:
        draw = np.nan_to_num(np.asarray(draws[column], dtype=float), nan=0.0)
        own = (
            np.asarray(person[column], dtype=float)
            if column in person.columns
            else np.zeros(len(person))
        )
        person[column] = np.where(drawn, draw, own)
    return person


def impute_over_incomes(
    dataset: UKSingleYearDataset, model, output_variables: list[str]
) -> pd.DataFrame:
    """
    Impute specified income components using trained model.

    Each person draws from SPI records in their earnings group
    (``frs_earnings_group``); children keep their own values.

    Args:
        dataset: PolicyEngine UK dataset to augment with income data.
        model: Fitted ``EarningsGroupIncomeModel``.
        output_variables: List of income components to impute.

    Returns:
        DataFrame with imputed income components.
    """
    dataset = dataset.copy()
    input_df = income_model_inputs(dataset)
    output_df = model.predict(input_df)
    dataset.person = apply_income_draws(
        dataset.person, output_df, input_df.earnings_group, output_variables
    )

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
    based on age, gender, region and earnings group. Creates a synthetic
    population with the imputed income data.

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
    dataset.household["household_is_spi_synthetic"] = False
    zero_weight_copy = dataset.copy()
    zero_weight_copy.household.household_weight = 0
    zero_weight_copy.household["household_is_spi_synthetic"] = True
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
