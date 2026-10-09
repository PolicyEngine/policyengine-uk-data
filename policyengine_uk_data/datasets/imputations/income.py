"""
Income imputation using Survey of Personal Incomes data.

This module imputes detailed income components (employment, self-employment,
pensions, property, savings interest, dividends) using machine learning
models trained on HMRC Survey of Personal Incomes (SPI) data.

The draw is conditioned on each person's earnings group (see
``EARNINGS_GROUPS``) as well as their age, gender and region, so the incomes
agree with the employment status the FRS donor row keeps.

Within that cell a person does not draw at a random quantile: each income is
drawn at the person's rank of the same income among FRS people in the same
cell (``draw_quantiles``). A part-time employee with low FRS pay draws low
SPI pay, and a donor at the top of their cell draws from the top of the SPI.
The ranks are uniform within every cell, so each group's first output keeps
the forest's distribution for the cell, in expectation over the forest's
within-band age noise. Later outputs are conditioned on those drawn before
at correlated FRS ranks, so they carry more dependence between incomes than
the SPI does, though less than microimpute's single random quantile per
person did.
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
#
# People with both pay and a trade are split by which is their main source.
# The SPI's MAINSRCE is the main source indicator (1 pay, 2 occupational
# pension, 3 sole trader, 4 partnership, 5 other, 6 claims case), and the
# main source of income is one of the variables HMRC stratifies the
# self-assessment part of the sample by (HMRC, "Survey of Personal Incomes
# 2022-23: public use tape documentation", sample design, p. 3, and variable
# list, p. 15). It is not the larger income: in 2022-23, 57% (weighted) of
# those whose main source is a trade had more pay than profit. The FRS
# equivalent is the main job's status. Where neither names pay or a trade
# (any other MAINSRCE code, including -1 not classified; an FRS main job that
# is neither), the larger of pay and profit decides. FRS people with both are
# too few (about 300) for their rank within a cell to say much, so the main
# source is most of what links their draw to their own jobs.
NO_EARNINGS = "NO_EARNINGS"
EMPLOYEE = "EMPLOYEE"
SELF_EMPLOYED = "SELF_EMPLOYED"
EMPLOYEE_MAIN_AND_SELF_EMPLOYED = "EMPLOYEE_MAIN_AND_SELF_EMPLOYED"
SELF_EMPLOYED_MAIN_AND_EMPLOYEE = "SELF_EMPLOYED_MAIN_AND_EMPLOYEE"
EARNINGS_GROUPS = (
    NO_EARNINGS,
    EMPLOYEE,
    SELF_EMPLOYED,
    EMPLOYEE_MAIN_AND_SELF_EMPLOYED,
    SELF_EMPLOYED_MAIN_AND_EMPLOYEE,
)
PAY_GROUPS = (
    EMPLOYEE,
    EMPLOYEE_MAIN_AND_SELF_EMPLOYED,
    SELF_EMPLOYED_MAIN_AND_EMPLOYEE,
)
TRADE_GROUPS = (
    SELF_EMPLOYED,
    EMPLOYEE_MAIN_AND_SELF_EMPLOYED,
    SELF_EMPLOYED_MAIN_AND_EMPLOYEE,
)
SPI_PAY_MAIN_SOURCE = 1
SPI_TRADE_MAIN_SOURCES = (3, 4)
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


def earnings_group(has_pay, has_trade, pay_is_main) -> np.ndarray:
    """Earnings group from whether a person has pay and a trade, and, for
    people with both, whether pay is their main source."""
    has_pay = np.asarray(has_pay, dtype=bool)
    has_trade = np.asarray(has_trade, dtype=bool)
    both = has_pay & has_trade
    pay_is_main = np.asarray(pay_is_main, dtype=bool)
    return np.select(
        [both & pay_is_main, both, has_trade, has_pay],
        [
            EMPLOYEE_MAIN_AND_SELF_EMPLOYED,
            SELF_EMPLOYED_MAIN_AND_EMPLOYEE,
            SELF_EMPLOYED,
            EMPLOYEE,
        ],
        NO_EARNINGS,
    ).astype(object)


def spi_earnings_group(
    employment_income, self_employment_income, self_employed_indicator, main_source
) -> np.ndarray:
    """Earnings group of SPI records.

    Pay is PAY + EPB + TAXTERM. A trade is SEINC_NUM = 1 (self-employment
    pages filed, whatever the profit) or any assessable profit. Pay is the
    main source if MAINSRCE says pay, or if it names neither pay nor a trade
    and pay is at least the profit.
    """
    pay = np.asarray(employment_income, dtype=float)
    profit = np.asarray(self_employment_income, dtype=float)
    main_source = np.asarray(main_source)
    has_trade = (np.asarray(self_employed_indicator) == 1) | (profit > 0)
    pay_is_main = (main_source == SPI_PAY_MAIN_SOURCE) | (
        ~np.isin(main_source, SPI_TRADE_MAIN_SOURCES) & (pay >= profit)
    )
    return earnings_group(pay > 0, has_trade, pay_is_main)


def frs_earnings_group(
    employment_status, age, employment_income, self_employment_income
) -> np.ndarray:
    """Earnings group of FRS rows, or ``NOT_IMPUTED`` for children.

    An employee main job always draws pay and a self-employed main job always
    draws a trade, whatever the FRS recorded for the donor. Earnings the FRS
    records outside the main job (a second job or a side trade) add the other
    source, and the main job's source is the main one. Everyone else, retired,
    unemployed or inactive, draws from SPI records with neither, unless the
    FRS records earnings for them.
    """
    status = np.asarray(employment_status, dtype=object)
    pay = np.asarray(employment_income, dtype=float)
    profit = np.asarray(self_employment_income, dtype=float)
    employee = np.isin(status, EMPLOYEE_STATUSES)
    self_employed = np.isin(status, SELF_EMPLOYED_STATUSES)
    has_pay = employee | (pay > 0)
    has_trade = self_employed | (profit > 0)
    pay_is_main = employee | (~self_employed & (pay >= profit))
    is_child = (status == CHILD_STATUS) | (np.asarray(age, dtype=float) < 16)
    return np.where(
        is_child, NOT_IMPUTED, earnings_group(has_pay, has_trade, pay_is_main)
    ).astype(object)


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
        spi.employment_income,
        spi.self_employment_income,
        spi.SEINC_NUM,
        spi.MAINSRCE,
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

# The SPI gives age only as a band, so the forests learn nothing within a
# band. A person's rank is taken within (earnings group, SPI age band,
# gender, region): the cells the forests condition on. Ranking over a wider
# pool would bias the draw: a London donor's national rank is higher than
# their rank in London, and London's SPI distribution is already higher.
SPI_AGE_BAND_STARTS = sorted({low for code, (low, _) in AGE_RANGES.items() if code > 0})


def spi_age_band(age) -> np.ndarray:
    """SPI age band (1 = 16-24, ..., 7 = 74 and over; 0 = under 16)."""
    return np.searchsorted(
        SPI_AGE_BAND_STARTS, np.asarray(age, dtype=float), side="right"
    )


def rank_cells(inputs: pd.DataFrame) -> np.ndarray:
    """Integer cell of each person: earnings group, SPI age band, gender, region."""
    keys = pd.DataFrame(
        {
            "earnings_group": np.asarray(inputs["earnings_group"], dtype=object),
            "band": spi_age_band(inputs["age"]),
            "gender": np.asarray(inputs["gender"], dtype=object),
            "region": np.asarray(inputs["region"], dtype=object),
        }
    )
    return keys.groupby(list(keys.columns), sort=True, dropna=False).ngroup().to_numpy()


def rank_quantiles(values, cells, weights, ids, seed: int = 0) -> np.ndarray:
    """Each row's quantile, in [0, 1), in its cell's weighted distribution of
    ``values``.

    Within a cell, rows are ordered by value, rows with equal values (most
    often zero) in random order. Each row then holds the interval [weight of
    the rows before it, that plus its own weight) as a share of the cell's
    weight, and takes a uniform point in it. The intervals tile [0, 1), so
    the quantiles are uniform within every cell, and a lower value always
    gets a lower quantile than a higher one in the same cell. A cell of one
    row gets a uniform random quantile. A cell with no weight is ranked with
    equal weights. Random numbers are drawn in (cell, value, id) order, so
    the result does not depend on the order of the rows; ids must be unique.
    """
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    cells = pd.factorize(np.asarray(cells), sort=True)[0]
    ids = np.asarray(ids)
    if np.isnan(values).any() or np.isnan(weights).any():
        raise ValueError("Ranked values and weights must not be missing")
    if (weights < 0).any():
        raise ValueError("Rank weights must not be negative")
    if pd.Series(ids).duplicated().any():
        raise ValueError("Ranked ids must be unique")
    n = len(values)
    rng = np.random.default_rng(seed)
    canonical = np.lexsort((ids, values, cells))
    tie_order, offset = np.empty(n), np.empty(n)
    tie_order[canonical] = rng.random(n)
    offset[canonical] = rng.random(n)
    order = np.lexsort((tie_order, values, cells))
    sorted_rows = pd.DataFrame({"cell": cells[order], "weight": weights[order]})
    total = sorted_rows.groupby("cell").weight.transform("sum").to_numpy()
    if (total <= 0).any():
        sorted_rows.loc[total <= 0, "weight"] = 1.0
        total = sorted_rows.groupby("cell").weight.transform("sum").to_numpy()
    weight = sorted_rows.weight.to_numpy()
    before = sorted_rows.groupby("cell").weight.cumsum().to_numpy() - weight
    quantiles = np.empty(n)
    quantiles[order] = np.minimum(
        (before + offset[order] * weight) / total, np.nextafter(1.0, 0.0)
    )
    return quantiles


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

# Each output is drawn at the forest's conditional quantile nearest the
# person's rank, on this grid of 1,000 midpoints (0.0005 to 0.9995). The
# draw used to be microimpute 1.8's: a random pick from ten quantiles between
# 1/11 and 10/11, so no one drew from the top or bottom 9% of the forest's
# conditional distribution at their predictors.
DRAW_QUANTILE_GRID = (np.arange(1_000) + 0.5) / 1_000
# Seed for the rank tie-breaks (and for the random quantiles of a draw made
# without ranks).
DRAW_SEED = 0


def draw_quantile_column(variable: str) -> str:
    return f"{variable}_draw_quantile"


def draw_at_quantiles(results, X: pd.DataFrame, quantiles: dict, grid) -> pd.DataFrame:
    """Draw every output of a fitted microimpute QRF at given quantiles.

    Outputs are drawn in the model's order, each conditioned on the ones
    drawn before it, as microimpute's own ``predict`` does; but where that
    picks a random quantile, row ``i`` of ``variable`` takes the forest's
    conditional quantile ``grid[int(quantiles[variable][i] * len(grid))]``.
    With microimpute 1.8's grid and its random draws this reproduces its
    ``predict`` exactly (tested).
    """
    from microimpute.models.imputer import _ConstantValueModel
    from microimpute.models.qrf import (
        _get_sequential_predictors,
        _QRFModel,
        _RandomForestClassifierModel,
    )

    grid = np.asarray(grid, dtype=float)
    k = len(grid)
    augmented, _ = results.preprocess_data_types(
        X, results.original_predictors, getattr(results, "dummy_processor", None)
    )
    output = pd.DataFrame(index=X.index)
    for i, variable in enumerate(results.imputed_variables):
        model = results.models[variable]
        columns = results._get_encoded_predictors(
            _get_sequential_predictors(results.predictors, results.imputed_variables, i)
        )
        if isinstance(model, _ConstantValueModel):
            values = np.asarray(model.predict(augmented))
        elif isinstance(model, _QRFModel):
            features = augmented[columns]
            if hasattr(model, "_align_features"):  # microimpute >= 2
                features = model._align_features(features)
            pred = np.asarray(model.qrf.predict(features, quantiles=list(grid)))
            pred = pred.reshape(len(features), k)
            index = np.clip(
                (np.asarray(quantiles[variable], dtype=float) * k).astype(int), 0, k - 1
            )
            values = pred[np.arange(len(pred)), index]
        elif isinstance(model, _RandomForestClassifierModel):
            values = np.asarray(model.predict(augmented[columns], return_probs=False))
        else:
            raise TypeError(
                f"Cannot draw {variable} at given quantiles from a "
                f"{type(model).__name__}"
            )
        output[variable] = values
        augmented[variable] = values
        augmented = results._encode_imputed_variable(augmented, variable)
    return output


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
    Each output is drawn at the quantile in the input's
    ``draw_quantile_column(output)`` (see ``draw_quantiles``); without those
    columns, at independent uniform random quantiles. ``predict`` returns NaN
    for rows in no group (``NOT_IMPUTED``).
    """

    def __init__(self, models: dict, metadata: dict | None = None):
        self.models = models
        self.metadata = metadata or {}

    @property
    def imputed_variables(self) -> list[str]:
        return list(IMPUTATIONS)

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        groups = np.asarray(X["earnings_group"], dtype=object)
        rng = np.random.default_rng(DRAW_SEED)
        quantiles = {
            variable: (
                np.asarray(X[draw_quantile_column(variable)], dtype=float)
                if draw_quantile_column(variable) in X
                else rng.random(len(X))
            )
            for variable in IMPUTATIONS
        }
        output = pd.DataFrame(np.nan, index=X.index, columns=IMPUTATIONS)
        for group, model in self.models.items():
            in_group = groups == group
            if in_group.any():
                inputs = pd.DataFrame(
                    {column: np.asarray(X[column])[in_group] for column in PREDICTORS}
                )
                draws = draw_at_quantiles(
                    model.model,
                    inputs,
                    {v: q[in_group] for v, q in quantiles.items()},
                    DRAW_QUANTILE_GRID,
                )
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


def draw_quantiles(
    dataset: UKSingleYearDataset, inputs: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Each person's draw quantile for every output, indexed by person ID.

    The quantile is the person's rank of the same income in ``dataset``,
    among people in the same cell (``rank_cells``), at household weights:
    pay sets the pay draw, profit the profit draw, a private pension the
    pension draw. Ties, most often at zero, are broken at random, so an
    employee with no recorded pay draws from the bottom of their cell and
    an output the FRS does not record (gift aid) is drawn at a uniform
    random quantile.
    """
    if inputs is None:
        inputs = income_model_inputs(dataset)
    person = dataset.person
    weights = person.person_household_id.map(
        dataset.household.set_index("household_id").household_weight
    )
    cells = rank_cells(inputs)
    return pd.DataFrame(
        {
            draw_quantile_column(variable): rank_quantiles(
                person[variable]
                if variable in person.columns
                else np.zeros(len(person)),
                cells,
                weights,
                person.person_id,
                seed=DRAW_SEED + i,
            )
            for i, variable in enumerate(IMPUTATIONS)
        },
        index=person.person_id.to_numpy(),
    )


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
    dataset: UKSingleYearDataset,
    model,
    output_variables: list[str],
    quantiles: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Impute specified income components using trained model.

    Each person draws from SPI records in their earnings group
    (``frs_earnings_group``), at their draw quantiles; children keep their
    own values.

    Args:
        dataset: PolicyEngine UK dataset to augment with income data.
        model: Fitted ``EarningsGroupIncomeModel``.
        output_variables: List of income components to impute.
        quantiles: Draw quantiles by person ID (``draw_quantiles``),
            covering every person in ``dataset``. Defaults to ranks within
            ``dataset`` itself.

    Returns:
        DataFrame with imputed income components.
    """
    dataset = dataset.copy()
    input_df = income_model_inputs(dataset)
    if quantiles is None:
        quantiles = draw_quantiles(dataset, input_df)
    person_quantiles = quantiles.loc[dataset.person.person_id.to_numpy()]
    for column in person_quantiles.columns:
        input_df[column] = person_quantiles[column].to_numpy()
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


def clear_frs_reported_capital(dataset: UKSingleYearDataset) -> UKSingleYearDataset:
    """Set ``pension_credit_reported_capital`` to -1 (none recorded).

    Used on the SPI-synthetic copy. The FRS benefit-unit capital belongs to the
    FRS donor, whose incomes the SPI imputation replaces; keeping it would
    assess an SPI-income unit on the donor's capital. With -1, policyengine-uk
    uses the household capital proxy for these rows.
    """
    if "pension_credit_reported_capital" in dataset.benunit.columns:
        dataset.benunit["pension_credit_reported_capital"] = -1.0
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
    zero_weight_copy = clear_frs_reported_capital(zero_weight_copy)
    zero_weight_copy = subsample_dataset(zero_weight_copy, 10_000)

    model = create_income_model()
    # Ranks come from the full, weighted FRS: the copy is an unweighted
    # subsample of it.
    quantiles = draw_quantiles(dataset)

    # Impute just dividends on the original, full variable set on the copy

    zero_weight_copy = impute_over_incomes(
        zero_weight_copy,
        model,
        IMPUTATIONS,
        quantiles,
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

    # The copy keeps its FRS donor's employment status but now has SPI
    # incomes, so derive its gainful self-employment flag again from the
    # copy's own values.
    if "uc_is_in_gainful_self_employment" in zero_weight_copy.person.columns:
        from policyengine_uk_data.datasets.frs import (
            derive_uc_is_in_gainful_self_employment,
        )

        person = zero_weight_copy.person
        person["uc_is_in_gainful_self_employment"] = (
            derive_uc_is_in_gainful_self_employment(
                person.employment_status,
                person.self_employment_income,
                person.employment_income,
            )
        )

    dataset = impute_over_incomes(
        dataset,
        model,
        ["dividend_income"],
        quantiles,
    )

    zero_weight_copy.validate()
    dataset.validate()

    data = stack_datasets(
        dataset,
        zero_weight_copy,
    )

    return strip_internal_disability_reported_amounts(data)
