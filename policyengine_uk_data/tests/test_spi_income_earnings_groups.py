"""SPI income draws are conditioned on earnings group.

Invariants (Hypothesis properties unless noted):

1. FRS rows: children (FRS child table or under 16) are ``NOT_IMPUTED``.
   Everyone else is in exactly one earnings group, which has pay if and only
   if the main job is as an employee or the FRS records pay, and a trade if
   and only if the main job is self-employment or the FRS records a profit.
   People with both are in the employee-main group if and only if the main
   job is as an employee, or is neither and pay is at least the profit.
2. SPI records: the group has pay if and only if PAY + EPB + TAXTERM > 0, and
   a trade if and only if SEINC_NUM = 1 or PROFITS > 0. People with both are
   in the employee-main group if and only if MAINSRCE is 1 (pay), or is not 3
   or 4 (a trade) and pay is at least the profit.
3. Both mappings are monotone (more of one income adds that source and leaves
   the other alone) and give the same answer elementwise as row by row.
4. Training sample: every group with weight gets at least
   ``MIN_GROUP_SAMPLE_SHARE`` of the nominal sample size, groups without
   weight get none, and groups above the floor get their weighted share.
   Floors are not offset elsewhere, so the total lies between the nominal size
   (less rounding) and the nominal size plus one floor per group.
5. ``generate_spi_table`` resamples each group only from its own records.
6. Model draws: pay is positive exactly in the groups with pay, profit is zero
   in the groups without a trade, and ``NOT_IMPUTED`` rows get no draw.
7. ``apply_income_draws`` overwrites drawn rows (missing draws become zero),
   leaves ``NOT_IMPUTED`` rows and other columns alone, and does not mutate
   its input.
8. A cached model in another format or for another grouping is retrained.
9. On a built enhanced FRS, SPI-synthetic rows' incomes agree with their
   employment status (skipped when no build is present).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from policyengine_uk_data.datasets.imputations import income as income_module
from policyengine_uk_data.datasets.imputations.income import (
    CHILD_STATUS,
    EARNINGS_GROUPS,
    EMPLOYEE,
    EMPLOYEE_MAIN_AND_SELF_EMPLOYED,
    EMPLOYEE_STATUSES,
    IMPUTATIONS,
    MIN_GROUP_SAMPLE_SHARE,
    NO_EARNINGS,
    NOT_IMPUTED,
    PREDICTORS,
    SELF_EMPLOYED,
    SELF_EMPLOYED_MAIN_AND_EMPLOYEE,
    SELF_EMPLOYED_STATUSES,
    EarningsGroupIncomeModel,
    apply_income_draws,
    earnings_group_sample_sizes,
    frs_earnings_group,
    generate_spi_table,
    spi_earnings_group,
)

PAY_GROUPS = set(income_module.PAY_GROUPS)
TRADE_GROUPS = set(income_module.TRADE_GROUPS)
BOTH_GROUPS = {EMPLOYEE_MAIN_AND_SELF_EMPLOYED, SELF_EMPLOYED_MAIN_AND_EMPLOYEE}
MAIN_SOURCES = (-1, 1, 2, 3, 4, 5, 6)
NON_WORKING_STATUSES = (
    "UNEMPLOYED",
    "RETIRED",
    "STUDENT",
    "CARER",
    "LONG_TERM_DISABLED",
    "SHORT_TERM_DISABLED",
    "OTHER_INACTIVE",
)
STATUSES = (
    (CHILD_STATUS,) + EMPLOYEE_STATUSES + SELF_EMPLOYED_STATUSES + NON_WORKING_STATUSES
)
REGIONS = ("LONDON", "WALES", "SCOTLAND", "NORTH_EAST")

# Generation and QRF draws are slow on a loaded runner; that is not a failure.
RELAXED = settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])

amounts = st.one_of(
    st.just(0.0), st.floats(0.01, 5e6, allow_nan=False, allow_infinity=False)
)


def test_statuses_are_policyengine_uk_enum_names():
    from policyengine_uk.variables.household.income.employment_status import (
        EmploymentStatus,
    )

    names = {status.name for status in EmploymentStatus}
    # Every status the FRS build writes is covered: an employee, self-employed,
    # child or out-of-work group.
    assert set(STATUSES) == names


@st.composite
def frs_people(draw):
    n = draw(st.integers(1, 40))
    return pd.DataFrame(
        {
            "employment_status": draw(
                st.lists(st.sampled_from(STATUSES), min_size=n, max_size=n)
            ),
            "age": draw(st.lists(st.integers(0, 95), min_size=n, max_size=n)),
            "employment_income": draw(st.lists(amounts, min_size=n, max_size=n)),
            "self_employment_income": draw(st.lists(amounts, min_size=n, max_size=n)),
        }
    )


def _frs_groups(people: pd.DataFrame) -> np.ndarray:
    return frs_earnings_group(
        people.employment_status,
        people.age,
        people.employment_income,
        people.self_employment_income,
    )


@RELAXED
@given(frs_people())
def test_frs_group_follows_status_and_recorded_earnings(people):
    groups = _frs_groups(people)
    status = people.employment_status.to_numpy()
    child = (status == CHILD_STATUS) | (people.age.to_numpy() < 16)

    assert (groups[child] == NOT_IMPUTED).all()
    adult = groups[~child]
    assert np.isin(adult, EARNINGS_GROUPS).all()

    has_pay = np.isin(status, EMPLOYEE_STATUSES) | (
        people.employment_income.to_numpy() > 0
    )
    has_trade = np.isin(status, SELF_EMPLOYED_STATUSES) | (
        people.self_employment_income.to_numpy() > 0
    )
    np.testing.assert_array_equal(np.isin(adult, list(PAY_GROUPS)), has_pay[~child])
    np.testing.assert_array_equal(np.isin(adult, list(TRADE_GROUPS)), has_trade[~child])


@RELAXED
@given(frs_people(), st.floats(0.01, 1e6), st.sampled_from(["pay", "profit"]))
def test_frs_group_monotone_in_income(people, extra, source):
    before = _frs_groups(people)
    column = "employment_income" if source == "pay" else "self_employment_income"
    after = _frs_groups(people.assign(**{column: people[column] + extra}))
    imputed = before != NOT_IMPUTED
    np.testing.assert_array_equal(after == NOT_IMPUTED, ~imputed)
    gained, other = (
        (PAY_GROUPS, TRADE_GROUPS) if source == "pay" else (TRADE_GROUPS, PAY_GROUPS)
    )
    # The source that grew is now present; the other source is unchanged.
    assert np.isin(after[imputed], list(gained)).all()
    np.testing.assert_array_equal(
        np.isin(after[imputed], list(other)), np.isin(before[imputed], list(other))
    )


@RELAXED
@given(frs_people())
def test_frs_group_elementwise_equals_rowwise(people):
    vectorised = _frs_groups(people)
    rowwise = [
        frs_earnings_group(
            [row.employment_status],
            [row.age],
            [row.employment_income],
            [row.self_employment_income],
        )[0]
        for row in people.itertuples()
    ]
    as_lists = frs_earnings_group(
        people.employment_status.tolist(),
        people.age.tolist(),
        people.employment_income.tolist(),
        people.self_employment_income.tolist(),
    )
    assert list(vectorised) == rowwise == list(as_lists)


@RELAXED
@given(
    st.lists(
        st.tuples(
            amounts, amounts, st.sampled_from([-1, 0, 1]), st.sampled_from(MAIN_SOURCES)
        ),
        min_size=1,
        max_size=40,
    )
)
def test_spi_group_follows_pay_and_self_employment_pages(records):
    pay, profit, indicator, source = (np.array(column) for column in zip(*records))
    groups = spi_earnings_group(pay, profit, indicator, source)
    assert np.isin(groups, EARNINGS_GROUPS).all()
    np.testing.assert_array_equal(np.isin(groups, list(PAY_GROUPS)), pay > 0)
    np.testing.assert_array_equal(
        np.isin(groups, list(TRADE_GROUPS)), (indicator == 1) | (profit > 0)
    )
    # People with both: MAINSRCE decides, then the larger income.
    both = np.isin(groups, list(BOTH_GROUPS))
    pay_main = (source == 1) | (~np.isin(source, [3, 4]) & (pay >= profit))
    np.testing.assert_array_equal(
        groups[both] == EMPLOYEE_MAIN_AND_SELF_EMPLOYED, pay_main[both]
    )
    rowwise = [spi_earnings_group([p], [q], [i], [s])[0] for p, q, i, s in records]
    assert list(groups) == rowwise


@RELAXED
@given(frs_people())
def test_frs_both_group_follows_main_job(people):
    groups = _frs_groups(people)
    status = people.employment_status.to_numpy()
    both = np.isin(groups, list(BOTH_GROUPS))
    pay, profit = (
        people.employment_income.to_numpy(),
        people.self_employment_income.to_numpy(),
    )
    pay_main = np.isin(status, EMPLOYEE_STATUSES) | (
        ~np.isin(status, SELF_EMPLOYED_STATUSES) & (pay >= profit)
    )
    np.testing.assert_array_equal(
        groups[both] == EMPLOYEE_MAIN_AND_SELF_EMPLOYED, pay_main[both]
    )


@RELAXED
@given(
    st.lists(
        st.tuples(
            amounts, amounts, st.sampled_from([-1, 0, 1]), st.sampled_from(MAIN_SOURCES)
        ),
        min_size=1,
        max_size=40,
    ),
    st.floats(0.01, 1e6),
    st.sampled_from(["pay", "profit"]),
)
def test_spi_group_monotone_in_income(records, extra, source):
    pay, profit, indicator, main = (
        np.array(column, dtype=float) for column in zip(*records)
    )
    before = spi_earnings_group(pay, profit, indicator, main)
    if source == "pay":
        after = spi_earnings_group(pay + extra, profit, indicator, main)
        gained, other = PAY_GROUPS, TRADE_GROUPS
    else:
        after = spi_earnings_group(pay, profit + extra, indicator, main)
        gained, other = TRADE_GROUPS, PAY_GROUPS
    assert np.isin(after, list(gained)).all()
    np.testing.assert_array_equal(
        np.isin(after, list(other)), np.isin(before, list(other))
    )


@RELAXED
@given(
    st.dictionaries(
        st.sampled_from(EARNINGS_GROUPS),
        st.one_of(st.just(0.0), st.floats(1e-3, 1e8)),
        min_size=1,
    ),
    st.integers(1, 200_000),
)
def test_group_sample_sizes(weights, sample_size):
    sizes = earnings_group_sample_sizes(weights, sample_size)
    positive = {group for group, w in weights.items() if w > 0}
    assert set(sizes) == positive
    if not positive:
        return
    floor = int(np.ceil(MIN_GROUP_SAMPLE_SHARE * sample_size))
    total = sum(weights[group] for group in positive)
    for group, size in sizes.items():
        share = sample_size * weights[group] / total
        assert size >= floor
        if share > floor + 1:
            assert abs(size - share) <= 0.5 + 1e-9
    groups = len(positive)
    assert sample_size - 0.5 * groups <= sum(sizes.values())
    assert sum(sizes.values()) <= sample_size + groups * (floor + 0.5)
    assert sizes == earnings_group_sample_sizes(weights, sample_size)


def _raw_spi(rng: np.random.Generator, n: int) -> pd.DataFrame:
    """A raw SPI-shaped frame with all four earnings groups."""
    group = rng.choice(EARNINGS_GROUPS, n)
    has_pay = np.isin(group, list(PAY_GROUPS))
    has_trade = np.isin(group, list(TRADE_GROUPS))
    raw = {
        column: np.zeros(n)
        for column in set(income_module.SPI_RENAMES.values())
        | {"PAY", "EPB", "TAXTERM", "SEINC_NUM", "GIFTINV"}
    }
    raw["PAY"] = np.where(has_pay, rng.lognormal(10, 1, n), 0.0)
    raw["EPB"] = np.where(has_pay & (rng.random(n) < 0.1), 500.0, 0.0)
    # Three in ten traders make no assessable profit.
    raw["PROFITS"] = np.where(
        has_trade & (rng.random(n) < 0.7), rng.lognormal(9, 1, n), 0.0
    )
    raw["SEINC_NUM"] = has_trade.astype(int)
    # MAINSRCE agrees with the group for people with both; anything for others.
    raw["MAINSRCE"] = np.select(
        [
            group == EMPLOYEE_MAIN_AND_SELF_EMPLOYED,
            group == SELF_EMPLOYED_MAIN_AND_EMPLOYEE,
        ],
        [1, rng.choice([3, 4], n)],
        rng.choice(MAIN_SOURCES, n),
    )
    for column in ("INCBBS", "DIVIDENDS", "PENSION", "INCPROP", "GIFTAID", "GIFTINV"):
        raw[column] = rng.exponential(1_000, n) * (rng.random(n) < 0.3)
    raw["FACT"] = rng.uniform(1, 500, n)
    raw["SEX"] = rng.choice([1, 2], n)
    raw["GORCODE"] = rng.choice([1, 7, 10, 11], n)
    raw["AGERANGE"] = rng.choice([1, 2, 3, 4, 5, 6, 7], n)
    return pd.DataFrame(raw)


@settings(deadline=None, max_examples=25, suppress_health_check=[HealthCheck.too_slow])
@given(st.integers(0, 2**31 - 1), st.integers(50, 2_000))
def test_generate_spi_table_resamples_within_groups(seed, sample_size):
    raw = _raw_spi(np.random.default_rng(seed), 600)
    table = generate_spi_table(raw.copy(), seed=seed % 1000, sample_size=sample_size)
    np.testing.assert_array_equal(
        table.earnings_group.to_numpy(),
        spi_earnings_group(
            table.employment_income,
            table.self_employment_income,
            table.SEINC_NUM,
            table.MAINSRCE,
        ),
    )
    raw_groups = spi_earnings_group(
        raw.PAY + raw.EPB + raw.TAXTERM, raw.PROFITS, raw.SEINC_NUM, raw.MAINSRCE
    )
    weights = raw.FACT.groupby(raw_groups).sum().to_dict()
    assert table.earnings_group.value_counts().to_dict() == (
        earnings_group_sample_sizes(weights, sample_size)
    )


@pytest.fixture(scope="module")
def fitted_model():
    """A real per-group QRF fitted through ``generate_spi_table``."""
    from policyengine_uk_data.utils.qrf import QRF

    table = generate_spi_table(
        _raw_spi(np.random.default_rng(1), 4_000), seed=0, sample_size=2_000
    )
    models = {}
    for group in EARNINGS_GROUPS:
        training = table[table.earnings_group == group]
        model = QRF()
        model.fit(training[PREDICTORS], training[IMPUTATIONS])
        models[group] = model
    return EarningsGroupIncomeModel(models, income_module.get_income_model_metadata())


@st.composite
def model_inputs(draw):
    n = draw(st.integers(1, 30))
    return pd.DataFrame(
        {
            "age": draw(st.lists(st.floats(16, 90), min_size=n, max_size=n)),
            "gender": draw(
                st.lists(st.sampled_from(["MALE", "FEMALE"]), min_size=n, max_size=n)
            ),
            "region": draw(
                st.lists(
                    st.sampled_from(["NORTH_EAST", "LONDON", "WALES", "SCOTLAND"]),
                    min_size=n,
                    max_size=n,
                )
            ),
            "earnings_group": draw(
                st.lists(
                    st.sampled_from(EARNINGS_GROUPS + (NOT_IMPUTED,)),
                    min_size=n,
                    max_size=n,
                )
            ),
        },
        index=draw(
            st.lists(st.integers(0, 10**6), min_size=n, max_size=n, unique=True)
        ),
    )


@settings(
    deadline=None,
    max_examples=40,
    suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow],
)
@given(model_inputs())
def test_model_draws_agree_with_group(fitted_model, inputs):
    draws = fitted_model.predict(inputs)
    assert list(draws.columns) == IMPUTATIONS
    assert draws.index.equals(inputs.index)
    groups = inputs.earnings_group.to_numpy()
    drawn = groups != NOT_IMPUTED
    assert draws[~drawn].isna().all().all()
    assert draws[drawn].notna().all().all()
    pay = draws.employment_income.to_numpy()[drawn]
    profit = draws.self_employment_income.to_numpy()[drawn]
    np.testing.assert_array_equal(pay > 0, np.isin(groups[drawn], list(PAY_GROUPS)))
    assert (profit[~np.isin(groups[drawn], list(TRADE_GROUPS))] == 0).all()
    assert (draws[drawn].to_numpy() >= 0).all()


@pytest.mark.parametrize("group", income_module.TRADE_GROUPS)
def test_model_draws_some_profit_for_traders(fitted_model, group):
    """Every trade group draws zero and positive profits, as the SPI has both."""
    n = 400
    inputs = pd.DataFrame(
        {
            "age": np.linspace(20, 80, n),
            "gender": ["MALE", "FEMALE"] * (n // 2),
            "region": ["LONDON"] * n,
            "earnings_group": [group] * n,
        }
    )
    profit = fitted_model.predict(inputs).self_employment_income
    assert 0.3 < (profit > 0).mean() < 1


@st.composite
def person_and_draws(draw):
    n = draw(st.integers(1, 30))
    columns = ["employment_income", "dividend_income", "rent"]
    person = pd.DataFrame(
        {c: draw(st.lists(amounts, min_size=n, max_size=n)) for c in columns}
    )
    draws = pd.DataFrame(
        {
            c: draw(
                st.lists(st.one_of(amounts, st.just(np.nan)), min_size=n, max_size=n)
            )
            for c in IMPUTATIONS
        }
    )
    groups = draw(
        st.lists(
            st.sampled_from(EARNINGS_GROUPS + (NOT_IMPUTED,)), min_size=n, max_size=n
        )
    )
    outputs = draw(
        st.lists(st.sampled_from(IMPUTATIONS), min_size=1, max_size=4, unique=True)
    )
    return person, draws, np.array(groups, dtype=object), outputs


@RELAXED
@given(person_and_draws())
def test_apply_income_draws(case):
    person, draws, groups, outputs = case
    before = person.copy()
    result = apply_income_draws(person, draws, groups, outputs)
    pd.testing.assert_frame_equal(person, before)

    kept = groups == NOT_IMPUTED
    for column in outputs:
        own = (
            before[column].to_numpy()
            if column in before.columns
            else np.zeros(len(before))
        )
        np.testing.assert_array_equal(result[column].to_numpy()[kept], own[kept])
        np.testing.assert_array_equal(
            result[column].to_numpy()[~kept],
            np.nan_to_num(draws[column].to_numpy(dtype=float)[~kept], nan=0.0),
        )
    for column in set(before.columns) - set(outputs):
        pd.testing.assert_series_equal(result[column], before[column])


def test_old_single_model_cache_is_retrained(tmp_path, monkeypatch):
    import pickle
    from types import SimpleNamespace

    cache = tmp_path / "income_spi_2022_23.pkl"
    old_metadata = {
        key: value
        for key, value in income_module.get_income_model_metadata().items()
        if key not in ("earnings_groups", "min_group_sample_share")
    }
    with cache.open("wb") as f:
        pickle.dump(
            {
                "model": SimpleNamespace(imputed_variables=list(IMPUTATIONS)),
                "input_columns": PREDICTORS,
                "metadata": old_metadata,
            },
            f,
        )
    sentinel = object()
    monkeypatch.setattr(income_module, "INCOME_MODEL_PATH", cache)
    monkeypatch.setattr(income_module, "save_imputation_models", lambda: sentinel)
    assert income_module.create_income_model() is sentinel


def test_cache_missing_a_group_is_retrained(tmp_path, monkeypatch, fitted_model):
    cache = tmp_path / "income_spi_2022_23.pkl"
    partial = EarningsGroupIncomeModel(
        {g: m for g, m in fitted_model.models.items() if g != SELF_EMPLOYED},
        income_module.get_income_model_metadata(),
    )
    partial.save(cache)
    sentinel = object()
    monkeypatch.setattr(income_module, "INCOME_MODEL_PATH", cache)
    monkeypatch.setattr(income_module, "save_imputation_models", lambda: sentinel)
    assert income_module.create_income_model() is sentinel


def test_current_cache_round_trips(tmp_path, monkeypatch, fitted_model):
    cache = tmp_path / "income_spi_2022_23.pkl"
    fitted_model.save(cache)
    monkeypatch.setattr(income_module, "INCOME_MODEL_PATH", cache)
    monkeypatch.setattr(
        income_module,
        "save_imputation_models",
        lambda: pytest.fail("a current cache should be reused"),
    )
    loaded = income_module.create_income_model()
    inputs = pd.DataFrame(
        {
            "age": [30.0, 50.0, 70.0, 10.0],
            "gender": ["MALE", "FEMALE", "MALE", "FEMALE"],
            "region": ["LONDON", "WALES", "SCOTLAND", "LONDON"],
            "earnings_group": [EMPLOYEE, SELF_EMPLOYED, NO_EARNINGS, NOT_IMPUTED],
        }
    )
    pd.testing.assert_frame_equal(loaded.predict(inputs), fitted_model.predict(inputs))


def test_built_enhanced_frs_spi_rows_agree_with_status(enhanced_frs):
    person = enhanced_frs.person
    household = enhanced_frs.household.set_index("household_id")
    spi = person.person_household_id.map(household.household_is_spi_synthetic)
    if not spi.any():
        pytest.skip("No SPI-synthetic rows in this build")
    status = person.employment_status.astype(str)
    rows = person[spi.to_numpy(dtype=bool)]
    rows_status = status[spi.to_numpy(dtype=bool)]

    employees = rows_status.isin(EMPLOYEE_STATUSES)
    assert (rows.employment_income[employees] > 0).all()

    children = rows_status.eq(CHILD_STATUS)
    assert (rows.employment_income[children] == 0).all()
    assert (rows.self_employment_income[children] == 0).all()

    # The SPI self-employed group draws a profit for about nine in ten
    # (zero for traders who break even or make a loss); before the groups
    # it was under one in ten.
    self_employed = rows_status.isin(SELF_EMPLOYED_STATUSES)
    if self_employed.sum() >= 50:
        assert (rows.self_employment_income[self_employed] > 0).mean() > 0.6

    # Out of work: earnings only where the FRS donor recorded some, which
    # the FRS does for almost no one.
    out_of_work = rows_status.isin(NON_WORKING_STATUSES)
    if out_of_work.sum() >= 50:
        earning = (rows.employment_income[out_of_work] > 0) | (
            rows.self_employment_income[out_of_work] > 0
        )
        assert earning.mean() < 0.02
