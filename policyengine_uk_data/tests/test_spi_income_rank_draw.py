"""SPI incomes are drawn at each person's FRS rank within their cell.

Invariants (Hypothesis properties unless noted):

1. ``rank_quantiles`` returns quantiles in [0, 1). Within a cell a lower
   value never gets a higher quantile (and gets a lower one if it has
   weight), and the rows' intervals (the weight
   before a row, and that plus its own, over the cell's weight) tile [0, 1)
   with each quantile inside its row's interval: the quantiles are uniform
   within every cell, whatever the ties and weights.
2. With equal weights and m rows per grid point in a cell, every point of
   ``DRAW_QUANTILE_GRID`` is drawn by exactly m rows (no tail is cut off).
3. The quantiles do not depend on row order, on any strictly increasing
   transform of the values, on the values in other cells, or on the scale of
   a cell's weights; equal seeds give equal results; a row whose value is
   unique in its cell stays in its own interval whatever the seed. Across
   seeds, a row's quantile is uniform on its interval (not a fixed point in
   it).
4. Missing values or weights, negative weights and duplicate ids are
   rejected.
5. ``draw_at_quantiles`` reproduces microimpute 1.8's ``predict`` exactly
   at microimpute's own random draws and grid (differential test).
6. The first output drawn is non-decreasing in the quantile for a fixed
   person, and every draw lies within the range of the training data. Each
   output is drawn at its own quantile: raising one output's quantile never
   lowers it and never changes an output drawn before it. A model type the
   draw does not know is an error.
7. ``EarningsGroupIncomeModel.predict`` draws each group at the quantiles in
   the inputs, gives no draw to ``NOT_IMPUTED`` rows, and without quantiles
   is deterministic.
8. ``impute_over_incomes`` passes each person's quantiles through by person
   ID, so a subsample drawn with the full data's quantiles gets exactly
   those quantiles, and a higher FRS income never draws lower than a lower
   one in the same cell. ``draw_quantiles`` ranks within ``rank_cells`` at
   household weights: in every cell each person's quantile lies in their
   slice of the cell's household weight. Outputs everyone ties on get
   independent quantiles. (``impute_income`` uses one set of full-FRS
   quantiles for both of its draws: test_imputation_source_flags.py.)
9. On a built enhanced FRS, SPI-synthetic employees' pay rises with their
   hours as it does in the FRS (skipped when no build is present).
"""

from __future__ import annotations

import importlib.metadata

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from policyengine_uk_data.datasets.imputations import income as income_module
from policyengine_uk_data.datasets.imputations.income import (
    DRAW_QUANTILE_GRID,
    EMPLOYEE,
    EMPLOYEE_STATUSES,
    IMPUTATIONS,
    NO_EARNINGS,
    NOT_IMPUTED,
    PREDICTORS,
    SELF_EMPLOYED,
    EarningsGroupIncomeModel,
    draw_at_quantiles,
    draw_quantile_column,
    rank_cells,
    rank_quantiles,
    spi_age_band,
)

RELAXED = settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])

values_strategy = st.one_of(
    st.just(0.0),
    st.sampled_from([100.0, 5_000.0, 20_000.0]),  # heaped values, as in the FRS
    st.floats(-1e4, 1e7, allow_nan=False, allow_infinity=False),
)


@st.composite
def ranked_rows(draw, max_rows=60):
    n = draw(st.integers(1, max_rows))
    return pd.DataFrame(
        {
            "value": draw(st.lists(values_strategy, min_size=n, max_size=n)),
            "cell": draw(st.lists(st.integers(0, 4), min_size=n, max_size=n)),
            "weight": draw(
                st.lists(
                    st.one_of(st.just(0.0), st.floats(1e-3, 1e4)),
                    min_size=n,
                    max_size=n,
                )
            ),
            "id": draw(
                st.lists(st.integers(0, 10**9), min_size=n, max_size=n, unique=True)
            ),
        }
    )


def _rank(rows: pd.DataFrame, seed: int = 0) -> np.ndarray:
    return rank_quantiles(rows.value, rows.cell, rows.weight, rows.id, seed)


@RELAXED
@given(ranked_rows(), st.integers(0, 2**32 - 1))
def test_rank_quantiles_tile_each_cell(rows, seed):
    u = _rank(rows, seed)
    assert ((u >= 0) & (u < 1)).all()
    for _, cell in rows.assign(u=u).groupby("cell"):
        weight = cell.weight.to_numpy()
        if weight.sum() <= 0:
            weight = np.ones(len(cell))
        # Rows without weight can share a quantile with the next value up.
        order = np.lexsort((cell.value.to_numpy(), cell.u.to_numpy()))
        w = weight[order]
        total = w.sum()
        before = np.cumsum(w) - w
        q = cell.u.to_numpy()[order]
        assert (q >= before / total - 1e-12).all()
        assert (q <= (before + w) / total + 1e-12).all()
        # Lower values get lower quantiles.
        values = cell.value.to_numpy()[order]
        assert (np.diff(values) >= 0).all()


@RELAXED
@given(ranked_rows())
def test_rank_quantiles_increase_with_value(rows):
    u = _rank(rows)
    for _, cell in rows.assign(u=u).groupby("cell"):
        v, q = cell.value.to_numpy(), cell.u.to_numpy()
        # A cell with no weight is ranked with equal weights.
        weighted = (
            cell.weight.to_numpy() > 0
            if cell.weight.sum() > 0
            else np.ones(len(cell), bool)
        )
        lower = v[:, None] < v[None, :]
        assert (q[:, None] <= q[None, :])[lower].all()
        assert (q[:, None] < q[None, :])[lower & weighted[:, None]].all()


@RELAXED
@given(st.integers(1, 50), st.integers(0, 2**32 - 1), st.booleans())
def test_equal_weights_fill_every_grid_point(m, seed, all_tied):
    k = len(DRAW_QUANTILE_GRID)
    n = m * k
    rng = np.random.default_rng(seed)
    values = np.zeros(n) if all_tied else rng.integers(0, 50, n).astype(float)
    u = rank_quantiles(values, np.zeros(n), np.ones(n), np.arange(n), seed)
    index = np.clip((u * k).astype(int), 0, k - 1)
    assert (np.bincount(index, minlength=k) == m).all()


@RELAXED
@given(ranked_rows(), st.randoms(use_true_random=False), st.integers(0, 1000))
def test_rank_quantiles_ignore_row_order(rows, random, seed):
    order = list(range(len(rows)))
    random.shuffle(order)
    shuffled = rows.iloc[order]
    a = pd.Series(_rank(rows, seed), index=rows.id)
    b = pd.Series(_rank(shuffled, seed), index=shuffled.id)
    pd.testing.assert_series_equal(a.sort_index(), b.sort_index())


@RELAXED
@given(ranked_rows(), st.floats(1e-3, 1e3), st.floats(-1e6, 1e6))
def test_rank_quantiles_ignore_increasing_transforms(rows, scale, shift):
    transformed = rows.assign(value=rows.value * scale + shift)
    # A transform that merges distinct values (float rounding) changes ties.
    assume(
        transformed.groupby("cell")
        .value.nunique()
        .equals(rows.groupby("cell").value.nunique())
    )
    np.testing.assert_array_equal(_rank(rows), _rank(transformed))


@RELAXED
@given(ranked_rows(), st.integers(0, 4), st.data())
def test_rank_quantiles_cells_are_independent(rows, cell, data):
    changed = rows.copy()
    in_cell = changed.cell == cell
    changed.loc[in_cell, "value"] = data.draw(
        st.lists(
            values_strategy, min_size=int(in_cell.sum()), max_size=int(in_cell.sum())
        )
    )
    np.testing.assert_array_equal(_rank(rows)[~in_cell], _rank(changed)[~in_cell])


@RELAXED
@given(ranked_rows(), st.integers(0, 4), st.floats(1e-2, 1e2))
def test_rank_quantiles_ignore_weight_scale(rows, cell, factor):
    scaled = rows.copy()
    scaled.loc[scaled.cell == cell, "weight"] *= factor
    np.testing.assert_allclose(_rank(rows), _rank(scaled), rtol=1e-9, atol=1e-12)


@RELAXED
@given(ranked_rows(), st.integers(0, 2**32 - 1), st.integers(0, 2**32 - 1))
def test_rank_quantiles_seeds(rows, seed_a, seed_b):
    np.testing.assert_array_equal(_rank(rows, seed_a), _rank(rows, seed_a))
    a, b = _rank(rows, seed_a), _rank(rows, seed_b)
    weight = rows.weight.where(
        rows.groupby("cell").weight.transform("sum") > 0, 1.0
    ).to_numpy()
    total = pd.Series(weight).groupby(rows.cell.to_numpy()).transform("sum").to_numpy()
    unique_in_cell = ~rows.duplicated(["cell", "value"], keep=False).to_numpy()
    # A row with a value of its own keeps its interval of width weight/total.
    width = weight / total
    assert (np.abs(a - b)[unique_in_cell] <= width[unique_in_cell] + 1e-12).all()


def test_rank_quantiles_are_uniform_within_each_slice():
    """Across seeds a row takes a uniform point in its slice, not a fixed
    one: a lone donor does not always draw its cell's median."""
    from scipy.stats import kstest

    seeds = range(2_000)
    lone = [rank_quantiles([5.0], [0], [1.0], [7], seed)[0] for seed in seeds]
    assert kstest(lone, "uniform").pvalue > 1e-3
    pair = np.array(
        [rank_quantiles([1.0, 2.0], [0, 0], [1.0, 3.0], [1, 2], seed) for seed in seeds]
    )
    assert kstest(pair[:, 0], "uniform", args=(0, 0.25)).pvalue > 1e-3
    assert kstest(pair[:, 1], "uniform", args=(0.25, 0.75)).pvalue > 1e-3


@pytest.mark.parametrize(
    "values, weights, ids",
    [
        ([1.0, np.nan], [1.0, 1.0], [1, 2]),
        ([1.0, 2.0], [1.0, np.nan], [1, 2]),
        ([1.0, 2.0], [1.0, -1.0], [1, 2]),
        ([1.0, 2.0], [1.0, 1.0], [1, 1]),
    ],
)
def test_rank_quantiles_reject_bad_inputs(values, weights, ids):
    with pytest.raises(ValueError):
        rank_quantiles(values, [0, 0], weights, ids)


def test_spi_age_bands_match_spi_codes():
    from policyengine_uk_data.datasets.spi import AGE_RANGES

    for code, (low, high) in AGE_RANGES.items():
        if code > 0:
            assert (spi_age_band([low, (low + high) / 2, high - 0.01]) == code).all()
    assert spi_age_band([0, 15.9])[0] == 0 and spi_age_band([95])[0] == 7


def test_rank_cells_separate_every_key():
    inputs = pd.DataFrame(
        {
            "earnings_group": [EMPLOYEE, EMPLOYEE, SELF_EMPLOYED, EMPLOYEE, EMPLOYEE],
            "age": [30, 31, 30, 50, 30],
            "gender": ["MALE", "MALE", "MALE", "MALE", "FEMALE"],
            "region": ["LONDON"] * 5,
        }
    )
    cells = rank_cells(inputs)
    assert cells[0] == cells[1]
    assert len(set(cells[[0, 2, 3, 4]])) == 4


# ---- the draw ---------------------------------------------------------------

OUTPUTS = ["employment_income", "dividend_income", "gift_aid"]


@pytest.fixture(scope="module")
def fitted_results():
    """A microimpute QRF on synthetic, SPI-like data with zeros and a tail."""
    from policyengine_uk_data.utils.qrf import QRF

    rng = np.random.default_rng(3)
    n = 600
    X = pd.DataFrame(
        {
            "age": rng.uniform(16, 90, n),
            "gender": rng.choice(["MALE", "FEMALE"], n),
            "region": rng.choice(["LONDON", "WALES", "SCOTLAND"], n),
        }
    )
    pay = rng.lognormal(10, 0.8, n) * np.where(X.region == "LONDON", 1.5, 1.0)
    y = pd.DataFrame(
        {
            "employment_income": pay,
            "dividend_income": (rng.random(n) < 0.3) * rng.exponential(pay / 10),
            "gift_aid": (rng.random(n) < 0.1) * rng.exponential(200, n),
        }
    )
    model = QRF()
    model.fit(X, y)
    return model.model, X, y


people = st.builds(
    lambda age, gender, region: (age, gender, region),
    st.floats(16, 90),
    st.sampled_from(["MALE", "FEMALE"]),
    st.sampled_from(["LONDON", "WALES", "SCOTLAND"]),
)


def _frame(rows):
    return pd.DataFrame(rows, columns=PREDICTORS)


@pytest.mark.skipif(
    importlib.metadata.version("microimpute").split(".")[0] != "1",
    reason="microimpute 2+ draws on another grid; the locked version is 1.x",
)
@settings(
    deadline=None,
    max_examples=30,
    suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow],
)
@given(st.lists(people, min_size=1, max_size=40))
def test_draw_reproduces_microimpute_predict(fitted_results, rows):
    results, _, _ = fitted_results
    X = _frame(rows)
    expected = results.predict(X)
    u = np.random.default_rng(results.seed).beta(1, 1, size=len(X))
    grid = np.linspace(1 / 11, 1 - 1 / 11, 10)
    drawn = draw_at_quantiles(results, X, {v: u for v in OUTPUTS}, grid)
    for v in OUTPUTS:
        np.testing.assert_array_equal(drawn[v].to_numpy(), expected[v].to_numpy())


@settings(
    deadline=None,
    max_examples=30,
    suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow],
)
@given(people, st.lists(st.floats(0, 1, exclude_max=True), min_size=2, max_size=12))
def test_first_output_rises_with_the_quantile(fitted_results, person, us):
    results, _, y = fitted_results
    us = np.sort(np.array(us))
    X = _frame([person] * len(us))
    drawn = draw_at_quantiles(results, X, {v: us for v in OUTPUTS}, DRAW_QUANTILE_GRID)
    assert (np.diff(drawn[OUTPUTS[0]].to_numpy()) >= 0).all()
    for v in OUTPUTS:
        assert (drawn[v] >= y[v].min() - 1e-9).all()
        assert (drawn[v] <= y[v].max() + 1e-9).all()


def test_draw_reaches_the_tails(fitted_results):
    """The grid's ends are the forest's 0.05th and 99.95th percentiles."""
    results, X, _ = fitted_results
    row = X.iloc[[0, 0]].reset_index(drop=True)
    u = np.array([0.0, np.nextafter(1.0, 0.0)])
    drawn = draw_at_quantiles(results, row, {v: u for v in OUTPUTS}, DRAW_QUANTILE_GRID)
    model = results.models[OUTPUTS[0]]
    features, _ = results.preprocess_data_types(
        row, results.original_predictors, getattr(results, "dummy_processor", None)
    )
    columns = results._get_encoded_predictors(results.predictors)
    ends = np.asarray(
        model.qrf.predict(
            features[columns], quantiles=[DRAW_QUANTILE_GRID[0], DRAW_QUANTILE_GRID[-1]]
        )
    )[0]
    np.testing.assert_array_equal(drawn[OUTPUTS[0]].to_numpy(), ends)


@pytest.mark.parametrize("j", range(len(OUTPUTS)))
def test_each_output_is_drawn_at_its_own_quantile(fitted_results, j):
    """Raising one output's quantile raises that output and leaves every
    output drawn before it unchanged."""
    results, X, _ = fitted_results
    X = X.iloc[:50].reset_index(drop=True)
    low = {v: np.zeros(len(X)) for v in OUTPUTS}
    high = {**low, OUTPUTS[j]: np.full(len(X), np.nextafter(1.0, 0.0))}
    a = draw_at_quantiles(results, X, low, DRAW_QUANTILE_GRID)
    b = draw_at_quantiles(results, X, high, DRAW_QUANTILE_GRID)
    for v in OUTPUTS[:j]:
        np.testing.assert_array_equal(a[v].to_numpy(), b[v].to_numpy())
    assert (b[OUTPUTS[j]] >= a[OUTPUTS[j]]).all()
    assert (b[OUTPUTS[j]] > a[OUTPUTS[j]]).mean() > 0.5


def test_draw_rejects_unknown_model_types(fitted_results):
    import copy

    results, X, _ = fitted_results
    results = copy.copy(results)
    results.models = {**results.models, OUTPUTS[1]: object()}
    u = {v: np.zeros(3) for v in OUTPUTS}
    with pytest.raises(TypeError):
        draw_at_quantiles(results, X.iloc[:3], u, DRAW_QUANTILE_GRID)


# ---- the group model and the wiring ------------------------------------------


@pytest.fixture(scope="module")
def group_model(fitted_results):
    from policyengine_uk_data.utils.qrf import QRF

    results, X, y = fitted_results
    full = y.reindex(columns=IMPUTATIONS, fill_value=0.0)
    models = {}
    for group in (EMPLOYEE, NO_EARNINGS):
        model = QRF()
        model.fit(X, full)
        models[group] = model
    return EarningsGroupIncomeModel(models, income_module.get_income_model_metadata())


@settings(
    deadline=None,
    max_examples=25,
    suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow],
)
@given(
    st.lists(
        st.tuples(people, st.sampled_from([EMPLOYEE, NO_EARNINGS, NOT_IMPUTED])),
        min_size=1,
        max_size=30,
    ),
    st.integers(0, 2**32 - 1),
)
def test_group_model_draws_at_the_given_quantiles(group_model, rows, seed):
    X = _frame([r[0] for r in rows])
    X["earnings_group"] = [r[1] for r in rows]
    rng = np.random.default_rng(seed)
    for v in IMPUTATIONS:
        X[draw_quantile_column(v)] = rng.random(len(X))
    drawn = group_model.predict(X)
    groups = X.earnings_group.to_numpy()
    assert drawn[groups == NOT_IMPUTED].isna().all().all()
    for group, model in group_model.models.items():
        m = groups == group
        if m.any():
            expected = draw_at_quantiles(
                model.model,
                X.loc[m, PREDICTORS].reset_index(drop=True),
                {v: X[draw_quantile_column(v)].to_numpy()[m] for v in IMPUTATIONS},
                DRAW_QUANTILE_GRID,
            )
            np.testing.assert_array_equal(
                drawn.loc[m, IMPUTATIONS].to_numpy(), expected[IMPUTATIONS].to_numpy()
            )
    # Without quantiles the draw is random but reproducible.
    bare = X[PREDICTORS + ["earnings_group"]]
    pd.testing.assert_frame_equal(group_model.predict(bare), group_model.predict(bare))


class _EchoModel:
    """Draws each output as the person's quantile for it."""

    def predict(self, X):
        drawn = pd.DataFrame(
            {v: X[draw_quantile_column(v)].to_numpy() for v in IMPUTATIONS},
            index=X.index,
        )
        drawn[X.earnings_group.to_numpy() == NOT_IMPUTED] = np.nan
        return drawn


class _Dataset:
    def __init__(self, person, household):
        self.person, self.household = person, household

    def copy(self):
        return _Dataset(self.person.copy(), self.household.copy())


@st.composite
def frs_datasets(draw):
    n = draw(st.integers(2, 40))
    households = draw(st.integers(1, n))
    person = pd.DataFrame(
        {
            "person_id": np.arange(n) * 7 + 3,
            "person_household_id": draw(
                st.lists(st.integers(0, households - 1), min_size=n, max_size=n)
            ),
            "age": draw(st.lists(st.integers(10, 80), min_size=n, max_size=n)),
            "gender": draw(
                st.lists(st.sampled_from(["MALE", "FEMALE"]), min_size=n, max_size=n)
            ),
            "employment_status": draw(
                st.lists(
                    st.sampled_from(["FT_EMPLOYED", "PT_EMPLOYED", "RETIRED"]),
                    min_size=n,
                    max_size=n,
                )
            ),
        }
    )
    for v in IMPUTATIONS[:6]:
        person[v] = draw(
            st.lists(st.one_of(st.just(0.0), st.floats(1, 1e6)), min_size=n, max_size=n)
        )
    household = pd.DataFrame(
        {
            "household_id": np.arange(households),
            "household_weight": draw(
                st.lists(st.floats(1, 3_000), min_size=households, max_size=households)
            ),
            "region": draw(
                st.lists(
                    st.sampled_from(["LONDON", "WALES"]),
                    min_size=households,
                    max_size=households,
                )
            ),
        }
    )
    return _Dataset(person, household)


def _inputs_from_columns(dataset):
    person = dataset.person
    region = person.person_household_id.map(
        dataset.household.set_index("household_id").region
    )
    inputs = pd.DataFrame(
        {
            "age": person.age.to_numpy(),
            "gender": person.gender.to_numpy(),
            "region": region.to_numpy(),
        }
    )
    inputs["earnings_group"] = income_module.frs_earnings_group(
        person.employment_status,
        inputs.age,
        person.employment_income,
        person.self_employment_income,
    )
    return inputs


@settings(deadline=None, max_examples=40, suppress_health_check=[HealthCheck.too_slow])
@given(frs_datasets(), st.data())
def test_impute_over_incomes_passes_quantiles_by_person(dataset, data):
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(income_module, "income_model_inputs", _inputs_from_columns)
        quantiles = income_module.draw_quantiles(dataset)
        keep = data.draw(
            st.lists(
                st.sampled_from(list(dataset.household.household_id)),
                min_size=1,
                unique=True,
            )
        )
        sub = _Dataset(
            dataset.person[dataset.person.person_household_id.isin(keep)].reset_index(
                drop=True
            ),
            dataset.household[dataset.household.household_id.isin(keep)].reset_index(
                drop=True
            ),
        )
        result = income_module.impute_over_incomes(
            sub, _EchoModel(), IMPUTATIONS, quantiles
        )
    groups = _inputs_from_columns(sub).earnings_group.to_numpy()
    drawn = groups != NOT_IMPUTED
    expected = quantiles.loc[sub.person.person_id.to_numpy()]
    for v in IMPUTATIONS:
        got = result.person[v].to_numpy()
        np.testing.assert_array_equal(
            got[drawn], expected[draw_quantile_column(v)].to_numpy()[drawn]
        )
        if v in sub.person.columns:
            np.testing.assert_array_equal(got[~drawn], sub.person[v].to_numpy()[~drawn])
    # Within a cell, more FRS pay never draws lower.
    inputs = _inputs_from_columns(dataset)
    cells = rank_cells(inputs)
    pay = dataset.person.employment_income.to_numpy()
    u = quantiles[draw_quantile_column("employment_income")].to_numpy()
    same = cells[:, None] == cells[None, :]
    assert (u[:, None] < u[None, :])[same & (pay[:, None] < pay[None, :])].all()


@settings(deadline=None, max_examples=60, suppress_health_check=[HealthCheck.too_slow])
@given(frs_datasets())
def test_draw_quantiles_tile_each_cell_at_household_weights(dataset):
    """Ranks are taken within ``rank_cells`` at household weights, so in
    every cell each person's quantile lies in their slice of the cell's
    household weight, for every output."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(income_module, "income_model_inputs", _inputs_from_columns)
        quantiles = income_module.draw_quantiles(dataset)
    cells = rank_cells(_inputs_from_columns(dataset))
    weight = dataset.person.person_household_id.map(
        dataset.household.set_index("household_id").household_weight
    ).to_numpy()
    for v in IMPUTATIONS:
        q = quantiles[draw_quantile_column(v)].to_numpy()
        for cell in np.unique(cells):
            order = np.argsort(q[cells == cell], kind="stable")
            w = weight[cells == cell][order]
            before = np.cumsum(w) - w
            u = q[cells == cell][order]
            assert (u >= before / w.sum() - 1e-12).all()
            assert (u <= (before + w) / w.sum() + 1e-12).all()


def test_tied_outputs_get_independent_quantiles():
    """Outputs everyone ties on (gift aid, which the FRS does not record)
    are ranked in independent random orders, so they do not share one
    quantile per person, as microimpute 1.8's draw did."""
    n = 2_000
    person = pd.DataFrame(
        {
            "person_id": np.arange(n),
            "person_household_id": np.arange(n),
            "age": 70,
            "gender": "FEMALE",
            "employment_status": "RETIRED",
        }
    )
    for v in IMPUTATIONS[:6]:
        person[v] = 0.0
    household = pd.DataFrame(
        {"household_id": np.arange(n), "household_weight": 1.0, "region": "WALES"}
    )
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(income_module, "income_model_inputs", _inputs_from_columns)
        quantiles = income_module.draw_quantiles(_Dataset(person, household))
    rho = quantiles.corr(method="spearman").to_numpy()
    assert np.abs(rho[~np.eye(len(rho), dtype=bool)]).max() < 0.1


def test_built_enhanced_frs_spi_pay_rises_with_hours(enhanced_frs):
    from scipy.stats import spearmanr

    person = enhanced_frs.person
    household = enhanced_frs.household.set_index("household_id")
    spi = person.person_household_id.map(household.household_is_spi_synthetic).to_numpy(
        dtype=bool
    )
    if not spi.any():
        pytest.skip("No SPI-synthetic rows in this build")
    status = person.employment_status.astype(str).to_numpy()
    working = (
        np.isin(status, EMPLOYEE_STATUSES)
        & (person.employment_income > 0).to_numpy()
        & (person.hours_worked > 0).to_numpy()
    )
    frs_rho = spearmanr(
        person.hours_worked[working & ~spi], person.employment_income[working & ~spi]
    )[0]
    spi_rho = spearmanr(
        person.hours_worked[working & spi], person.employment_income[working & spi]
    )[0]
    # Random draws gave SPI rows almost no link between hours and pay.
    assert spi_rho > 0.5 * frs_rho
