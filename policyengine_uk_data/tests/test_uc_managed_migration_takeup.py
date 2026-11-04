"""Tests for the Move to Universal Credit claim flag.

``would_claim_uc_at_legacy_closure`` says whether a benefit unit claims
Universal Credit once a legacy benefit it reports closes (policyengine-uk
``legacy_benefits_closed``). Invariants:

1. Source: every combination is a "+"-join of LEGACY_BENEFITS in order, each
   rate is claimed / (claimed + did_not_claim) and lies in (0, 1), and the
   combinations add up to DWP's all-households row.
2. Every benefit unit reporting Universal Credit is True.
3. Every benefit unit reporting no legacy benefit is True, the model's
   default, so the column changes nothing outside the legacy cohorts.
4. A legacy reporter not on Universal Credit is True if it would claim
   anyway (would_claim_uc), and otherwise exactly when its draw is below
   (r - p) / (1 - p), where r is its combination's rate ("all" for
   combinations DWP does not report) and p its combination's share claiming
   anyway.
5. Each cohort's share claiming at the closure (would_claim_uc or the flag)
   lands within sampling error of its rate.
6. The draw is deterministic, has its own generator and leaves NumPy's global
   random state alone.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import yaml
from hypothesis import given, settings
from hypothesis import strategies as st

from policyengine_uk_data.parameters import (
    PARAMETERS_DIR,
    load_uc_managed_migration_claim_rates,
)
from policyengine_uk_data.utils.takeup import (
    LEGACY_BENEFITS,
    UC_MANAGED_MIGRATION_SEED,
    UC_MANAGED_MIGRATION_YEARS,
    assign_uc_claim_at_legacy_closure,
    legacy_benefit_combination,
)

RATES = load_uc_managed_migration_claim_rates()
REPORTED = [f"{benefit}_reported" for benefit in LEGACY_BENEFITS] + [
    "universal_credit_reported"
]


def frames(units: list[list[dict]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Person and benefit unit tables from a list of units of people."""
    rows = [
        {"person_benunit_id": 10 + i, **{c: person.get(c, 0.0) for c in REPORTED}}
        for i, people in enumerate(units)
        for person in people
    ]
    person = pd.DataFrame(rows, columns=["person_benunit_id", *REPORTED])
    # Benefit unit ids out of order, to catch positional mix-ups.
    benunit = pd.DataFrame({"benunit_id": [10 + i for i in range(len(units))]})
    return person, benunit.iloc[::-1].reset_index(drop=True)


people = st.lists(
    st.fixed_dictionaries(
        {c: st.sampled_from([0.0, 0.0, 0.0, 50.0, 3_000.0]) for c in REPORTED}
    ),
    min_size=1,
    max_size=3,
)


def test_source_combinations_and_rates():
    with open(PARAMETERS_DIR / "take_up" / "uc_managed_migration.yaml") as f:
        households = yaml.safe_load(f)["households"]
    assert set(households) == set(RATES)
    totals = {"claimed": 0, "did_not_claim": 0}
    for combination, counts in households.items():
        rate = counts["claimed"] / (counts["claimed"] + counts["did_not_claim"])
        assert RATES[combination] == rate and 0 < rate < 1
        if combination == "all":
            continue
        benefits = combination.split("+")
        assert benefits == [b for b in LEGACY_BENEFITS if b in benefits]
        for key in totals:
            totals[key] += counts[key]
    # Stat-Xplore applies disclosure control, so rows need not add exactly.
    for key, total in totals.items():
        assert abs(total - households["all"][key]) <= 20


def test_published_rates():
    # Spot checks against Stat-Xplore table MtUC Households 3.
    assert RATES["housing_benefit"] == pytest.approx(30_900 / 41_021)
    assert RATES["housing_benefit+esa_income"] == pytest.approx(448_253 / 457_370)
    assert RATES["working_tax_credit"] == pytest.approx(53_993 / 105_798)
    assert RATES["all"] == pytest.approx(0.8674, abs=1e-4)


@settings(max_examples=200, deadline=None, derandomize=True)
@given(st.lists(people, min_size=1, max_size=40), st.integers(0, 2**32 - 1))
def test_anchors_and_draws(units, seed):
    person, benunit = frames(units)
    flags = assign_uc_claim_at_legacy_closure(person, benunit, RATES, seed=seed)
    combination = legacy_benefit_combination(person, benunit)
    draws = np.random.default_rng(seed).random(len(benunit))
    assert flags.dtype == bool and len(flags) == len(benunit)
    for i, unit_id in enumerate(benunit["benunit_id"]):
        members = units[unit_id - 10]
        reports = {c for c in REPORTED if any(p[c] > 0 for p in members)}
        expected_combination = "+".join(
            b for b in LEGACY_BENEFITS if f"{b}_reported" in reports
        )
        assert combination[i] == expected_combination
        if "universal_credit_reported" in reports or not expected_combination:
            assert flags[i]
        else:
            rate = RATES.get(expected_combination, RATES["all"])
            assert flags[i] == (draws[i] < rate)


@pytest.mark.parametrize("anyway_share", [0.0, 0.3])
@pytest.mark.parametrize("combination", sorted(RATES))
def test_cohort_claim_shares_within_sampling_error(combination, anyway_share):
    """The share claiming at the closure (would_claim_uc or the flag) is
    DWP's rate, whatever share would claim anyway."""
    benefits = [] if combination == "all" else combination.split("+")
    if combination == "all":
        # A combination DWP does not report (income-related ESA with Income
        # Support) takes the all-households rate.
        benefits = ["esa_income", "income_support"]
    n = 20_000
    units = [[{f"{b}_reported": 100.0 for b in benefits}] for _ in range(n)]
    person, benunit = frames(units)
    anyway = np.random.default_rng(5).random(n) < anyway_share
    flags = assign_uc_claim_at_legacy_closure(
        person, benunit, RATES, seed=UC_MANAGED_MIGRATION_SEED, would_claim_uc=anyway
    )
    assert flags[anyway].all()
    rate = RATES[combination]
    claims = anyway | flags
    assert abs(claims.mean() - rate) <= 4 * np.sqrt(rate * (1 - rate) / n)


@settings(max_examples=100, deadline=None, derandomize=True)
@given(
    st.lists(people, min_size=1, max_size=40),
    st.integers(0, 2**32 - 1),
    st.data(),
)
def test_conditional_draw(units, seed, data):
    """Exact characterisation with would_claim_uc: units claiming anyway are
    True, and the rest are True exactly when their draw is below
    (rate - p) / (1 - p) for their combination."""
    person, benunit = frames(units)
    anyway = np.array(
        data.draw(st.lists(st.booleans(), min_size=len(units), max_size=len(units)))
    )
    flags = assign_uc_claim_at_legacy_closure(
        person, benunit, RATES, seed=seed, would_claim_uc=anyway
    )
    combination = legacy_benefit_combination(person, benunit)
    on_uc = (
        benunit["benunit_id"]
        .isin(person.loc[person["universal_credit_reported"] > 0, "person_benunit_id"])
        .values
    )
    draws = np.random.default_rng(seed).random(len(benunit))
    drawn = ~on_uc & (combination != "")
    assert flags[~drawn | anyway].all()
    for c in set(combination[drawn]):
        cohort = drawn & (combination == c)
        p = anyway[cohort].mean()
        rate = RATES.get(c, RATES["all"])
        q = 0.0 if p >= 1 else min(max((rate - p) / (1 - p), 0.0), 1.0)
        free = cohort & ~anyway
        assert (flags[free] == (draws[free] < q)).all(), c


def test_deterministic_and_isolated():
    units = [[{"housing_benefit_reported": 1.0}] for _ in range(500)]
    person, benunit = frames(units)
    np.random.seed(7)
    before = np.random.get_state()[1].copy()
    first = assign_uc_claim_at_legacy_closure(person, benunit, RATES, seed=1)
    after = np.random.get_state()[1]
    assert (before == after).all()
    assert (first == assign_uc_claim_at_legacy_closure(person, benunit, RATES, 1)).all()
    assert (first != assign_uc_claim_at_legacy_closure(person, benunit, RATES, 2)).any()


def _built_flags(dataset):
    benunit = dataset.benunit
    if "would_claim_uc_at_legacy_closure" not in benunit.columns:
        pytest.skip("Dataset predates the Move to Universal Credit claim flag")
    combination = legacy_benefit_combination(dataset.person, benunit)
    on_uc = (
        benunit["benunit_id"]
        .isin(
            dataset.person.loc[
                dataset.person["universal_credit_reported"] > 0, "person_benunit_id"
            ]
        )
        .values
    )
    return benunit["would_claim_uc_at_legacy_closure"].values, combination, on_uc


@pytest.mark.parametrize("fixture", ["frs", "enhanced_frs"])
def test_built_dataset_anchors(fixture, request):
    flags, combination, on_uc = _built_flags(request.getfixturevalue(fixture))
    assert flags[on_uc].all()
    assert flags[combination == ""].all()


def test_built_frs_cohort_shares(frs):
    # The FRS build, before SPI rows and geography clones, has one row per
    # surveyed benefit unit, so its draws are independent. Among units the
    # model moves to Universal Credit, by first year moved, and not claiming
    # anyway, the flag's share is the conditional rate.
    from policyengine_uk import Microsimulation

    from policyengine_uk_data.datasets.frs import first_year_moved_to_uc

    flags, combination, on_uc = _built_flags(frs)
    anyway = frs.benunit["would_claim_uc"].values.astype(bool)
    first_year = first_year_moved_to_uc(
        Microsimulation(dataset=frs),
        frs.benunit,
        frs.time_period,
        UC_MANAGED_MIGRATION_YEARS,
    )
    drawn = ~on_uc & (combination != "") & (first_year > 0)
    for c, year in set(zip(combination[drawn], first_year[drawn])):
        cohort = drawn & (combination == c) & (first_year == year)
        p = anyway[cohort].mean()
        rate = RATES.get(c, RATES["all"])
        q = 0.0 if p >= 1 else min(max((rate - p) / (1 - p), 0.0), 1.0)
        free = cohort & ~anyway
        m = free.sum()
        if m >= 30:
            share = flags[free].mean()
            assert abs(share - q) <= 4 * np.sqrt(max(q * (1 - q), 1e-9) / m) + 1e-9, c


def test_closure_draw_leaves_the_seed_100_sequence_alone(tmp_path, monkeypatch):
    """create_frs's closure draw has its own generator, so the seed=100 draws
    after it are pinned. These are the synthetic one-parent fixture's values
    on main; any draw that consumes the shared generator before them, at the
    call site or inside, moves them."""
    from policyengine_uk_data.tests.test_legacy_benefit_proxies import (
        create_single_adult_frs,
    )

    dataset = create_single_adult_frs(tmp_path, monkeypatch, with_child=True)
    np.testing.assert_allclose(
        dataset.person["higher_earner_tie_break"],
        [0.5352481053616507, 0.9957771042185515],
    )
    np.testing.assert_allclose(
        dataset.person["attends_private_school_random_draw"],
        [0.501946084696926, 0.7710225764025075],
    )


class _Dataset:
    def __init__(self, person, benunit, household):
        self.person, self.benunit, self.household = person, benunit, household
        self.time_period = 2024

    def copy(self):
        return _Dataset(self.person.copy(), self.benunit.copy(), self.household.copy())

    def validate(self):
        return None


def test_spi_rows_redraw_the_closure_flag_from_their_own_receipt(monkeypatch):
    """impute_income redraws the flag on the SPI-synthetic copy after stage 2
    rewrites its receipt; the FRS rows keep create_frs's draw."""
    from policyengine_uk_data.datasets import disability_benefits
    from policyengine_uk_data.datasets.imputations import frs_only
    from policyengine_uk_data.datasets.imputations import income as income_module

    n = 200
    person = pd.DataFrame(
        {
            "person_id": range(1, n + 1),
            "person_household_id": range(1, n + 1),
            "person_benunit_id": range(1, n + 1),
            **{c: 0.0 for c in REPORTED},
            "housing_benefit_reported": 1_000.0,
        }
    )
    for column in [
        "employment_income",
        "self_employment_income",
        "savings_interest_income",
        "dividend_income",
        "private_pension_income",
        "property_income",
    ]:
        person[column] = 0.0
    benunit = pd.DataFrame(
        {"benunit_id": range(1, n + 1), "would_claim_uc_at_legacy_closure": False}
    )
    household = pd.DataFrame(
        {"household_id": range(1, n + 1), "household_weight": 1.0, "region": "LONDON"}
    )

    def stage_2(train_dataset, target_dataset):
        # As main's SPI-donor rules do, the copy ends up reporting no legacy
        # benefit.
        target_dataset = target_dataset.copy()
        target_dataset.person["housing_benefit_reported"] = 0.0
        return target_dataset

    def stack(left, right):
        return _Dataset(
            pd.concat([left.person, right.person], ignore_index=True),
            pd.concat([left.benunit, right.benunit], ignore_index=True),
            pd.concat([left.household, right.household], ignore_index=True),
        )

    monkeypatch.setattr(income_module, "create_income_model", lambda: object())
    monkeypatch.setattr(
        income_module, "subsample_dataset", lambda dataset, _size: dataset.copy()
    )
    monkeypatch.setattr(
        income_module, "impute_over_incomes", lambda dataset, _model, _vars: dataset
    )
    monkeypatch.setattr(frs_only, "impute_frs_only_variables", stage_2)
    monkeypatch.setattr(
        disability_benefits,
        "strip_internal_disability_reported_amounts",
        lambda dataset: dataset,
    )
    monkeypatch.setattr(income_module, "stack_datasets", stack)

    result = income_module.impute_income(_Dataset(person, benunit, household))
    flags = result.benunit["would_claim_uc_at_legacy_closure"].values
    # FRS rows keep their flag; SPI rows report no legacy benefit, so True.
    assert not flags[:n].any()
    assert flags[n:].all()


def test_units_not_moved_are_neither_drawn_nor_counted():
    """Only units the model moves to Universal Credit set and take the
    conditional rate. Pension Credit route families reporting the same
    benefit, none of whom would claim anyway, stay True (the default) and do
    not pull the moved units' claim share above DWP's rate."""
    n = 20_000
    units = [[{"child_tax_credit_reported": 100.0}] for _ in range(2 * n)]
    person, benunit = frames(units)
    moves = (benunit["benunit_id"] < 10 + n).values
    anyway = moves & (np.random.default_rng(5).random(2 * n) < 0.3)
    flags = assign_uc_claim_at_legacy_closure(
        person,
        benunit,
        RATES,
        seed=UC_MANAGED_MIGRATION_SEED,
        would_claim_uc=anyway,
        first_move_year=np.where(moves, 2025, 0),
    )
    assert flags[~moves].all()
    rate = RATES["child_tax_credit"]
    claims = (anyway | flags)[moves]
    assert abs(claims.mean() - rate) <= 4 * np.sqrt(rate * (1 - rate) / n)


@pytest.mark.parametrize("seed", range(50))
def test_universal_credit_reporters_do_not_set_the_rate(seed):
    """A unit already reporting Universal Credit is True and outside ``p``,
    so the other units in its combination draw at DWP's rate itself."""
    units = [[{"child_tax_credit_reported": 100.0}] for _ in range(2)] + [
        [{"child_tax_credit_reported": 100.0, "universal_credit_reported": 100.0}]
    ]
    person, benunit = frames(units)
    on_uc = (benunit["benunit_id"] == 12).values
    flags = assign_uc_claim_at_legacy_closure(
        person, benunit, RATES, seed=seed, would_claim_uc=on_uc
    )
    draws = np.random.default_rng(seed).random(len(benunit))
    assert flags[on_uc].all()
    rate = RATES["child_tax_credit"]
    assert (flags[~on_uc] == (draws[~on_uc] < rate)).all()


def test_moves_to_uc_follows_the_closure_rule():
    """Who policyengine-uk moves to Universal Credit, case by case."""
    from policyengine_uk_data.datasets.frs import (
        derive_moves_to_uc_at_legacy_closure,
    )

    cases = [
        # (claimant or partner reports, Pension Credit age conditions,
        #  mixed-age couple, moved)
        ({"child_tax_credit"}, False, False, True),
        ({"child_tax_credit"}, True, False, False),
        ({"child_tax_credit", "pension_credit"}, False, False, False),
        ({"working_tax_credit"}, True, False, True),
        ({"child_tax_credit"}, True, True, False),
        ({"income_support"}, True, True, True),
        ({"income_support"}, True, False, False),
        ({"jsa_income", "pension_credit"}, True, True, False),
        ({"esa_income"}, False, False, True),
        ({"housing_benefit"}, False, False, True),
        # A mixed-age couple without the 2019 saving loses its award.
        ({"housing_benefit"}, False, True, True),
        ({"housing_benefit"}, True, True, False),
        ({"housing_benefit"}, True, False, False),
        (set(), False, False, False),
    ]
    benefits = (*LEGACY_BENEFITS, "pension_credit")
    moved = derive_moves_to_uc_at_legacy_closure(
        reports={b: np.array([b in case[0] for case in cases]) for b in benefits},
        pension_age_route=np.array([case[1] for case in cases]),
        mixed_age_couple=np.array([case[2] for case in cases]),
    )
    assert moved.tolist() == [case[3] for case in cases]


def test_each_first_year_group_claims_at_the_rate():
    """Units first moving in a later year are drawn on their own, so the
    movers of every year claim at DWP's rate: here half of the 2025 movers
    claim anyway and none of the 2027 movers do."""
    n = 20_000
    units = [[{"child_tax_credit_reported": 100.0}] for _ in range(2 * n)]
    person, benunit = frames(units)
    early = (benunit["benunit_id"] < 10 + n).values
    anyway = early & (np.random.default_rng(5).random(2 * n) < 0.5)
    flags = assign_uc_claim_at_legacy_closure(
        person,
        benunit,
        RATES,
        seed=UC_MANAGED_MIGRATION_SEED,
        would_claim_uc=anyway,
        first_move_year=np.where(early, 2025, 2027),
    )
    rate = RATES["child_tax_credit"]
    claims = anyway | flags
    for movers in (early, np.ones(2 * n, dtype=bool)):
        share = claims[movers].mean()
        assert abs(share - rate) <= 4 * np.sqrt(rate * (1 - rate) / movers.sum())


class _YearlySimulation:
    """A simulation whose Pension Credit route depends on the year: the
    couple meets the age conditions only in 2024, as a mixed-age couple
    whose older member is born too late for the saving from 2025."""

    class tax_benefit_system:
        variables = {"is_claimant_or_partner": None}

    def calculate(self, variable, year):
        values = {
            "person_benunit_id": [10, 10, 10],
            "is_claimant_or_partner": [True, True, False],
            "child_tax_credit_reported": [100.0, 0.0, 0.0],
            "housing_benefit_reported": [100.0, 0.0, 0.0],
            "benunit_id": [10],
            "meets_pension_credit_age_conditions": [year < 2025],
            "is_mixed_age_couple": [True],
        }.get(variable, [0.0, 0.0, 0.0])
        return pd.Series(values)


def test_a_unit_leaving_the_pension_route_later_is_drawn():
    """A unit on the Pension Credit route in the survey year that the model
    moves from 2025 is drawn, not given the default True (review r3)."""
    from policyengine_uk_data.datasets.frs import first_year_moved_to_uc

    benunit = pd.DataFrame({"benunit_id": [10]})
    sim = _YearlySimulation()
    assert first_year_moved_to_uc(sim, benunit, 2024, [2024]).tolist() == [0]
    assert first_year_moved_to_uc(sim, benunit, 2024, range(2025, 2036)).tolist() == [
        2025
    ]
