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
4. A legacy reporter not on Universal Credit is True exactly when its draw is
   below its combination's rate ("all" for combinations DWP does not report).
5. Each cohort's share lands within sampling error of its rate.
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


@pytest.mark.parametrize("combination", sorted(RATES))
def test_cohort_shares_within_sampling_error(combination):
    benefits = [] if combination == "all" else combination.split("+")
    if combination == "all":
        # A combination DWP does not report (income-related ESA with Income
        # Support) takes the all-households rate.
        benefits = ["esa_income", "income_support"]
    n = 20_000
    units = [[{f"{b}_reported": 100.0 for b in benefits}] for _ in range(n)]
    person, benunit = frames(units)
    flags = assign_uc_claim_at_legacy_closure(
        person, benunit, RATES, seed=UC_MANAGED_MIGRATION_SEED
    )
    rate = RATES[combination]
    assert abs(flags.mean() - rate) <= 4 * np.sqrt(rate * (1 - rate) / n)


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
    # surveyed benefit unit, so its draws are independent.
    flags, combination, on_uc = _built_flags(frs)
    drawn = ~on_uc & (combination != "")
    for c in set(combination[drawn]):
        in_cohort = drawn & (combination == c)
        n = in_cohort.sum()
        rate = RATES.get(c, RATES["all"])
        if n >= 30:
            share = flags[in_cohort].mean()
            assert abs(share - rate) <= 4 * np.sqrt(rate * (1 - rate) / n), c
