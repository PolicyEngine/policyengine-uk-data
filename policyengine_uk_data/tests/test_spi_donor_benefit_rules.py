"""Benefit reports and the flags built from them on SPI-donor rows.

Properties of ``apply_spi_donor_benefit_rules`` for any input:

1. Every column in ``SPI_DONOR_ZEROED_PERSON_VARIABLES`` is zero afterwards.
2. Nothing else changes except ``receives_benefits_in_own_right`` and the
   report-anchored take-up flags; the input is not mutated.
3. A benefit unit with a member reporting an anchoring benefit claims it.
4. ``receives_benefits_in_own_right`` is true exactly when the person
   reports one of ``BENEFITS_IN_OWN_RIGHT_REPORTED_COLUMNS``.
5. The rules are deterministic and idempotent.

The fixtures are synthetic, not survey records.
"""

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from policyengine_uk.data import UKSingleYearDataset

from policyengine_uk_data.datasets.disability_benefits import (
    add_disability_benefit_flags_from_reported_amounts,
)
from policyengine_uk_data.datasets.frs import (
    BENEFITS_IN_OWN_RIGHT_REPORTED_COLUMNS,
    REPORTED_TAKEUP_ANCHORS,
)
from policyengine_uk_data.datasets.imputations import frs_only
from policyengine_uk_data.datasets.imputations.frs_only import (
    FRS_ONLY_PERSON_VARIABLES,
    SPI_DONOR_ZEROED_PERSON_VARIABLES,
    apply_spi_donor_benefit_rules,
)
from policyengine_uk_data.parameters import load_take_up_rate

YEAR = 2024
REPORT_COLUMNS = sorted(
    set(FRS_ONLY_PERSON_VARIABLES) | set(SPI_DONOR_ZEROED_PERSON_VARIABLES)
)
DERIVED = {"receives_benefits_in_own_right", *REPORTED_TAKEUP_ANCHORS}


def _dataset(benunit_sizes, reports, flags) -> UKSingleYearDataset:
    n_benunits, n_people = len(benunit_sizes), sum(benunit_sizes)
    benunit_of_person = np.repeat(np.arange(n_benunits), benunit_sizes)
    person = pd.DataFrame(
        {
            "person_id": np.arange(n_people),
            "person_benunit_id": benunit_of_person,
            "person_household_id": benunit_of_person,
            "age": np.full(n_people, 40),
            "receives_benefits_in_own_right": np.asarray(flags[:n_people]),
        }
    )
    for i, column in enumerate(REPORT_COLUMNS):
        person[column] = np.asarray(reports[i][:n_people], dtype=float)
    benunit = pd.DataFrame({"benunit_id": np.arange(n_benunits)})
    for j, flag in enumerate(REPORTED_TAKEUP_ANCHORS):
        benunit[flag] = np.roll(np.asarray(flags[:n_benunits]), j)
    household = pd.DataFrame(
        {"household_id": np.arange(n_benunits), "household_weight": 0.0}
    )
    return UKSingleYearDataset(
        person=person, benunit=benunit, household=household, fiscal_year=YEAR
    )


@st.composite
def datasets(draw):
    sizes = draw(st.lists(st.integers(1, 4), min_size=1, max_size=12))
    n = sum(sizes)
    amount = st.one_of(st.just(0.0), st.floats(0.01, 50_000))
    reports = [draw(st.lists(amount, min_size=n, max_size=n)) for _ in REPORT_COLUMNS]
    flags = draw(st.lists(st.booleans(), min_size=n, max_size=n))
    return _dataset(sizes, reports, flags)


def _reporting_benunits(person, benunit, column):
    reporters = person.loc[person[column] > 0, "person_benunit_id"]
    return benunit.benunit_id.isin(set(reporters)).values


@settings(max_examples=60, deadline=None)
@given(datasets())
def test_rules_zero_listed_reports_and_touch_nothing_else(dataset):
    before = dataset.copy()
    after = apply_spi_donor_benefit_rules(dataset)

    for column in SPI_DONOR_ZEROED_PERSON_VARIABLES:
        assert (after.person[column] == 0).all(), column
    kept = [c for c in after.person.columns if c not in DERIVED]
    unchanged = [c for c in kept if c not in SPI_DONOR_ZEROED_PERSON_VARIABLES]
    pd.testing.assert_frame_equal(after.person[unchanged], before.person[unchanged])
    pd.testing.assert_frame_equal(after.household, before.household)
    pd.testing.assert_frame_equal(dataset.person, before.person)
    pd.testing.assert_frame_equal(dataset.benunit, before.benunit)


@settings(max_examples=60, deadline=None)
@given(datasets())
def test_rules_rebuild_own_right_flag_from_own_reports(dataset):
    after = apply_spi_donor_benefit_rules(dataset).person
    own = after[list(BENEFITS_IN_OWN_RIGHT_REPORTED_COLUMNS)].sum(axis=1) > 0
    assert (after.receives_benefits_in_own_right == own).all()


@settings(
    max_examples=60,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(dataset=datasets())
def test_reporters_claim_whatever_is_zeroed(dataset, monkeypatch):
    # With no anchoring report zeroed, reporters must still claim: the
    # anchors are rebuilt from the rows' own reports, not left as copies.
    anchoring = {column for _, column in REPORTED_TAKEUP_ANCHORS.values()}
    monkeypatch.setattr(
        frs_only,
        "SPI_DONOR_ZEROED_PERSON_VARIABLES",
        [c for c in SPI_DONOR_ZEROED_PERSON_VARIABLES if c not in anchoring],
    )
    after = apply_spi_donor_benefit_rules(dataset)
    for flag, (_, column) in REPORTED_TAKEUP_ANCHORS.items():
        reports = _reporting_benunits(after.person, after.benunit, column)
        assert after.benunit[flag].values[reports].all(), flag


@settings(max_examples=40, deadline=None)
@given(datasets())
def test_rules_are_deterministic_and_idempotent(dataset):
    once = apply_spi_donor_benefit_rules(dataset)
    again = apply_spi_donor_benefit_rules(dataset)
    twice = apply_spi_donor_benefit_rules(once)
    for frame in ("person", "benunit", "household"):
        pd.testing.assert_frame_equal(getattr(once, frame), getattr(again, frame))
        pd.testing.assert_frame_equal(getattr(once, frame), getattr(twice, frame))


def test_unreported_units_claim_at_the_take_up_rate():
    n = 40_000
    rng = np.random.default_rng(0)
    reports = [rng.gamma(2, 1_000, n) for _ in REPORT_COLUMNS]
    dataset = _dataset([1] * n, reports, np.ones(n, dtype=bool))
    after = apply_spi_donor_benefit_rules(dataset).benunit
    for flag, (rate_name, _) in REPORTED_TAKEUP_ANCHORS.items():
        rate = load_take_up_rate(rate_name, YEAR)
        assert after[flag].mean() == pytest.approx(rate, abs=0.01), flag


def test_stage_two_keeps_drawn_values_of_kept_reports(monkeypatch):
    """Zeroing does not change the QRF draws of the reports that are kept."""
    from policyengine_uk_data.tests.test_frs_only_imputation import _fake_dataset

    train = _fake_dataset(person_rows=400, seed=0)
    for column in ("esa_contrib_reported", "state_pension_reported", "ssmg_reported"):
        train.person[column] = np.where(
            np.random.default_rng(1).random(400) < 0.3, 5_000.0, 0.0
        )
    target = _fake_dataset(person_rows=80, seed=1)
    target.person["ssmg_reported"] = 600.0
    target.person["receives_benefits_in_own_right"] = True
    target.benunit["would_claim_uc"] = True

    ruled = frs_only.impute_frs_only_variables(train, target)
    monkeypatch.setattr(frs_only, "SPI_DONOR_ZEROED_PERSON_VARIABLES", [])
    unruled = frs_only.impute_frs_only_variables(train, target)

    for column in SPI_DONOR_ZEROED_PERSON_VARIABLES:
        assert (ruled.person[column] == 0).all(), column
    kept = [
        c
        for c in FRS_ONLY_PERSON_VARIABLES
        if c not in SPI_DONOR_ZEROED_PERSON_VARIABLES
    ]
    pd.testing.assert_frame_equal(ruled.person[kept], unruled.person[kept])
    assert not ruled.person.receives_benefits_in_own_right.any()

    # Disability flags come from the zeroed reports: ESA (contributory) was
    # drawn for some people but no longer marks them disabled.
    assert (unruled.person.esa_contrib_reported > 0).any()
    flags = ["is_disabled_for_benefits", "is_severely_disabled_for_benefits"]
    recomputed = add_disability_benefit_flags_from_reported_amounts(
        ruled.person.drop(columns=flags), int(str(ruled.time_period)[:4])
    )
    pd.testing.assert_frame_equal(ruled.person[flags], recomputed[flags])
