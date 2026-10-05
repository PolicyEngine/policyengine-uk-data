"""Benefit reports and the flags built from them on SPI-donor rows.

Properties of ``apply_spi_donor_benefit_rules`` for any input:

1. Every column in ``SPI_DONOR_ZEROED_PERSON_VARIABLES`` is zero afterwards,
   and every column in ``SPI_DONOR_RESTORED_PERSON_VARIABLES`` equals the
   donor's value.
2. Nothing else changes except ``receives_benefits_in_own_right`` and the
   redrawn take-up flags, and the input is not mutated.
3. A benefit unit with a member reporting UC or Pension Credit claims it.
4. ``receives_benefits_in_own_right`` holds exactly when the person reports
   one of ``BENEFITS_IN_OWN_RIGHT_REPORTED_COLUMNS``.
5. The rules are deterministic and idempotent, and no flag's draws depend on
   which other columns are present.

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
    SPI_DONOR_REDRAWN_TAKEUP_FLAGS,
    SPI_DONOR_RESTORED_PERSON_VARIABLES,
    SPI_DONOR_ZEROED_PERSON_VARIABLES,
    apply_spi_donor_benefit_rules,
)
from policyengine_uk_data.parameters import load_take_up_rate

YEAR = 2024
REPORT_COLUMNS = sorted(
    set(FRS_ONLY_PERSON_VARIABLES) | set(SPI_DONOR_ZEROED_PERSON_VARIABLES)
)
CHANGED = {"receives_benefits_in_own_right", *SPI_DONOR_REDRAWN_TAKEUP_FLAGS}
RULED = set(SPI_DONOR_ZEROED_PERSON_VARIABLES) | set(
    SPI_DONOR_RESTORED_PERSON_VARIABLES
)
ANCHORING_REPORTS = {
    REPORTED_TAKEUP_ANCHORS[flag][1] for flag in SPI_DONOR_REDRAWN_TAKEUP_FLAGS
}


def test_rule_sets_are_pinned():
    """Each column's treatment is a decision; changing one must be deliberate."""
    assert set(SPI_DONOR_ZEROED_PERSON_VARIABLES) == {
        "universal_credit_reported",
        "pension_credit_reported",
        "housing_benefit_reported",
        "income_support_reported",
        "working_tax_credit_reported",
        "child_tax_credit_reported",
        "jsa_income_reported",
        "esa_income_reported",
        "ssmg_reported",
        "jsa_contrib_reported",
        "esa_contrib_reported",
        "incapacity_benefit_reported",
        "sda_reported",
        "child_benefit_reported",
    }
    assert set(SPI_DONOR_RESTORED_PERSON_VARIABLES) == {
        "iidb_reported",
        "afcs_reported",
        "bsp_reported",
    }
    # Kept as drawn: every report the QRF draws that no rule above touches.
    drawn_reports = {c for c in FRS_ONLY_PERSON_VARIABLES if c.endswith("_reported")}
    assert drawn_reports - RULED == {
        "state_pension_reported",
        "winter_fuel_allowance_reported",
        "attendance_allowance_reported",
        "dla_sc_reported",
        "dla_m_reported",
        "pip_m_reported",
        "pip_dl_reported",
        "carers_allowance_reported",
        "maternity_allowance_reported",
        "council_tax_benefit_reported",
    }
    assert set(SPI_DONOR_REDRAWN_TAKEUP_FLAGS) == {"would_claim_uc", "would_claim_pc"}
    assert not set(SPI_DONOR_ZEROED_PERSON_VARIABLES) & set(
        SPI_DONOR_RESTORED_PERSON_VARIABLES
    )


def _dataset(benunit_sizes, reports, flags, id_seed=0) -> UKSingleYearDataset:
    """Benefit units of the given sizes, with shuffled, gapped ids."""
    rng = np.random.default_rng(id_seed)
    n_benunits, n_people = len(benunit_sizes), sum(benunit_sizes)
    benunit_ids = rng.permutation(n_benunits) * 7 + 3
    benunit_of_person = np.repeat(benunit_ids, benunit_sizes)
    person = pd.DataFrame(
        {
            "person_id": rng.permutation(n_people) * 5 + 11,
            "person_benunit_id": benunit_of_person,
            "person_household_id": benunit_of_person,
            "age": np.full(n_people, 40),
            "receives_benefits_in_own_right": np.asarray(flags[:n_people]),
        }
    )
    for i, column in enumerate(REPORT_COLUMNS):
        person[column] = np.asarray(reports[i][:n_people], dtype=float)
    person = person.sample(frac=1, random_state=id_seed).reset_index(drop=True)
    benunit = pd.DataFrame({"benunit_id": benunit_ids})
    for j, flag in enumerate(REPORTED_TAKEUP_ANCHORS):
        benunit[flag] = np.roll(np.asarray(flags[:n_benunits]), j)
    household = pd.DataFrame({"household_id": benunit_ids, "household_weight": 0.0})
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
    return _dataset(sizes, reports, flags, id_seed=draw(st.integers(0, 10**6)))


def _donor(dataset, seed=1):
    """A donor person table: the same people with different report values."""
    donor = dataset.person.copy()
    rng = np.random.default_rng(seed)
    for column in SPI_DONOR_RESTORED_PERSON_VARIABLES:
        donor[column] = np.where(rng.random(len(donor)) < 0.5, 0.0, 1_234.0)
    return donor


def _reporting_benunits(person, benunit, column):
    reporters = person.loc[person[column] > 0, "person_benunit_id"]
    return benunit.benunit_id.isin(set(reporters)).values


@settings(max_examples=60, deadline=None)
@given(datasets())
def test_rules_zero_restore_and_touch_nothing_else(dataset):
    before = dataset.copy()
    donor = _donor(dataset)
    after = apply_spi_donor_benefit_rules(dataset, donor)

    for column in SPI_DONOR_ZEROED_PERSON_VARIABLES:
        assert (after.person[column] == 0).all(), column
    for column in SPI_DONOR_RESTORED_PERSON_VARIABLES:
        np.testing.assert_array_equal(after.person[column], donor[column])
    unchanged = [c for c in after.person.columns if c not in CHANGED | RULED]
    pd.testing.assert_frame_equal(after.person[unchanged], before.person[unchanged])
    kept_flags = [c for c in after.benunit.columns if c not in CHANGED]
    pd.testing.assert_frame_equal(after.benunit[kept_flags], before.benunit[kept_flags])
    pd.testing.assert_frame_equal(after.household, before.household)
    pd.testing.assert_frame_equal(dataset.person, before.person)
    pd.testing.assert_frame_equal(dataset.benunit, before.benunit)


@settings(max_examples=30, deadline=None)
@given(datasets())
def test_restored_columns_untouched_without_donor(dataset):
    after = apply_spi_donor_benefit_rules(dataset)
    restored = list(SPI_DONOR_RESTORED_PERSON_VARIABLES)
    pd.testing.assert_frame_equal(after.person[restored], dataset.person[restored])


@settings(
    max_examples=60,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(dataset=datasets())
def test_flags_follow_own_reports_whatever_is_zeroed(dataset, monkeypatch):
    # With the own-right and anchoring reports left in place, the rebuilt
    # flags must follow them, not the donor's flags.
    monkeypatch.setattr(
        frs_only,
        "SPI_DONOR_ZEROED_PERSON_VARIABLES",
        [
            c
            for c in SPI_DONOR_ZEROED_PERSON_VARIABLES
            if c not in ANCHORING_REPORTS | set(BENEFITS_IN_OWN_RIGHT_REPORTED_COLUMNS)
        ],
    )
    after = apply_spi_donor_benefit_rules(dataset)
    own = after.person[list(BENEFITS_IN_OWN_RIGHT_REPORTED_COLUMNS)].sum(axis=1) > 0
    assert (after.person.receives_benefits_in_own_right == own).all()
    for flag in SPI_DONOR_REDRAWN_TAKEUP_FLAGS:
        column = REPORTED_TAKEUP_ANCHORS[flag][1]
        reports = _reporting_benunits(after.person, after.benunit, column)
        assert after.benunit[flag].values[reports].all(), flag


@settings(max_examples=40, deadline=None)
@given(datasets())
def test_rules_are_deterministic_and_idempotent(dataset):
    donor = _donor(dataset)
    once = apply_spi_donor_benefit_rules(dataset, donor)
    again = apply_spi_donor_benefit_rules(dataset, donor)
    twice = apply_spi_donor_benefit_rules(once, donor)
    for frame in ("person", "benunit", "household"):
        pd.testing.assert_frame_equal(getattr(once, frame), getattr(again, frame))
        pd.testing.assert_frame_equal(getattr(once, frame), getattr(twice, frame))


@settings(max_examples=30, deadline=None)
@given(datasets())
def test_draws_do_not_depend_on_other_columns(dataset):
    full = apply_spi_donor_benefit_rules(dataset)
    dataset.benunit = dataset.benunit.drop(columns=["would_claim_uc"])
    partial = apply_spi_donor_benefit_rules(dataset)
    pd.testing.assert_series_equal(
        full.benunit.would_claim_pc, partial.benunit.would_claim_pc
    )


def test_unreported_units_claim_at_the_take_up_rate():
    n = 40_000
    rng = np.random.default_rng(0)
    reports = [rng.gamma(2, 1_000, n) for _ in REPORT_COLUMNS]
    dataset = _dataset([1] * n, reports, np.ones(n, dtype=bool))
    after = apply_spi_donor_benefit_rules(dataset).benunit
    for flag in SPI_DONOR_REDRAWN_TAKEUP_FLAGS:
        rate = load_take_up_rate(REPORTED_TAKEUP_ANCHORS[flag][0], YEAR)
        assert after[flag].mean() == pytest.approx(rate, abs=0.01), flag
    assert after.would_claim_child_benefit.all()


def test_take_up_rates_are_read_for_the_dataset_year(monkeypatch):
    years = []

    def rate(name, year):
        years.append(year)
        return 0.5

    monkeypatch.setattr("policyengine_uk_data.datasets.frs.load_take_up_rate", rate)
    dataset = _dataset([1, 2], [[1.0] * 3 for _ in REPORT_COLUMNS], [True] * 3)
    dataset = UKSingleYearDataset(
        person=dataset.person,
        benunit=dataset.benunit,
        household=dataset.household,
        fiscal_year=2031,
    )
    apply_spi_donor_benefit_rules(dataset)
    assert years == [2031] * len(SPI_DONOR_REDRAWN_TAKEUP_FLAGS)


def test_stage_two_applies_the_rules_and_keeps_drawn_values(monkeypatch):
    """Through ``impute_frs_only_variables``: zeroed, restored, kept and flags."""
    from policyengine_uk_data.tests.test_frs_only_imputation import _fake_dataset

    train = _fake_dataset(person_rows=400, seed=0)
    rng = np.random.default_rng(1)
    for column in ("esa_contrib_reported", "state_pension_reported", "iidb_reported"):
        train.person[column] = np.where(rng.random(400) < 0.3, 5_000.0, 0.0)
    target = _fake_dataset(person_rows=80, seed=1)
    target.person["ssmg_reported"] = 600.0
    target.person["iidb_reported"] = np.where(rng.random(80) < 0.5, 0.0, 777.0)
    target.person["receives_benefits_in_own_right"] = True
    target.benunit["would_claim_uc"] = True
    target.benunit["would_claim_child_benefit"] = False

    ruled = frs_only.impute_frs_only_variables(train, target)
    monkeypatch.setattr(frs_only, "SPI_DONOR_ZEROED_PERSON_VARIABLES", [])
    monkeypatch.setattr(frs_only, "SPI_DONOR_RESTORED_PERSON_VARIABLES", [])
    unruled = frs_only.impute_frs_only_variables(train, target)

    for column in SPI_DONOR_ZEROED_PERSON_VARIABLES:
        assert (ruled.person[column] == 0).all(), column
    np.testing.assert_array_equal(
        ruled.person.iidb_reported, target.person.iidb_reported
    )
    assert not np.array_equal(unruled.person.iidb_reported, target.person.iidb_reported)
    kept = [c for c in FRS_ONLY_PERSON_VARIABLES if c not in RULED]
    pd.testing.assert_frame_equal(ruled.person[kept], unruled.person[kept])
    assert not ruled.person.receives_benefits_in_own_right.any()
    assert not ruled.benunit.would_claim_child_benefit.any()
    # Every unit entered claiming UC as the donor; with the reports zeroed,
    # the redraw at the 55% rate leaves some 80 units out.
    assert not ruled.benunit.would_claim_uc.all()

    # Disability flags come from the final reports: ESA (contributory) was
    # drawn for some people but no longer marks them disabled.
    assert (unruled.person.esa_contrib_reported > 0).any()
    flags = ["is_disabled_for_benefits", "is_severely_disabled_for_benefits"]
    recomputed = add_disability_benefit_flags_from_reported_amounts(
        ruled.person.drop(columns=flags), int(str(ruled.time_period)[:4])
    )
    pd.testing.assert_frame_equal(ruled.person[flags], recomputed[flags])


def test_council_tax_reduction_keeps_the_stage_two_draw():
    """SPI rows carry the QRF's CTR draw through unchanged, as before the
    rules: not zeroed, not the donor's. Zeroing waits for #499."""
    from policyengine_uk_data.tests.test_frs_only_imputation import _fake_dataset

    train = _fake_dataset(person_rows=400, seed=0)
    rng = np.random.default_rng(2)
    train.person["council_tax_benefit_reported"] = np.where(
        rng.random(400) < 0.3, 1_200.0, 0.0
    )
    target = _fake_dataset(person_rows=80, seed=1)
    target.person["council_tax_benefit_reported"] = 999.0
    outputs = [c for c in FRS_ONLY_PERSON_VARIABLES if c in target.person.columns]

    # The draw alone, which is what stage two returned before the rules.
    drawn = frs_only._impute_outputs(train, target.copy(), outputs).person
    ruled = frs_only.impute_frs_only_variables(train, target).person

    np.testing.assert_array_equal(
        ruled.council_tax_benefit_reported, drawn.council_tax_benefit_reported
    )
    assert (drawn.council_tax_benefit_reported > 0).any()
    assert (drawn.council_tax_benefit_reported != 999.0).any()
