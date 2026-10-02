import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from policyengine_uk_data.datasets.frs import (
    UC_BENEFIT_CODE,
    UC_START_UP_PERIOD_MONTHS,
    add_uc_start_up_period,
    derive_uc_is_in_startup_period,
    frs_interview_date,
    parse_frs_uc_claim_start,
    uc_claim_began_in_start_up_window,
)

START_UP = "uc_is_in_startup_period"
months = st.one_of(st.just(np.nan), st.floats(0, 200, allow_nan=False))
years = st.one_of(st.just(np.nan), st.integers(0, 60).map(float))
draws = st.floats(0, 1, allow_nan=False, exclude_max=True)
shares = st.floats(0, 1, allow_nan=False)


def sas_date(day: str) -> int:
    return (pd.Timestamp(day) - pd.Timestamp("1960-01-01")).days


def test_interview_date_is_a_sas_date():
    assert frs_interview_date([0])[0] == pd.Timestamp("1960-01-01")
    assert frs_interview_date([sas_date("2024-04-01")])[0] == pd.Timestamp("2024-04-01")


def test_claim_start_is_month_day_year_and_blank_when_unlinked():
    parsed = parse_frs_uc_claim_start(["1/31/2024", " 12/1/2019 ", "", None, "  "])
    assert parsed[0] == pd.Timestamp("2024-01-31")
    assert parsed[1] == pd.Timestamp("2019-12-01")
    assert parsed[2:].isna().all()


@pytest.mark.parametrize("value", ["31/1/2024", "2024-01-31", "45000", "1/2024"])
def test_claim_start_in_another_format_raises(value):
    with pytest.raises(ValueError, match="UCSTART"):
        parse_frs_uc_claim_start(["1/31/2024", value])


def test_start_up_window_is_twelve_months():
    assert UC_START_UP_PERIOD_MONTHS == 12
    window = uc_claim_began_in_start_up_window(
        [0.0, 11.99, 12.0, 30.0, np.nan, np.nan],
        [True, True, True, True, False, True],
        np.full(6, 0.5),
        0.0,
    )
    assert window.tolist() == [True, True, False, False, False, False]


@settings(max_examples=500, deadline=None)
@given(months, st.booleans(), draws, shares)
def test_claim_window_invariants(month, reports_uc, draw, share):
    def window(m=month, r=reports_uc, d=draw, s=share):
        return uc_claim_began_in_start_up_window([m], [r], [d], s)[0]

    result = window()
    if np.isnan(month):
        # No start date: an unlinked UC record is drawn, no UC is false.
        assert result == (reports_uc and draw < share)
        assert not window(r=False)
    else:
        # A linked claim decides from its date alone.
        assert result == (month < UC_START_UP_PERIOD_MONTHS)
        assert window(d=0.0, s=1.0) == window(d=0.999, s=0.0) == result
        # A later start never takes the window away; an earlier one never adds it.
        assert window(m=month / 2) >= result
        assert window(m=month + 12) <= result
    # A higher imputed share never removes the window.
    assert window(s=1.0) >= result >= window(s=0.0)


@settings(max_examples=500, deadline=None)
@given(st.booleans(), st.booleans(), years)
def test_start_up_invariants(self_employed, claim_in_window, years_running):
    def start_up(se=self_employed, c=claim_in_window, y=years_running):
        return derive_uc_is_in_startup_period([se], [c], [y])[0]

    result = start_up()
    # Only the self-employed have a start-up period, and only with a recent
    # claim or a business under a year old.
    if result:
        assert self_employed
        assert claim_in_window or years_running < 1
    assert not start_up(se=False)
    # Either route alone suffices for the self-employed.
    assert start_up(se=True, c=True)
    assert start_up(se=True, y=0.0)
    # An older business never adds the period; a recent claim never removes it.
    if not np.isnan(years_running):
        assert start_up(y=years_running + 1) <= result
    assert start_up(c=True) >= result


@settings(max_examples=100, deadline=None)
@given(st.lists(st.tuples(st.booleans(), st.booleans(), years), max_size=40))
def test_start_up_vectorised_matches_elementwise(rows):
    se, claim, yrs = ([r[i] for r in rows] for i in range(3))
    result = derive_uc_is_in_startup_period(se, claim, yrs)
    assert result.dtype == bool
    assert result.tolist() == [
        derive_uc_is_in_startup_period([a], [b], [c])[0] for a, b, c in rows
    ]


INTERVIEW = "2024-10-01"


def raw_tables(people):
    """Raw-shaped FRS tables (ids already combined, as create_frs makes them).

    ``people`` rows: (household, benunit within household, empstati, seincam2,
    jobs [(jobtype, etype, sejblong)], ucstart or None for no UC, weight).
    """
    person_rows, job_rows, benefit_rows, households = [], [], [], {}
    for i, (hh, bu, empstati, profit, jobs, ucstart, weight) in enumerate(people):
        person_id = hh * 1000 + i + 1
        benunit_id = hh * 100 + bu
        households[hh] = weight
        person_rows.append(
            dict(
                person_id=person_id,
                benunit_id=benunit_id,
                household_id=hh,
                empstati=empstati,
                seincam2=profit,
            )
        )
        for jobtype, etype, sejblong in jobs:
            job_rows.append(
                dict(
                    person_id=person_id, jobtype=jobtype, etype=etype, sejblong=sejblong
                )
            )
        if ucstart is not None and benunit_id not in {
            b["benunit_id"] for b in benefit_rows
        }:
            benefit_rows.append(
                dict(
                    household_id=hh,
                    benunit_id=benunit_id,
                    person_id=person_id,
                    benefit=UC_BENEFIT_CODE,
                    ucstart=ucstart,
                )
            )
    person = pd.DataFrame(person_rows)
    household = pd.DataFrame(
        {
            "household_id": list(households),
            "intdate": sas_date(INTERVIEW),
            "gross4": list(households.values()),
        }
    ).set_index("household_id")
    job = pd.DataFrame(job_rows, columns=["person_id", "jobtype", "etype", "sejblong"])
    benefits = pd.DataFrame(
        benefit_rows,
        columns=["household_id", "benunit_id", "person_id", "benefit", "ucstart"],
    )
    # A non-UC benefit row for the last person, which must not count as a
    # UC claim.
    benefits.loc[len(benefits)] = dict(
        household_id=person.household_id.iloc[-1],
        benunit_id=person.benunit_id.iloc[-1],
        person_id=person.person_id.iloc[-1],
        benefit=3,
        ucstart="",
    )
    pe_person = pd.DataFrame({"person_id": person.person_id})
    pe_benunit = pd.DataFrame({"benunit_id": np.unique(person.benunit_id)})
    add_uc_start_up_period(
        pe_person,
        pe_benunit,
        person,
        household,
        job,
        benefits.drop(columns="ucstart"),
        benefits.ucstart.astype(object),
    )
    return pe_person[START_UP].tolist()


SE, EMPLOYEE = 3, 1
SE_MAIN_OLD = [(1, 4, 5)]
SE_MAIN_NEW = [(1, 4, 0)]
RECENT_CLAIM, OLD_CLAIM = "6/1/2024", "1/15/2020"


def test_start_up_cases():
    flags = raw_tables(
        [
            # Couple, claim four months old: the self-employed partner only.
            (1, 1, SE, 100.0, SE_MAIN_OLD, RECENT_CLAIM, 1.0),
            (1, 1, EMPLOYEE, 0.0, [(1, 1, -1)], RECENT_CLAIM, 1.0),
            # Old claim, old business: the floor's ordinary case.
            (2, 1, SE, 100.0, SE_MAIN_OLD, OLD_CLAIM, 1.0),
            # Old claim, business under a year old.
            (3, 1, SE, 100.0, SE_MAIN_NEW, OLD_CLAIM, 1.0),
            # No UC claim: the business decides.
            (4, 1, SE, 100.0, SE_MAIN_NEW, None, 1.0),
            (5, 1, SE, 100.0, SE_MAIN_OLD, None, 1.0),
            # A new side trade beside a job.
            (6, 1, EMPLOYEE, 50.0, [(1, 1, -1), (2, 4, 0)], OLD_CLAIM, 1.0),
            # An old main trade with a new side trade: the main trade decides.
            (7, 1, SE, 50.0, [(1, 2, 6), (2, 6, 0)], OLD_CLAIM, 1.0),
            # Self-employed by profit alone (no job row), recent claim.
            (8, 1, 0, -20.0, [], RECENT_CLAIM, 1.0),
            # Not self-employed at all, recent claim.
            (9, 1, EMPLOYEE, 0.0, [(1, 1, -1)], RECENT_CLAIM, 1.0),
            # Self-employed, unknown business duration, old claim.
            (10, 1, SE, 100.0, [(1, 4, -9)], OLD_CLAIM, 1.0),
            # A business in its second year, old claim.
            (11, 1, SE, 100.0, [(1, 4, 1)], OLD_CLAIM, 1.0),
            # A new side trade with no profit yet: self-employed by its job row.
            (12, 1, EMPLOYEE, 0.0, [(1, 1, -1), (2, 4, 0)], OLD_CLAIM, 1.0),
        ]
    )
    assert flags == [
        True,
        False,
        False,
        True,
        True,
        False,
        True,
        False,
        True,
        False,
        False,
        False,
        True,
    ]


@pytest.mark.parametrize(
    "linked_claims, expected",
    # Unlinked records take the survey-weighted share of linked self-employed
    # claims that began within the window: here all or none.
    [([RECENT_CLAIM, RECENT_CLAIM], True), ([OLD_CLAIM, OLD_CLAIM], False)],
)
def test_unlinked_claims_take_the_linked_share(linked_claims, expected):
    people = [
        (h + 1, 1, SE, 100.0, SE_MAIN_OLD, claim, 2.0)
        for h, claim in enumerate(linked_claims)
    ]
    people += [(10 + h, 1, SE, 100.0, SE_MAIN_OLD, "", 1.0) for h in range(20)]
    # Last: no UC claim, only another benefit, and an old business.
    people.append((99, 1, SE, 100.0, SE_MAIN_OLD, None, 1.0))
    flags = raw_tables(people)
    assert flags[: len(linked_claims)] == [expected] * len(linked_claims)
    assert flags[len(linked_claims) : -1] == [expected] * 20
    assert not flags[-1]


def test_unlinked_share_is_survey_weighted():
    # One recent linked claim with weight 3, one old with weight 1: share 0.75.
    people = [
        (1, 1, SE, 100.0, SE_MAIN_OLD, RECENT_CLAIM, 3.0),
        (2, 1, SE, 100.0, SE_MAIN_OLD, OLD_CLAIM, 1.0),
    ] + [(10 + h, 1, SE, 100.0, SE_MAIN_OLD, "", 1.0) for h in range(400)]
    unlinked = raw_tables(people)[2:]
    assert 0.68 < np.mean(unlinked) < 0.82


def test_imputation_leaves_global_random_state_alone():
    state = np.random.get_state()[1].copy()
    raw_tables([(1, 1, SE, 100.0, SE_MAIN_OLD, "", 1.0)])
    assert np.array_equal(np.random.get_state()[1], state)


@pytest.mark.parametrize("fixture", ["frs", "enhanced_frs"])
def test_built_dataset_start_up_period(fixture, request):
    dataset = request.getfixturevalue(fixture)
    person = dataset.person
    if START_UP not in person.columns:
        pytest.skip(f"{fixture} was built before this input existed")
    flag = person[START_UP]
    assert flag.dtype == bool
    # Children have no jobs and no claims of their own.
    assert not flag[person.age < 16].any()
    se = (
        person.employment_status.astype(str)
        .isin(["FT_SELF_EMPLOYED", "PT_SELF_EMPLOYED"])
        .to_numpy()
    )
    weight = (
        dataset.household.set_index("household_id")
        .household_weight.reindex(person.person_household_id)
        .to_numpy()
    )
    share = (weight * flag)[se].sum() / weight[se].sum()
    # FRS 2024-25: about 8% of self-employed adults run a business under a
    # year old, and UC claimants add a little.
    assert 0.03 < share < 0.25
