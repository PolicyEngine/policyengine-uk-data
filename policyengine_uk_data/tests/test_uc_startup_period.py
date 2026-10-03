import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from policyengine_uk_data.datasets.frs import (
    SELF_EMPLOYED_STATUSES,
    UC_BENEFIT_CODE,
    UC_START_UP_PERIOD_MONTHS,
    add_uc_start_up_period,
    completed_months,
    derive_uc_is_in_startup_period,
    frs_interview_date,
    parse_frs_uc_claim_start,
    uc_claim_began_in_start_up_window,
    years_running_trade,
)
from policyengine_uk_data.tests.test_imputation_source_flags import (
    _FakeDataset,
    _stack_without_remapping,
)

START_UP = "uc_is_in_startup_period"
months = st.one_of(st.just(np.nan), st.floats(0, 200, allow_nan=False))
years = st.one_of(st.just(np.nan), st.integers(0, 60).map(float))
draws = st.floats(0, 1, allow_nan=False, exclude_max=True)
shares = st.floats(0, 1, allow_nan=False)
dates = st.dates(
    min_value=pd.Timestamp("2000-01-01").date(),
    max_value=pd.Timestamp("2030-12-31").date(),
)


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


def test_completed_months_counts_whole_calendar_months():
    later = pd.Series(pd.to_datetime(["2024-10-01"] * 3 + ["2025-01-31", None]))
    earlier = pd.Series(
        pd.to_datetime(
            ["2023-10-01", "2023-10-02", "2024-06-15", "2024-12-31", "2024-01-01"]
        )
    )
    result = completed_months(later, earlier)
    assert result[:4].tolist() == [12, 11, 3, 1]
    assert np.isnan(result[4])


@settings(max_examples=500, deadline=None)
@given(dates, dates, st.integers(0, 40))
def test_completed_months_invariants(a, b, k):
    def m(later, earlier):
        return completed_months(
            pd.Series([pd.Timestamp(later)]), pd.Series([pd.Timestamp(earlier)])
        )[0]

    assert m(a, a) == 0
    # Swapping the dates negates the count, less one unless the days match.
    assert m(a, b) + m(b, a) == -(a.day != b.day)
    # k calendar months on: k whole months, or k - 1 when the day clips at a month end.
    shifted = pd.Timestamp(a) + pd.DateOffset(months=k)
    assert m(shifted, a) in (k, k - 1)
    if a.day <= 28:
        assert m(shifted, a) == k
    # A later end date never counts fewer months.
    assert m(pd.Timestamp(b) + pd.Timedelta(days=1), a) >= m(b, a)


def test_start_up_window_is_twelve_months():
    assert UC_START_UP_PERIOD_MONTHS == 12
    window = uc_claim_began_in_start_up_window(
        [0.0, 11.0, 12.0, 30.0, np.nan, np.nan],
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
@given(years, st.booleans(), st.booleans())
def test_trade_age_invariants(years_in_job, business, all_year):
    trade = years_running_trade([years_in_job], [business], [all_year])[0]
    # The trade's age is the job's, or unknown; never a different number.
    assert np.isnan(trade) or trade == years_in_job
    # Unknown only for a job under a year old, held by someone self-employed
    # all year, that is not described as running a business.
    lost = np.isnan(trade) and not np.isnan(years_in_job)
    assert lost == (years_in_job < 1 and not business and all_year)
    # A business's age is always its own.
    if business:
        assert np.array_equal(
            years_running_trade([years_in_job], [True], [all_year]),
            [years_in_job],
            equal_nan=True,
        )


@settings(max_examples=500, deadline=None)
@given(st.booleans(), st.booleans(), years)
def test_start_up_invariants(self_employed, claim_in_window, years_running):
    def start_up(se=self_employed, c=claim_in_window, y=years_running):
        return derive_uc_is_in_startup_period([se], [c], [y])[0]

    result = start_up()
    # Only the self-employed have a start-up period, and only with a recent
    # claim or a trade under a year old.
    if result:
        assert self_employed
        assert claim_in_window or years_running < 1
    assert not start_up(se=False)
    # Either route alone suffices for the self-employed.
    assert start_up(se=True, c=True)
    assert start_up(se=True, y=0.0)
    # An older trade never adds the period; a recent claim never removes it.
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
RECENT_CLAIM, OLD_CLAIM, UNLINKED = "6/1/2024", "1/15/2020", ""
SE, EMPLOYEE = 3, 1
BUSINESS, JOB = 2, 1
SE_MONTH, EMPLOYEE_MONTH = 3, 1


def job(jobtype=1, etype=4, years=5, jobbus=BUSINESS, seend=None):
    return dict(
        jobtype=jobtype, etype=etype, sejblong=years, jobbus=jobbus, seend=seend
    )


def adult(
    hh,
    bu=1,
    empstati=SE,
    profit=100.0,
    jobs=None,
    claims=(),
    weight=1.0,
    samesit=2,
    calendar=None,
):
    """One FRS adult. ``claims`` are UCSTART values for UC rows on their
    benefit unit (blank = unlinked); ``calendar`` is SDEMP01-12."""
    return dict(
        hh=hh,
        bu=bu,
        empstati=empstati,
        profit=profit,
        jobs=[job()] if jobs is None and empstati == SE else (jobs or []),
        claims=claims,
        weight=weight,
        samesit=samesit,
        calendar=calendar or [0] * 12,
    )


def start_up_flags(people):
    """Run add_uc_start_up_period on raw-shaped tables (ids combined as create_frs does)."""
    person_rows, job_rows, benefit_rows, households = [], [], [], {}
    for i, p in enumerate(people):
        person_id = p["hh"] * 1000 + i + 1
        benunit_id = p["hh"] * 100 + p["bu"]
        households[p["hh"]] = p["weight"]
        person_rows.append(
            dict(
                person_id=person_id,
                benunit_id=benunit_id,
                household_id=p["hh"],
                empstati=p["empstati"],
                seincam2=p["profit"],
                samesit=p["samesit"],
                **{
                    f"sdemp{m:02d}": code
                    for m, code in enumerate(p["calendar"], start=1)
                },
            )
        )
        job_rows += [dict(person_id=person_id, **j) for j in p["jobs"]]
        benefit_rows += [
            dict(
                household_id=p["hh"],
                benunit_id=benunit_id,
                benefit=UC_BENEFIT_CODE,
                ucstart_raw=c,
            )
            for c in p["claims"]
        ]
    person = pd.DataFrame(person_rows)
    # A non-UC benefit row on the last person's benefit unit, which must not
    # count as a UC claim.
    benefit_rows.append(
        dict(
            household_id=person.household_id.iloc[-1],
            benunit_id=person.benunit_id.iloc[-1],
            benefit=3,
            ucstart_raw="",
        )
    )
    household = pd.DataFrame(
        {
            "household_id": list(households),
            "intdate": sas_date(INTERVIEW),
            "gross4": list(households.values()),
        }
    ).set_index("household_id")
    jobs = pd.DataFrame(
        job_rows,
        columns=["person_id", "jobtype", "etype", "sejblong", "jobbus", "seend"],
    )
    jobs[["jobtype", "etype", "sejblong", "jobbus", "seend"]] = jobs[
        ["jobtype", "etype", "sejblong", "jobbus", "seend"]
    ].astype(float)
    benefits = pd.DataFrame(benefit_rows).astype({"ucstart_raw": object})
    pe_person = pd.DataFrame({"person_id": person.person_id})
    pe_benunit = pd.DataFrame({"benunit_id": np.unique(person.benunit_id)})
    add_uc_start_up_period(pe_person, pe_benunit, person, household, jobs, benefits)
    return pe_person[START_UP].tolist()


CASES = [
    # (description, adults, expected flags)
    (
        "couple, claim four months old: the self-employed partner only",
        [
            adult(1, claims=[RECENT_CLAIM]),
            adult(1, empstati=EMPLOYEE, profit=0.0, jobs=[job(etype=1, years=-1)]),
        ],
        [True, False],
    ),
    (
        "old claim, old business: the floor's ordinary case",
        [adult(1, claims=[OLD_CLAIM])],
        [False],
    ),
    (
        "old claim, business under a year old",
        [adult(1, jobs=[job(years=0)], claims=[OLD_CLAIM])],
        [True],
    ),
    (
        "a claim exactly 12 calendar months old is outside",
        [adult(1, claims=["10/1/2023"])],
        [False],
    ),
    (
        "a claim a day under 12 months old is inside",
        [adult(1, claims=["10/2/2023"])],
        [True],
    ),
    (
        "two UC rows on one unit: the latest start",
        [adult(1, claims=[OLD_CLAIM, RECENT_CLAIM])],
        [True],
    ),
    ("no UC: a new business", [adult(1, jobs=[job(years=0)])], [True]),
    ("no UC: an old business", [adult(1)], [False]),
    (
        "a new side trade beside a job",
        [
            adult(
                1,
                empstati=EMPLOYEE,
                profit=50.0,
                jobs=[job(etype=1, years=-1), job(jobtype=2, years=0, jobbus=JOB)],
                claims=[OLD_CLAIM],
            )
        ],
        [True],
    ),
    (
        "a new side trade with no profit yet: self-employed by its job row",
        [
            adult(
                1,
                empstati=EMPLOYEE,
                profit=0.0,
                jobs=[job(etype=1, years=-1), job(jobtype=2, years=0)],
            )
        ],
        [True],
    ),
    (
        "an old main trade with a new side trade: the main trade decides",
        [
            adult(
                1,
                jobs=[job(etype=2, years=6), job(jobtype=2, etype=6, years=0)],
                claims=[OLD_CLAIM],
            )
        ],
        [False],
    ),
    (
        "self-employed by profit alone, recent claim",
        [adult(1, empstati=0, profit=-20.0, jobs=[], claims=[RECENT_CLAIM])],
        [True],
    ),
    (
        "not self-employed, recent claim",
        [
            adult(
                1,
                empstati=EMPLOYEE,
                profit=0.0,
                jobs=[job(etype=1, years=-1)],
                claims=[RECENT_CLAIM],
            )
        ],
        [False],
    ),
    (
        "unknown business duration, old claim",
        [adult(1, jobs=[job(years=-9)], claims=[OLD_CLAIM])],
        [False],
    ),
    (
        "a business in its second year",
        [adult(1, jobs=[job(years=1)], claims=[OLD_CLAIM])],
        [False],
    ),
    (
        "self-employed all year, a new 'job': a new engagement in the same trade",
        [adult(1, jobs=[job(years=0, jobbus=JOB)], claims=[OLD_CLAIM])],
        [False],
    ),
    (
        "self-employed every month by the calendar, a new 'job'",
        [
            adult(
                1,
                jobs=[job(years=0, jobbus=JOB)],
                samesit=1,
                calendar=[SE_MONTH, 4] * 6,
            )
        ],
        [False],
    ),
    (
        "employed earlier in the year, a new 'job': a new trade",
        [
            adult(
                1,
                jobs=[job(years=0, jobbus=JOB)],
                samesit=1,
                calendar=[SE_MONTH] * 4 + [EMPLOYEE_MONTH] * 8,
            )
        ],
        [True],
    ),
    (
        "self-employed all year, a new business: a new trade",
        [adult(1, jobs=[job(years=0)])],
        [True],
    ),
    (
        "a self-employed job given up (SEEND) is not self-employment",
        [
            adult(
                1,
                empstati=5,
                profit=0.0,
                jobs=[job(years=0, seend=sas_date("2024-08-01"))],
                claims=[RECENT_CLAIM],
            )
        ],
        [False],
    ),
]


@pytest.mark.parametrize(
    "description, adults, expected", CASES, ids=[c[0] for c in CASES]
)
def test_start_up_cases(description, adults, expected):
    assert start_up_flags(adults) == expected


@pytest.mark.parametrize(
    "linked_claim, expected", [(RECENT_CLAIM, True), (OLD_CLAIM, False)]
)
def test_unlinked_claims_take_the_linked_self_employed_share(linked_claim, expected):
    people = [adult(h, claims=[linked_claim], weight=2.0) for h in (1, 2)]
    # Linked claimants who are not self-employed do not set the share.
    people += [
        adult(
            h,
            empstati=EMPLOYEE,
            profit=0.0,
            jobs=[job(etype=1, years=-1)],
            claims=[RECENT_CLAIM if not expected else OLD_CLAIM],
        )
        for h in range(3, 13)
    ]
    people += [adult(h, claims=[UNLINKED]) for h in range(20, 40)]
    # Last: no UC claim, only another benefit, and an old business.
    people.append(adult(99))
    flags = start_up_flags(people)
    assert flags[:2] == [expected] * 2
    assert flags[12:-1] == [expected] * 20
    assert not flags[-1]


@pytest.mark.parametrize(
    "people, low, high",
    [
        # A recent claim with weight 3 and an old one with weight 1: 0.75.
        (
            [adult(1, claims=[RECENT_CLAIM], weight=3.0), adult(2, claims=[OLD_CLAIM])],
            0.69,
            0.81,
        ),
        # Weighted per person, not per benefit unit: a self-employed couple on
        # a recent claim and a single trader on an old one give 2/3, not 1/2.
        (
            [adult(1, claims=[RECENT_CLAIM]), adult(1), adult(2, claims=[OLD_CLAIM])],
            0.61,
            0.72,
        ),
    ],
)
def test_unlinked_share_is_survey_weighted_per_person(people, low, high):
    linked = len(people)
    flags = start_up_flags(
        people + [adult(10 + h, claims=[UNLINKED]) for h in range(600)]
    )
    assert low < np.mean(flags[linked:]) < high


def test_imputation_leaves_global_random_state_alone():
    state = np.random.get_state()[1].copy()
    start_up_flags([adult(1, claims=[UNLINKED])])
    assert np.array_equal(np.random.get_state()[1], state)


@settings(max_examples=50, deadline=None)
@given(
    st.lists(
        st.tuples(
            st.sampled_from(
                ["FT_EMPLOYED", "FT_SELF_EMPLOYED", "PT_SELF_EMPLOYED", "UNEMPLOYED"]
            ),
            st.booleans(),
            st.sampled_from([0.0, 5_000.0]),
        ),
        min_size=1,
        max_size=10,
    )
)
def test_spi_copy_keeps_the_flag_only_with_self_employment(rows):
    from policyengine_uk_data.datasets import disability_benefits
    from policyengine_uk_data.datasets.imputations import frs_only
    from policyengine_uk_data.datasets.imputations import income as income_module

    n = len(rows)
    imputed_profit = [r[2] for r in rows]
    person = pd.DataFrame(
        {
            "person_id": np.arange(1, n + 1),
            "person_household_id": np.arange(1, n + 1),
            "person_benunit_id": np.arange(1, n + 1),
            "employment_status": [r[0] for r in rows],
            START_UP: [r[1] for r in rows],
            "employment_income": 0.0,
            "self_employment_income": 1_000.0,
            "savings_interest_income": 0.0,
            "dividend_income": 0.0,
            "private_pension_income": 0.0,
            "property_income": 0.0,
        }
    )
    household = pd.DataFrame(
        {
            "household_id": np.arange(1, n + 1),
            "household_weight": 1.0,
            "region": "LONDON",
        }
    )

    def impute_over_incomes(dataset, _model, output_variables):
        dataset = dataset.copy()
        if "self_employment_income" in output_variables:
            dataset.person["self_employment_income"] = imputed_profit
        return dataset

    with pytest.MonkeyPatch.context() as m:
        m.setattr(income_module, "create_income_model", lambda: object())
        m.setattr(income_module, "subsample_dataset", lambda d, _n: d.copy())
        m.setattr(income_module, "impute_over_incomes", impute_over_incomes)
        m.setattr(
            frs_only,
            "impute_frs_only_variables",
            lambda train_dataset, target_dataset: target_dataset,
        )
        m.setattr(
            disability_benefits,
            "strip_internal_disability_reported_amounts",
            lambda dataset: dataset,
        )
        m.setattr(income_module, "stack_datasets", _stack_without_remapping)
        result = income_module.impute_income(
            _FakeDataset(person=person, household=household)
        )

    flag = result.person[START_UP].to_numpy(dtype=bool)
    donor = np.array([r[1] for r in rows])
    # FRS rows keep their flag.
    assert np.array_equal(flag[:n], donor)
    # Copies keep it only where they are still self-employed.
    still_self_employed = np.isin([r[0] for r in rows], SELF_EMPLOYED_STATUSES) | (
        np.array(imputed_profit) != 0
    )
    assert np.array_equal(flag[n:], donor & still_self_employed)


@pytest.mark.parametrize("fixture", ["frs", "enhanced_frs"])
def test_built_dataset_start_up_period(fixture, request):
    dataset = request.getfixturevalue(fixture)
    person = dataset.person
    if START_UP not in person.columns:
        pytest.skip(f"{fixture} was built before this input existed")
    flag = person[START_UP].to_numpy()
    assert flag.dtype == bool
    # Children have no jobs and no claims of their own.
    assert not flag[person.age.to_numpy() < 16].any()
    se = person.employment_status.astype(str).isin(SELF_EMPLOYED_STATUSES).to_numpy()
    weight = (
        dataset.household.set_index("household_id")
        .household_weight.reindex(person.person_household_id)
        .to_numpy()
    )

    def share(mask):
        return (weight * flag)[mask].sum() / weight[mask].sum()

    # FRS 2024-25: about 9% of self-employed adults are flagged, mostly for a
    # business under a year old.
    assert 0.03 < share(se) < 0.25
    # Among self-employed adults in benefit units reporting UC, about 36% have
    # a claim or trade under a year old: the claim route dominates there.
    reports_uc = (
        pd.Series(person.universal_credit_reported.to_numpy() > 0)
        .groupby(person.person_benunit_id.to_numpy())
        .transform("any")
        .to_numpy()
    )
    assert 0.2 < share(se & reports_uc) < 0.55
