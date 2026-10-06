"""
Family Resources Survey (FRS) dataset processing for PolicyEngine UK.

This module processes raw FRS survey data into PolicyEngine UK dataset format,
handling household demographics, income, benefits, and other survey variables.
The FRS is the primary source of UK household survey data used for tax-benefit
modelling and policy analysis.
"""

import re
import warnings
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from policyengine_uk import CountryTaxBenefitSystem
from policyengine_uk.data import UKSingleYearDataset
from policyengine_uk.variables.household.income.employment_status import (
    EmploymentStatus,
)
from policyengine_uk_data.datasets.brma import assign_brmas, pick_household_brmas
from policyengine_uk_data.datasets.disability_benefits import (
    add_disability_benefit_categories_from_reported_amounts,
    add_disability_benefit_flags_from_reported_amounts,
    drop_internal_disability_reported_amounts,
)
from policyengine_uk_data.utils.benefit_units import claimant_or_partner_variable
from policyengine_uk_data.utils.datasets import (
    sum_to_entity,
    categorical,
    sum_from_positive_fields,
    sum_positive_variables,
    fill_with_mean,
    STORAGE_FOLDER,
)
from policyengine_uk_data.parameters import load_take_up_rate, load_parameter
from policyengine_uk_data.utils.takeup import assign_takeup_with_reported_anchors
from policyengine_uk_data.datasets.childcare.assumptions import (
    EXTENDED_HOURS_MEAN,
    EXTENDED_HOURS_SD,
)


# Canonical weeks-per-year conversion factor for annualising weekly survey
# variables. 365.25 / 7 ≈ 52.1786 accounts for leap years; using the rounded
# integer 52 would under-count by ~0.34%. Exposed at module level so sibling
# loaders (e.g. LCFS/ETB in `datasets/imputations/consumption.py`) can import
# the same value rather than re-defining `* 52` locally and drifting.
WEEKS_IN_YEAR = 365.25 / 7

LEGACY_JOBSEEKER_MIN_AGE = 18
HOURS_WORKED_WEEKS_PER_YEAR = 52
ESA_MIN_AGE = 16
ESA_HEALTH_EMPLOYMENT_STATUSES = (
    EmploymentStatus.LONG_TERM_DISABLED.name,
    EmploymentStatus.SHORT_TERM_DISABLED.name,
)
SELF_EMPLOYED_STATUSES = (
    EmploymentStatus.FT_SELF_EMPLOYED.name,
    EmploymentStatus.PT_SELF_EMPLOYED.name,
)
FORMULA_MODELED_EDUCATION_GRANT_VARIABLES = (
    "childcare_grant",
    "parents_learning_allowance",
    "adult_dependants_grant",
)
DISABLED_STUDENTS_ALLOWANCE_EXPENSE_INPUT = (
    "disabled_students_allowance_eligible_expenses"
)
DISABLED_STUDENTS_ALLOWANCE_FIRST_MODELED_YEAR = 2025
DISABLED_STUDENTS_ALLOWANCE_ELIGIBILITY_VARIABLES = (
    "maintenance_loan_in_england_system",
    "disabled_students_allowance_course_eligible",
    "disabled_students_allowance_has_qualifying_condition",
)
BENEFITS_IN_OWN_RIGHT_REPORTED_COLUMNS = (
    "universal_credit_reported",
    "jsa_contrib_reported",
    "jsa_income_reported",
    "esa_contrib_reported",
    "esa_income_reported",
)
# Take-up flags anchored on reported receipt: flag -> (take-up rate
# parameter, person-level report column). A benefit unit with any member
# reporting receipt claims with certainty; the rest are filled at random.
REPORTED_TAKEUP_ANCHORS = {
    "would_claim_child_benefit": ("child_benefit", "child_benefit_reported"),
    "would_claim_pc": ("pension_credit", "pension_credit_reported"),
    "would_claim_uc": ("universal_credit", "universal_credit_reported"),
}
NON_ADVANCED_EDUCATION_LEVELS = (
    "PRE_PRIMARY",
    "PRIMARY",
    "LOWER_SECONDARY",
    "UPPER_SECONDARY",
    "POST_SECONDARY",
)
# FRS government-training question variants use 10 or 13 for "None of these".
FRS_APPROVED_TRAINING_CODES = tuple(range(1, 10))
UNKNOWN_QUALIFYING_EDUCATION_OR_TRAINING_ENTRY_AGE = 1000
# FRS RENTPROF: whether ROYYR1 is a profit (1) or a loss (2).
FRS_RENTPROF_LOSS = 2


@lru_cache(maxsize=None)
def load_legacy_jobseeker_max_annual_hours(year: int) -> int:
    """Read the JSA single-claimant hours rule from policyengine-uk."""

    system = CountryTaxBenefitSystem()
    max_weekly_hours = int(system.parameters.gov.dwp.JSA.hours.single(str(year)))
    return max_weekly_hours * HOURS_WORKED_WEEKS_PER_YEAR


def require_variable(name: str, description: str) -> None:
    """Fail the build if the installed policyengine-uk lacks ``name``.

    ``UKSingleYearDataset`` drops columns absent from the tax-benefit system
    rather than raising, so writing an input the model does not define is a
    silent no-op. Anything this build depends on the model actually reading
    should be asserted here.
    """
    if name not in CountryTaxBenefitSystem().variables:
        raise RuntimeError(
            f"The installed policyengine-uk does not define {name!r}, so the "
            f"{description} would be written and then silently discarded when "
            "the dataset is loaded. Upgrade policyengine-uk to a release that "
            "defines it."
        )


def derive_legacy_jobseeker_proxy(
    age,
    employment_status,
    hours_worked,
    current_education,
    employment_status_reported,
    state_pension_age,
    max_annual_hours,
) -> np.ndarray:
    """Approximate legacy JSA claimant-state from observed survey data.

    This is intentionally a proxy, not a legislative determination. It
    identifies person-level working-age adults who report being unemployed
    and working less than the legacy JSA 16-hour weekly limit. The
    ``hours_worked`` input is the annualised FRS-derived measure used in the
    dataset, so the threshold is converted to annual hours here.
    """

    age = np.asarray(age)
    employment_status = np.asarray(employment_status)
    hours_worked = np.asarray(hours_worked)
    current_education = np.asarray(current_education)
    employment_status_reported = np.asarray(employment_status_reported)
    state_pension_age = np.asarray(state_pension_age)

    return (
        employment_status_reported
        & (age >= LEGACY_JOBSEEKER_MIN_AGE)
        & (age < state_pension_age)
        & (employment_status == "UNEMPLOYED")
        & (hours_worked < max_annual_hours)
        & (current_education == "NOT_IN_EDUCATION")
    )


def derive_esa_health_condition_proxy(
    age,
    employment_status,
    employment_status_reported,
    state_pension_age,
) -> np.ndarray:
    """Approximate working-age ESA health-related claimant-state.

    This proxy relies only on person-level labour market status, not on
    current disability or incapacity benefit receipt. It is a dataset-side
    approximation for future modelling, not a direct observation of ESA
    legal entitlement or LCW/LCWRA status.
    """

    age = np.asarray(age)
    employment_status = np.asarray(employment_status)
    employment_status_reported = np.asarray(employment_status_reported)
    state_pension_age = np.asarray(state_pension_age)
    disability_labour_market_state = np.isin(
        employment_status, ESA_HEALTH_EMPLOYMENT_STATUSES
    )

    return (
        employment_status_reported
        & (age >= ESA_MIN_AGE)
        & (age < state_pension_age)
        & disability_labour_market_state
    )


def derive_esa_support_group_proxy(
    age,
    employment_status,
    hours_worked,
    esa_health_condition_proxy,
    employment_status_reported,
    state_pension_age,
) -> np.ndarray:
    """Approximate a severe-health ESA subgroup akin to support group.

    This is a stricter subset of ``esa_health_condition_proxy`` intended
    for future legacy ESA approximation work. It uses only non-receipt
    labour market signals already available in the survey.
    """

    age = np.asarray(age)
    employment_status = np.asarray(employment_status)
    hours_worked = np.asarray(hours_worked)
    esa_health_condition_proxy = np.asarray(esa_health_condition_proxy)
    employment_status_reported = np.asarray(employment_status_reported)
    state_pension_age = np.asarray(state_pension_age)
    severe_health_evidence = (employment_status == "LONG_TERM_DISABLED") & (
        hours_worked <= 0
    )

    return (
        employment_status_reported
        & (age >= ESA_MIN_AGE)
        & (age < state_pension_age)
        & esa_health_condition_proxy
        & severe_health_evidence
    )


def reported_benunit_mask(
    person: pd.DataFrame, benunit: pd.DataFrame, person_column: str
) -> np.ndarray:
    """Benefit units with any member reporting a positive ``person_column``."""
    reporter_benunits = set(
        person.loc[person[person_column] > 0, "person_benunit_id"].values
    )
    return benunit["benunit_id"].isin(reporter_benunits).values


def assign_reported_takeup(
    person: pd.DataFrame,
    benunit: pd.DataFrame,
    flag: str,
    year: int,
    draws: np.ndarray,
) -> np.ndarray:
    """Take-up for one ``REPORTED_TAKEUP_ANCHORS`` flag: benefit units
    reporting receipt claim, and the rest are filled at random so the share
    claiming matches the take-up rate (see ``utils/takeup.py``)."""
    rate_name, report_column = REPORTED_TAKEUP_ANCHORS[flag]
    return assign_takeup_with_reported_anchors(
        draws,
        load_take_up_rate(rate_name, year),
        reported_mask=reported_benunit_mask(person, benunit, report_column),
    )


def derive_receives_benefits_in_own_right(pe_person: pd.DataFrame) -> pd.Series:
    """Identify people reporting adult benefits that end QYP status."""

    return (
        pe_person[list(BENEFITS_IN_OWN_RIGHT_REPORTED_COLUMNS)].fillna(0).sum(axis=1)
        > 0
    )


def frs_property_income(person: pd.DataFrame, household: pd.DataFrame) -> np.ndarray:
    """Annual property income each person reports in the FRS.

    Two FRS amounts, both weekly in the released data:

    - SUBRENT, rent the household received for letting part of its home to
      someone outside the household. The FRS asks every household (SubLet),
      whatever its tenure, so renting and rent-free households count too. It
      goes to the household reference person. ``household`` must be indexed
      by ``household_id``.
    - ROYYR1, the person's rent from other property, before tax and after
      allowable expenses. The questionnaire cannot take a negative amount,
      so a loss is entered as a positive amount with RENTPROF = 2 (question
      RentProf, "Is that a profit or a loss from the property?"). A loss
      counts as zero: it is not income, policyengine-uk has no property loss
      input, and it is not set against the household's SUBRENT.

    SUBRENT is used as reported. SUBALLOW records whether it is before (1)
    or after (2) allowable expenses, but the FRS collects no expense amount
    to take off the before-expenses answers.

    Negative values are FRS missing-value codes (-1 to -9), not amounts, so
    each amount is floored at zero before the two are added.

    CVPAY is not included. It is the rent that a boarder or lodger pays the
    householder, and it sits on the boarder's or lodger's own adult record.
    The FRS question (CvPay) asks how much rent [name] paid for board and
    lodging, after deducting any state benefits to help with rent.
    """
    is_head = person.hrpid == 1
    persons_household_subrent = (
        household.subrent.clip(lower=0).reindex(person.household_id).fillna(0).values
    )
    rent_from_other_property = person.royyr1.clip(lower=0).where(
        person.rentprof != FRS_RENTPROF_LOSS, 0
    )
    return (
        (is_head * persons_household_subrent + rent_from_other_property) * WEEKS_IN_YEAR
    ).values


def derive_is_in_non_advanced_education(
    current_education,
    is_apprentice=None,
) -> np.ndarray:
    """Identify current non-advanced education from PolicyEngine education states."""

    current_education = np.asarray(current_education)
    if is_apprentice is None:
        is_apprentice = np.zeros(len(current_education), dtype=bool)
    else:
        is_apprentice = np.asarray(is_apprentice)

    return np.isin(current_education, NON_ADVANCED_EDUCATION_LEVELS) & ~is_apprentice


def derive_is_in_approved_training_from_frs_person(
    person: pd.DataFrame,
) -> pd.Series:
    """Identify reported government training scheme participation in FRS."""

    if "train" not in person.columns:
        return pd.Series(False, index=person.index)

    train = pd.to_numeric(person.train, errors="coerce").fillna(0)
    return train.isin(FRS_APPROVED_TRAINING_CODES)


def derive_age_started_or_accepted_current_education_or_training(
    age,
    is_in_non_advanced_education,
    is_in_approved_training,
) -> np.ndarray:
    """Approximate the entry age for current QYP education or training.

    FRS observes current education/training status but not the age at which the
    current course or programme was started, enrolled on, or accepted. For
    people currently in qualifying education/training, cap the imputed entry
    age at 18 so observed 19-year-olds remain eligible for rules requiring
    entry before age 19.
    """

    age = np.asarray(age)
    in_qualifying_education_or_training = np.asarray(
        is_in_non_advanced_education
    ) | np.asarray(is_in_approved_training)

    return np.where(
        in_qualifying_education_or_training,
        np.minimum(age, 18),
        UNKNOWN_QUALIFYING_EDUCATION_OR_TRAINING_ENTRY_AGE,
    )


def derive_is_before_universal_credit_qualifying_young_person_terminal_date(
    age,
    is_in_non_advanced_education,
    is_in_approved_training,
) -> np.ndarray:
    """Approximate the UC terminal-date condition for observed 19-year-olds.

    FRS does not expose the date-of-birth and assessment-period detail needed
    to identify the exact 1 September terminal date. Use current qualifying
    education/training status as the microdata proxy for age-19 records.
    """

    age = np.asarray(age)
    in_qualifying_education_or_training = np.asarray(
        is_in_non_advanced_education
    ) | np.asarray(is_in_approved_training)

    return (age == 19) & in_qualifying_education_or_training


def add_legacy_benefit_proxies(
    pe_person: pd.DataFrame,
    employment_status_reported,
    state_pension_age,
    legacy_jobseeker_max_annual_hours,
) -> pd.DataFrame:
    """Populate person-scoped ESA/JSA proxy columns on the person frame.

    These remain person-level by design because the claimant-state inputs
    they approximate attach to individuals. Downstream benunit-level legacy
    benefit models should aggregate them explicitly rather than assuming the
    raw survey contains a benunit claimant-state field.
    """

    pe_person["legacy_jobseeker_proxy"] = derive_legacy_jobseeker_proxy(
        age=pe_person.age,
        employment_status=pe_person.employment_status,
        hours_worked=pe_person.hours_worked,
        current_education=pe_person.current_education,
        employment_status_reported=employment_status_reported,
        state_pension_age=state_pension_age,
        max_annual_hours=legacy_jobseeker_max_annual_hours,
    )
    pe_person["esa_health_condition_proxy"] = derive_esa_health_condition_proxy(
        age=pe_person.age,
        employment_status=pe_person.employment_status,
        employment_status_reported=employment_status_reported,
        state_pension_age=state_pension_age,
    )
    pe_person["esa_support_group_proxy"] = derive_esa_support_group_proxy(
        age=pe_person.age,
        employment_status=pe_person.employment_status,
        hours_worked=pe_person.hours_worked,
        esa_health_condition_proxy=pe_person.esa_health_condition_proxy,
        employment_status_reported=employment_status_reported,
        state_pension_age=state_pension_age,
    )
    return pe_person


def apply_legacy_benefit_proxies(
    pe_person: pd.DataFrame, sim, year: int, employment_status_reported
) -> pd.DataFrame:
    """Attach legacy ESA/JSA proxies using post-build simulation context."""

    state_pension_age = sim.calculate("state_pension_age", year).values
    legacy_jobseeker_max_annual_hours = load_legacy_jobseeker_max_annual_hours(year)
    return add_legacy_benefit_proxies(
        pe_person,
        employment_status_reported=employment_status_reported,
        state_pension_age=state_pension_age,
        legacy_jobseeker_max_annual_hours=legacy_jobseeker_max_annual_hours,
    )


def attach_legacy_benefit_proxies_from_frs_person(
    pe_person: pd.DataFrame, person: pd.DataFrame, sim, year: int
) -> pd.DataFrame:
    """Bridge raw FRS person fields into the proxy derivation hook."""

    employment_status_reported = person.empstati.fillna(0).to_numpy() > 0
    return apply_legacy_benefit_proxies(
        pe_person,
        sim,
        year,
        employment_status_reported=employment_status_reported,
    )


def derive_is_parent_from_frs_microdata(
    person_ids,
    person_benunit_ids,
    adult_person_ids,
    benunit_ids,
    dependent_children,
) -> np.ndarray:
    """Identify FRS adults in benefit units with dependent children.

    FRS benefit units contain either one adult or a couple plus any dependent
    children. Using the raw adult table and benefit-unit dependent-child count
    avoids ranking adults across the whole household when multiple benefit
    units share a household.
    """

    dependent_children_by_benunit = pd.Series(
        np.asarray(dependent_children, dtype=float),
        index=np.asarray(benunit_ids),
    )
    has_dependent_children = (
        pd.Series(np.asarray(person_benunit_ids))
        .map(dependent_children_by_benunit)
        .fillna(0)
        .to_numpy()
        > 0
    )
    is_adult_record = np.isin(np.asarray(person_ids), np.asarray(adult_person_ids))
    return is_adult_record & has_dependent_children


def derive_all_claimants_over_state_pension_age(
    person_benunit_ids,
    is_claimant_or_partner,
    is_over_state_pension_age,
    benunit_ids,
) -> np.ndarray:
    """Identify benefit units whose claimant and any partner have all reached
    State Pension age.

    Such a unit cannot claim Universal Credit (Welfare Reform Act 2012
    s.4(1)(b)). ``is_claimant_or_partner`` should be the variable that
    ``claimant_or_partner_variable`` names, so the rule matches the
    pension-age route of policyengine-uk's ``housing_benefit_eligible``.
    """

    claimant = np.asarray(is_claimant_or_partner, dtype=bool)
    over = claimant & np.asarray(is_over_state_pension_age, dtype=bool)
    counts = (
        pd.DataFrame(
            {
                "benunit": np.asarray(person_benunit_ids),
                "claimants": claimant.astype(int),
                "over": over.astype(int),
            }
        )
        .groupby("benunit")[["claimants", "over"]]
        .sum()
        .reindex(np.asarray(benunit_ids), fill_value=0)
    )
    return ((counts.claimants > 0) & (counts.over == counts.claimants)).to_numpy()


def derive_is_claimant_or_partner_from_frs_microdata(
    person_ids,
    person_benunit_ids,
    adult_person_ids,
) -> np.ndarray:
    """Identify each FRS benefit unit's single adult or couple.

    An FRS benefit unit is one adult or a couple plus any dependent children.
    The adult table holds the head (UPERSON 1) and any partner (UPERSON 2);
    the child table holds the dependent children. Any other adult in the
    household, such as a grown-up son or daughter, forms and heads their own
    benefit unit. So adult-table membership is policyengine-uk's
    `is_claimant_or_partner`, which the model otherwise infers from ages.
    """

    is_adult_record = np.isin(np.asarray(person_ids), np.asarray(adult_person_ids))
    per_benunit = pd.Series(is_adult_record).groupby(np.asarray(person_benunit_ids))
    counts = per_benunit.sum()
    if not counts.between(1, 2).all():
        raise ValueError(
            "Every FRS benefit unit needs one or two adult-table records; "
            f"{int((~counts.between(1, 2)).sum())} benefit units do not."
        )
    return is_adult_record


def derive_uc_is_in_gainful_self_employment(
    employment_status, self_employment_income, employment_income
) -> np.ndarray:
    """Whether each person is in gainful self-employment for Universal Credit.

    UC Regs 2013 reg 64(a) asks whether the person carries on a trade as their
    main employment. DWP's Advice for Decision Making starts from hours
    (H4031) but lets earnings outweigh them: someone who works more hours as
    an employee but earns more from self-employment is likely to be gainfully
    self-employed (H4034). So the flag is true for:

    - a self-employed main job (FRS EMPSTATI, the job the respondent names as
      their dominant activity, else the one with more hours), whatever its
      profit: a trade can make a loss or break even and still be carried on
      in expectation of profit (ADM H4013, H4054, H4503);
    - a side trade whose profit is above the person's employment income.

    policyengine-uk's default reads any self-employment income other than
    zero as gainful self-employment, and none as none.

    The flag is fixed from survey-year (or SPI-imputed) incomes. Uprating
    reprices the two incomes by different indices, so in a projected year a
    flagged side trade can earn less than the job, and the reverse; the flag
    does not follow.
    """
    self_employed_main_job = np.isin(
        np.asarray(employment_status, dtype=object), SELF_EMPLOYED_STATUSES
    )
    profit = np.asarray(self_employment_income, dtype=float)
    pay = np.asarray(employment_income, dtype=float)
    return self_employed_main_job | ((profit > 0) & (profit > pay))


def _as_non_negative_array(values) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return np.maximum(np.nan_to_num(values, nan=0.0), 0.0)


def allocate_reported_education_grants(
    reported_grants, grant_capacities: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    """Split aggregate FRS education grants across modelled grant capacity.

    The FRS reports several direct education grants in one aggregate field. When
    several modelled grants are plausible for the same person, allocate the
    reported amount proportionally to each grant's modelled capacity and keep any
    excess in the generic ``education_grants`` residual.
    """

    reported_grants = _as_non_negative_array(reported_grants)
    capacities = {
        variable: _as_non_negative_array(capacity)
        for variable, capacity in grant_capacities.items()
    }
    total_capacity = np.zeros_like(reported_grants, dtype=float)
    for variable, capacity in capacities.items():
        if capacity.shape != reported_grants.shape:
            raise ValueError(
                f"{variable} capacity has shape {capacity.shape}, "
                f"expected {reported_grants.shape}."
            )
        total_capacity += capacity

    allocation_fraction = np.divide(
        reported_grants,
        total_capacity,
        out=np.zeros_like(reported_grants, dtype=float),
        where=total_capacity > 0,
    )
    allocation_fraction = np.minimum(allocation_fraction, 1)

    allocations = {}
    allocated_total = np.zeros_like(reported_grants, dtype=float)
    for variable, capacity in capacities.items():
        allocation = capacity * allocation_fraction
        allocations[variable] = allocation
        allocated_total += allocation

    allocations["education_grants"] = np.maximum(reported_grants - allocated_total, 0)
    return allocations


def calculate_disabled_students_allowance_reported_grant_capacity(
    sim, policy_year: int, maximum: float
) -> np.ndarray:
    """DSA capacity for the first policy year the dataset will be simulated at.

    ``DISABLED_STUDENTS_ALLOWANCE_FIRST_MODELED_YEAR`` is the first policy
    year policyengine-uk models DSA, so the gate compares against the policy
    year rather than the survey year: an FRS 2024-25 build simulated from
    2025 must seed DSA expenses even though its survey year is 2024
    (uk-data#478).
    """
    if policy_year < DISABLED_STUDENTS_ALLOWANCE_FIRST_MODELED_YEAR:
        return np.zeros_like(
            np.asarray(
                sim.calculate(
                    DISABLED_STUDENTS_ALLOWANCE_ELIGIBILITY_VARIABLES[0], policy_year
                )
            ),
            dtype=float,
        )

    eligible = None
    for variable in DISABLED_STUDENTS_ALLOWANCE_ELIGIBILITY_VARIABLES:
        variable_eligible = np.asarray(sim.calculate(variable, policy_year), dtype=bool)
        eligible = (
            variable_eligible if eligible is None else eligible & variable_eligible
        )
    equivalent_support = np.asarray(
        sim.calculate(
            "disabled_students_allowance_receives_equivalent_support", policy_year
        ),
        dtype=bool,
    )
    return np.where(eligible & ~equivalent_support, float(maximum), 0.0)


def split_reported_education_grants(
    pe_person: pd.DataFrame,
    sim,
    year: int,
    dsa_maximum: float,
    policy_year: int | None = None,
) -> pd.DataFrame:
    """Move specific modelled grants out of the generic education-grant residual.

    PLA, ADG, and Childcare Grant remain formula-driven because they are
    calibration targets. Their modelled capacity is only used to avoid also
    counting the same reported FRS grant amount in the generic residual.
    DSA lacks a modelled amount signal, so its allocation seeds eligible
    expenses directly where the DSA parameter is available.

    ``year`` is the survey year the grant capacities are evaluated at.
    ``policy_year`` (default ``year``) is the first policy year the dataset
    will be simulated at; it gates the DSA seed, which policyengine-uk only
    models from ``DISABLED_STUDENTS_ALLOWANCE_FIRST_MODELED_YEAR``.
    """

    if policy_year is None:
        policy_year = year

    grant_capacities = {
        variable: sim.calculate(variable, year)
        for variable in FORMULA_MODELED_EDUCATION_GRANT_VARIABLES
    }
    grant_capacities[DISABLED_STUDENTS_ALLOWANCE_EXPENSE_INPUT] = (
        calculate_disabled_students_allowance_reported_grant_capacity(
            sim, policy_year, dsa_maximum
        )
    )
    allocations = allocate_reported_education_grants(
        pe_person["education_grants"], grant_capacities
    )

    pe_person["education_grants"] = allocations["education_grants"]
    pe_person[DISABLED_STUDENTS_ALLOWANCE_EXPENSE_INPUT] = allocations[
        DISABLED_STUDENTS_ALLOWANCE_EXPENSE_INPUT
    ]

    return pe_person


FRS_RELEASE_FOLDER_PATTERN = re.compile(r"^frs_(\d{4})_(\d{2})$")


def survey_year_from_frs_folder_name(raw_frs_folder) -> int | None:
    """Survey year encoded in an FRS release folder name (``frs_2024_25`` -> 2024).

    Returns ``None`` for folders outside the release naming convention, such
    as synthetic fixtures in tests.
    """
    match = FRS_RELEASE_FOLDER_PATTERN.match(Path(raw_frs_folder).name)
    if match is None:
        return None
    return int(match.group(1))


def validate_frs_survey_year(raw_frs_folder, year: int) -> None:
    """Refuse a ``year`` that does not match the FRS release being read.

    ``year`` stamps the dataset's ``time_period``, selects vintage-dependent
    columns, and thresholds reported benefit amounts against that fiscal
    year's rates. Nothing in ``create_frs`` uprates, so passing the survey
    year plus one asserts the survey's amounts as the next year's and
    mis-thresholds every disability category and flag (uk-data#477). The
    folder name is the release identity available here: ``frs_2024_25`` is
    survey year 2024. A folder outside that convention (a synthetic fixture,
    an ad hoc extraction) cannot be checked, so the function warns and
    returns rather than guessing.
    """
    folder_survey_year = survey_year_from_frs_folder_name(raw_frs_folder)
    if folder_survey_year is None:
        warnings.warn(
            f"FRS folder {Path(raw_frs_folder).name!r} does not follow the "
            f"frs_YYYY_YY release naming, so year={year} cannot be checked "
            "against the release being read.",
            stacklevel=2,
        )
        return
    if int(year) != folder_survey_year:
        raise ValueError(
            f"FRS folder {Path(raw_frs_folder).name!r} is survey year "
            f"{folder_survey_year} (fiscal year {folder_survey_year}/"
            f"{(folder_survey_year + 1) % 100:02d}) but year={year} was passed. "
            "`year` is the survey year and stamps time_period; it does not "
            "uprate. Build with the survey year and uprate the saved dataset "
            "with `uprate_dataset` instead."
        )


def derive_pension_credit_reported_capital(benunit: pd.DataFrame) -> np.ndarray:
    """Each benefit unit's capital as the FRS records it, for Pension Credit.

    Uses ``TOTCAPB4``, DWP's derived benefit-unit total of the adults' savings
    and investments, which its below-average-resources statistics use in place
    of ``TOTCAPB3`` since it became available in 2019/20; ``TOTCAPB3`` is the
    fallback for earlier survey years. Pension Credit counts the claimant's
    capital and, under the State Pension Credit Act 2002 s. 5, the partner's,
    and this is a benefit-unit measure. It is an approximation of Pension
    Credit capital, not the assessed figure: it covers financial assets only
    (second homes and land, which Pension Credit also counts, are not in it),
    and no Schedule V disregard or reg. 19 valuation is applied to it. The
    household wealth imputation instead draws a household's wealth from Wealth
    and Assets Survey households with similar income, composition, tenure and
    region, with no information on means-tested receipt, and policyengine-uk
    spreads it over the household's pension-age adults.

    A missing or negative value gives -1, so policyengine-uk falls back to the
    household proxy.
    """
    capital = np.full(len(benunit), np.nan)
    for column in ("totcapb3", "totcapb4"):  # later columns take precedence
        if column in benunit.columns:
            # Plain float64, so nullable (pd.NA) inputs become NaN and the
            # validity mask is a plain bool array with no missing entries.
            values = (
                pd.to_numeric(benunit[column], errors="coerce")
                .astype("float64")
                .to_numpy(dtype=float, na_value=np.nan)
            )
            valid = np.isfinite(values) & (values >= 0)
            capital = np.where(valid, values, capital)
    return np.where(np.isfinite(capital) & (capital >= 0), capital, -1.0)


def create_frs(
    raw_frs_folder: str,
    year: int,
    include_internal_disability_reported_amounts: bool = False,
    policy_year: int | None = None,
) -> UKSingleYearDataset:
    """
    Process raw FRS data into PolicyEngine UK dataset format.

    Transforms the Family Resources Survey microdata from raw tab-delimited
    files into a structured PolicyEngine UK dataset with person, benefit unit,
    and household-level variables mapped to the appropriate tax-benefit system
    variables.

    Args:
        raw_frs_folder: Path to folder containing raw FRS .tab files.
        year: Survey year for the dataset: the fiscal year the fieldwork
            covers (2024 for FRS 2024-25). It stamps ``time_period``, selects
            vintage-dependent survey columns, and thresholds reported benefit
            amounts against that fiscal year's rates. It must match the
            release folder being read; nothing here uprates.
        include_internal_disability_reported_amounts: Keep raw disability
            benefit amount intermediates for downstream imputation. Public
            saved datasets should leave this as ``False``.
        policy_year: First policy year the dataset will be simulated at (the
            release's calibration year in the build). Gates seeds for
            programmes policyengine-uk models from a later year than the
            survey, currently Disabled Students' Allowance. Defaults to
            ``year``.

    Returns:
        UKSingleYearDataset with processed FRS data ready for policy simulation.
    """
    raw_folder = Path(raw_frs_folder)
    if not raw_folder.exists():
        raise FileNotFoundError(f"Raw folder {raw_folder} does not exist.")
    validate_frs_survey_year(raw_folder, year)
    if policy_year is None:
        policy_year = year
    if int(policy_year) < int(year):
        raise ValueError(
            f"policy_year={policy_year} precedes survey year={year}; the dataset "
            "cannot be simulated at a policy year before its survey year."
        )

    frs = {}
    # Store SALSAC values before numeric conversion (for salary sacrifice
    # imputation)
    job_salsac_raw = None
    for file in raw_folder.glob("*.tab"):
        table_name = file.stem
        # Read raw data first
        df_raw = pd.read_csv(file, sep="\t")
        df_raw.columns = df_raw.columns.str.lower()

        # Preserve SALSAC column from job table before numeric conversion
        # SALSAC indicates salary sacrifice participation:
        # '1' = Yes, '2' = No, ' ' or blank = skip/not asked
        if table_name == "job" and "salsac" in df_raw.columns:
            job_salsac_raw = df_raw["salsac"].copy()

        # Make numeric where possible
        df = df_raw.apply(pd.to_numeric, errors="coerce")

        # Standardise column names to lower case (already done above)
        # df.columns = df.columns.str.lower()

        # Edit ID variables for simplicity
        if "sernum" in df.columns:
            df.rename(columns={"sernum": "household_id"}, inplace=True)

        if "benunit" in df.columns:
            # In the tables, benunit is the index of the benefit unit *within* the household.
            df.rename(columns={"benunit": "benunit_id"}, inplace=True)
            df["benunit_id"] = (df["household_id"] * 1e2 + df["benunit_id"]).astype(int)

        if "person" in df.columns:
            df.rename(columns={"person": "person_id"}, inplace=True)
            df["person_id"] = (df["household_id"] * 1e3 + df["person_id"]).astype(int)

        frs[table_name] = df

    # Combine adult and child tables for convenience

    frs["person"] = pd.concat([frs["adult"], frs["child"]]).sort_index().fillna(0)

    person = frs["person"]
    # Sort by benunit_id for the same reason the household table is sorted
    # below: the model declares benunit entities in sorted-id order
    # (np.unique(benunit_id)), so if the raw table order differs (the case
    # from the 2024-25 FRS onward), every benunit-level variable — including
    # benunit_id itself — lands on the wrong benefit unit.
    benunit = frs["benunit"].sort_values("benunit_id").reset_index(drop=True)
    household = frs["househol"]
    # Sort by household_id so positional reads below (e.g.
    # `household.gross4.values`) align with `pe_household["household_id"]`,
    # which is built from the sorted unique person household ids. Without this
    # the household grossing weight and every other household-level variable
    # land on the wrong household whenever the raw FRS household table is not
    # already ordered by sernum (the case from the 2024-25 FRS onward),
    # scrambling weights and collapsing the population.
    household = household.set_index("household_id").sort_index()
    pension = frs["pension"]
    oddjob = frs["oddjob"]
    account = frs["accounts"]
    job = frs["job"]
    # Add raw SALSAC column to job table for salary sacrifice imputation
    # SALSAC values: '1' = Yes (participates), '2' = No, ' '/blank = not asked
    if job_salsac_raw is not None:
        job["salsac_raw"] = job_salsac_raw.values
    benefits = frs["benefits"]
    maintenance = frs["maint"]
    pen_prov = frs["penprov"]
    childcare = frs["chldcare"]
    extchild = frs["extchild"]
    mortgage = frs["mortgage"]

    pe_person = pd.DataFrame()
    pe_benunit = pd.DataFrame()
    pe_household = pd.DataFrame()

    # Add primary and foreign keys
    pe_person["person_id"] = person.person_id
    pe_person["person_benunit_id"] = person.benunit_id
    pe_person["person_household_id"] = person.household_id
    pe_benunit["benunit_id"] = benunit.benunit_id
    pe_household["household_id"] = person.household_id.sort_values().unique()

    # Add grossing weights
    pe_household["household_weight"] = household.gross4.values

    # Add basic personal variables
    age = person.age80 + person.age
    pe_person["age"] = age
    # birth_year should be calculated from age and period in the model,
    # not stored as static data (see PolicyEngine/policyengine-uk#1352)
    # Age fields are AGE80 (top-coded) and AGE in the adult and child tables, respectively.
    pe_person["gender"] = np.where(person.sex == 1, "MALE", "FEMALE")
    pe_person["hours_worked"] = np.maximum(person.tothours, 0) * 52
    pe_person["is_household_head"] = person.hrpid == 1
    pe_person["is_benunit_head"] = person.uperson == 1
    dependent_children = (
        benunit.depchldb
        if "depchldb" in benunit
        else frs["child"]
        .groupby("benunit_id")
        .size()
        .reindex(benunit.benunit_id)
        .fillna(0)
        .to_numpy()
    )
    pe_person["is_parent"] = derive_is_parent_from_frs_microdata(
        person_ids=pe_person.person_id,
        person_benunit_ids=pe_person.person_benunit_id,
        adult_person_ids=frs["adult"].person_id,
        benunit_ids=pe_benunit.benunit_id,
        dependent_children=dependent_children,
    )
    pe_person["is_claimant_or_partner"] = (
        derive_is_claimant_or_partner_from_frs_microdata(
            person_ids=pe_person.person_id,
            person_benunit_ids=pe_person.person_benunit_id,
            adult_person_ids=frs["adult"].person_id,
        )
    )
    MARITAL = [
        "MARRIED",
        "SINGLE",
        "SINGLE",
        "WIDOWED",
        "SEPARATED",
        "DIVORCED",
    ]
    pe_person["marital_status"] = categorical(
        person.marital, 2, range(1, 7), MARITAL
    ).fillna("SINGLE")

    # Add education levels
    if "fted" in person.columns:
        fted = person.fted
    else:
        fted = person.educft  # Renamed in FRS 2022-23
    typeed2 = person.typeed2

    def determine_education_level(fted_val, typeed2_val, age_val):
        # By default, not in education
        if fted_val in (2, -1, 0):
            return "NOT_IN_EDUCATION"
        # In pre-primary
        elif typeed2_val == 1:
            return "PRE_PRIMARY"
        # In primary education
        elif (
            typeed2_val in (2, 4)
            or (typeed2_val in (3, 8) and age_val < 11)
            or (typeed2_val == 0 and fted_val == 1 and age_val > 5 and age_val < 11)
        ):
            return "PRIMARY"
        # In lower secondary
        elif (
            typeed2_val in (5, 6)
            or (typeed2_val in (3, 8) and age_val >= 11 and age_val <= 16)
            or (typeed2_val == 0 and fted_val == 1 and age_val <= 16)
        ):
            return "LOWER_SECONDARY"
        # In upper secondary
        elif (
            typeed2_val == 7
            or (typeed2_val in (3, 8) and age_val > 16)
            or (typeed2_val == 0 and fted_val == 1 and age_val > 16)
        ):
            return "UPPER_SECONDARY"
        # In post-secondary
        elif typeed2_val in (7, 8) and age_val >= 19:
            return "POST_SECONDARY"
        # In tertiary
        elif typeed2_val == 9 or (typeed2_val == 0 and fted_val == 1 and age_val >= 19):
            return "TERTIARY"
        else:
            return "NOT_IN_EDUCATION"

    # Apply the function to determine education level
    pe_person["current_education"] = pd.Series(
        [determine_education_level(f, t, a) for f, t, a in zip(fted, typeed2, age)],
        index=pe_person.index,
    )
    pe_person["is_in_non_advanced_education"] = derive_is_in_non_advanced_education(
        pe_person.current_education
    )
    pe_person["is_in_approved_training"] = (
        derive_is_in_approved_training_from_frs_person(person)
    )
    pe_person["age_started_or_accepted_current_education_or_training"] = (
        derive_age_started_or_accepted_current_education_or_training(
            age,
            pe_person.is_in_non_advanced_education,
            pe_person.is_in_approved_training,
        )
    )
    pe_person["is_before_universal_credit_qualifying_young_person_terminal_date"] = (
        derive_is_before_universal_credit_qualifying_young_person_terminal_date(
            age,
            pe_person.is_in_non_advanced_education,
            pe_person.is_in_approved_training,
        )
    )

    # Add highest education from EDUCQUAL (highest qualification achieved)
    # Codes from FRS ADT_324X classification; unmapped codes default to UPPER_SECONDARY
    EDUCQUAL_MAP = {
        1: "NOT_COMPLETED_PRIMARY",
        2: "LOWER_SECONDARY",  # GCSE D-G / CSE 2-5
        3: "LOWER_SECONDARY",  # GCSE A-C / O-level A-C
        4: "UPPER_SECONDARY",  # AS-level
        5: "UPPER_SECONDARY",  # A-level (1 subject)
        6: "UPPER_SECONDARY",  # A-level (2 subjects)
        7: "UPPER_SECONDARY",  # A-level (3+ subjects)
        8: "LOWER_SECONDARY",  # Scottish Standard/Ordinary Grade
        9: "UPPER_SECONDARY",  # Scottish Higher Grade
        10: "UPPER_SECONDARY",  # Scottish 6th Year Studies
        11: "POST_SECONDARY",  # HNC/HND
        12: "POST_SECONDARY",  # City & Guilds advanced / BTEC National
        13: "UPPER_SECONDARY",  # City & Guilds craft / BTEC General
        14: "POST_SECONDARY",  # ONC/OND / BTEC National (lower)
        15: "UPPER_SECONDARY",  # City & Guilds foundation
        16: "POST_SECONDARY",  # RSA advanced
        17: "TERTIARY",  # First/foundation degree
        18: "TERTIARY",  # Second degree
        19: "TERTIARY",  # Higher degree (Masters/PhD)
        20: "TERTIARY",  # PGCE / teaching qualification
        21: "TERTIARY",  # Nursing/paramedical qualification
        66: "UPPER_SECONDARY",  # NVQ/SVQ Level 1
        67: "UPPER_SECONDARY",  # NVQ/SVQ Level 2
        68: "UPPER_SECONDARY",  # NVQ/SVQ Level 3
        69: "POST_SECONDARY",  # NVQ/SVQ Level 4
        70: "TERTIARY",  # NVQ/SVQ Level 5
    }
    # Codes 22-65 and 71-85 are further vocational/professional qualifications;
    # treat as POST_SECONDARY. Codes 86-87 are catch-alls; treat as UPPER_SECONDARY.
    for code in range(22, 66):
        EDUCQUAL_MAP[code] = "POST_SECONDARY"
    for code in range(71, 86):
        EDUCQUAL_MAP[code] = "POST_SECONDARY"
    EDUCQUAL_MAP[86] = "UPPER_SECONDARY"
    EDUCQUAL_MAP[87] = "UPPER_SECONDARY"

    educqual = pd.to_numeric(person.educqual, errors="coerce")
    pe_person["highest_education"] = educqual.map(EDUCQUAL_MAP).fillna(
        "UPPER_SECONDARY"
    )

    # Add employment status
    EMPLOYMENTS = [
        "CHILD",
        "FT_EMPLOYED",
        "PT_EMPLOYED",
        "FT_SELF_EMPLOYED",
        "PT_SELF_EMPLOYED",
        "UNEMPLOYED",
        "RETIRED",
        "STUDENT",
        "CARER",
        "LONG_TERM_DISABLED",
        "SHORT_TERM_DISABLED",
    ]
    pe_person["employment_status"] = categorical(
        person.empstati, 1, range(12), EMPLOYMENTS
    ).fillna("LONG_TERM_DISABLED")

    # Add employer sector of the main job from FRS `mjobsect`
    # (1 = private, 2 = public; missing/blank = not in paid work).
    EMPLOYMENT_SECTORS = ["NOT_EMPLOYED", "PRIVATE", "PUBLIC"]
    pe_person["employment_sector"] = categorical(
        pd.to_numeric(person.mjobsect, errors="coerce"),
        0,
        [0, 1, 2],
        EMPLOYMENT_SECTORS,
    ).fillna("NOT_EMPLOYED")

    # Standard Industrial Classification (2007) division of the main job from
    # FRS `sic` (0 if unknown; 84 = public administration and defence).
    pe_person["sic_industry_division"] = (
        pd.to_numeric(person.sic, errors="coerce").fillna(0).clip(lower=0).astype(int)
    )

    REGIONS = [
        "NORTH_EAST",
        "NORTH_WEST",
        "YORKSHIRE",
        "EAST_MIDLANDS",
        "WEST_MIDLANDS",
        "EAST_OF_ENGLAND",
        "LONDON",
        "SOUTH_EAST",
        "SOUTH_WEST",
        "WALES",
        "SCOTLAND",
        "NORTHERN_IRELAND",
        "UNKNOWN",
    ]
    pe_household["region"] = categorical(
        household.gvtregno, 14, [1, 2] + list(range(4, 15)), REGIONS
    ).values
    TENURES = [
        "RENT_FROM_COUNCIL",
        "RENT_FROM_HA",
        "RENT_PRIVATELY",
        "RENT_PRIVATELY",
        "OWNED_OUTRIGHT",
        "OWNED_WITH_MORTGAGE",
    ]
    pe_household["tenure_type"] = categorical(
        household.ptentyp2, 3, range(1, 7), TENURES
    ).values
    frs["num_bedrooms"] = household.bedroom6
    ACCOMMODATIONS = [
        "HOUSE_DETACHED",
        "HOUSE_SEMI_DETACHED",
        "HOUSE_TERRACED",
        "FLAT",
        "CONVERTED_HOUSE",
        "MOBILE",
        "OTHER",
    ]
    pe_household["accommodation_type"] = categorical(
        household.typeacc, 1, range(1, 8), ACCOMMODATIONS
    ).values

    # Impute Council Tax

    # In Scotland, council tax bills are collected together with Scottish
    # Water and sewerage charges, and the FRS CTANNUAL variable includes
    # them. Net them off (they are weekly variables; CTANNUAL is annual) so
    # council_tax is tax only: the water charges are already captured
    # separately in water_and_sewerage_charges, so leaving them in both
    # double-counts them and overstates Scottish council tax by roughly
    # £500 per household (~25% of the Scottish total).
    SCOTLAND_GVTREGNO = 12
    scottish_water_annual = pd.Series(
        np.where(
            household.gvtregno == SCOTLAND_GVTREGNO,
            (
                np.maximum(household.csewamt.fillna(0), 0)
                + np.maximum(household.cwatamtd.fillna(0), 0)
            )
            * (365.25 / 7),
            0,
        ),
        index=household.index,
    )
    ctannual_tax_only = np.maximum(household.ctannual - scottish_water_annual, 0)

    # Only ~25% of household report Council Tax bills - use
    # these to build a model to impute missing values
    CT_valid = household.ctannual > 0

    # Find the mean reported Council Tax bill for a given
    # (region, CT band, is-single-person-household) triplet
    region = household.gvtregno[CT_valid]
    band = household.ctband[CT_valid]
    single_person = (household.adulth == 1)[CT_valid]
    ctannual = ctannual_tax_only[CT_valid]

    # Build the table
    ct_mean = ctannual.groupby([region, band, single_person], dropna=False).mean()
    ct_mean = ct_mean.replace(-1, ct_mean.mean())

    # For every household consult the table to find the imputed
    # Council Tax bill
    pairs = household.set_index(
        [household.gvtregno, household.ctband, (household.adulth == 1)]
    )
    hh_CT_mean = pd.Series(index=pairs.index)
    has_mean = pairs.index.isin(ct_mean.index)
    hh_CT_mean[has_mean] = ct_mean[pairs.index[has_mean]].values
    hh_CT_mean[~has_mean] = 0
    ct_imputed = hh_CT_mean

    # For households which originally reported Council Tax,
    # use the reported value. Otherwise, use the imputed value
    council_tax = pd.Series(
        np.where(
            # 2018 FRS uses blanks for missing values, 2019 FRS
            # uses -1 for missing values
            (household.ctannual < 0) | household.ctannual.isna(),
            np.maximum(ct_imputed, 0).values,
            ctannual_tax_only,
        )
    )
    pe_household["council_tax"] = council_tax.fillna(0)
    BANDS = ["A", "B", "C", "D", "E", "F", "G", "H", "I"]
    # Band 1 is the most common
    pe_household["council_tax_band"] = (
        categorical(household.ctband, 1, range(1, 10), BANDS).fillna("D").values
    )
    # Domestic rates variables are all weeklyised, unlike Council Tax variables (despite the variable name suggesting otherwise)
    if year < 2021:
        DOMESTIC_RATES_VARIABLE = "rtannual"
    else:
        DOMESTIC_RATES_VARIABLE = "niratlia"
    pe_household["domestic_rates"] = (
        np.select(
            [
                household[DOMESTIC_RATES_VARIABLE] >= 0,
                household.rt2rebam >= 0,
                True,
            ],
            [
                household[DOMESTIC_RATES_VARIABLE],
                household.rt2rebam,
                0,
            ],
        )
        * 52
    ).astype(float)

    WEEKS_IN_YEAR = 365.25 / 7

    pe_person["employment_income"] = np.maximum(0, person.inearns) * WEEKS_IN_YEAR

    pension_payment = sum_to_entity(
        pension.penpay * (pension.penpay > 0),
        pension.person_id,
        person.person_id,
    )
    pension_tax_paid = sum_to_entity(
        (pension.ptamt * ((pension.ptinc == 2) & (pension.ptamt > 0))),
        pension.person_id,
        person.person_id,
    )
    pension_deductions_removed = sum_to_entity(
        pension.poamt
        * (((pension.poinc == 2) | (pension.penoth == 1)) & (pension.poamt > 0)),
        pension.person_id,
        person.person_id,
    )

    pe_person["private_pension_income"] = (
        pension_payment + pension_tax_paid + pension_deductions_removed
    ) * WEEKS_IN_YEAR

    pe_person["self_employment_income"] = np.maximum(0, person.seincam2) * WEEKS_IN_YEAR
    # policyengine-uk releases without this input skip the column and keep
    # their formula.
    pe_person["uc_is_in_gainful_self_employment"] = (
        derive_uc_is_in_gainful_self_employment(
            pe_person.employment_status,
            pe_person.self_employment_income,
            pe_person.employment_income,
        )
    )

    INVERTED_BASIC_RATE = 1.25

    pe_person["tax_free_savings_income"] = np.maximum(
        0,
        sum_to_entity(
            account.accint * (account.account == 21),
            account.person_id,
            person.person_id,
        )
        * WEEKS_IN_YEAR,
    )
    taxable_savings_interest = (
        sum_to_entity(
            (account.accint * np.where(account.acctax == 1, INVERTED_BASIC_RATE, 1))
            * (account.account.isin((1, 3, 5, 27, 28))),
            account.person_id,
            person.person_id,
        )
        * WEEKS_IN_YEAR
    )
    pe_person["savings_interest_income"] = np.maximum(
        0,
        taxable_savings_interest + pe_person["tax_free_savings_income"].values,
    )
    pe_person["dividend_income"] = np.maximum(
        0,
        sum_to_entity(
            (account.accint * np.where(account.invtax == 1, INVERTED_BASIC_RATE, 1))
            * (
                ((account.account == 6) & (account.invtax == 1))  # GGES
                | account.account.isin((7, 8))  # Stocks/shares/UITs
            ),
            account.person_id,
            person.index,
        )
        * 52,
    )
    pe_person["property_income"] = frs_property_income(person, household)
    maintenance_to_self = np.maximum(
        pd.Series(np.where(person.mntus1 == 2, person.mntusam1, person.mntamt1)).fillna(
            0
        ),
        0,
    )
    maintenance_from_dwp = person.mntamt2
    pe_person["maintenance_income"] = (
        sum_positive_variables([maintenance_to_self, maintenance_from_dwp])
        * WEEKS_IN_YEAR
    )

    odd_job_income = sum_to_entity(
        oddjob.ojamt * (oddjob.ojnow == 1), oddjob.person_id, person.person_id
    )

    MISC_INCOME_FIELDS = [
        "allpay2",
        "royyr2",
        "royyr3",
        "royyr4",
        "chamtern",
        "chamttst",
    ]

    pe_person["miscellaneous_income"] = (
        odd_job_income + sum_from_positive_fields(person, MISC_INCOME_FIELDS)
    ) * WEEKS_IN_YEAR

    PRIVATE_TRANSFER_INCOME_FIELDS = [
        "apamt",
        "apdamt",
        "pareamt",
        "allpay2",
        "allpay3",
        "allpay4",
    ]

    pe_person["private_transfer_income"] = (
        sum_from_positive_fields(person, PRIVATE_TRANSFER_INCOME_FIELDS) * WEEKS_IN_YEAR
    )

    pe_person["lump_sum_income"] = person.redamt

    pe_person["student_loan_repayments"] = person.slrepamt * WEEKS_IN_YEAR

    BENEFIT_CODES = dict(
        child_benefit=3,
        income_support=19,
        housing_benefit=94,
        attendance_allowance=12,
        dla_sc=1,
        dla_m=2,
        iidb=15,
        carers_allowance=13,
        sda=10,
        afcs=8,
        ssmg=22,
        pension_credit=4,
        child_tax_credit=91,
        working_tax_credit=90,
        state_pension=5,
        winter_fuel_allowance=62,
        incapacity_benefit=17,
        universal_credit=95,
        pip_m=97,
        pip_dl=96,
    )
    for benefit, code in BENEFIT_CODES.items():
        pe_person[benefit + "_reported"] = (
            sum_to_entity(
                benefits.benamt * (benefits.benefit == code),
                benefits.person_id.values,
                person.person_id,
            )
            * WEEKS_IN_YEAR
        )

    pe_person = add_disability_benefit_categories_from_reported_amounts(
        pe_person,
        year,
        inplace=True,
    )

    pe_person["jsa_contrib_reported"] = (
        sum_to_entity(
            benefits.benamt * (benefits.var2.isin((1, 3))) * (benefits.benefit == 14),
            benefits.person_id,
            person.person_id,
        )
        * WEEKS_IN_YEAR
    )
    pe_person["jsa_income_reported"] = (
        sum_to_entity(
            benefits.benamt * (benefits.var2.isin((2, 4))) * (benefits.benefit == 14),
            benefits.person_id,
            person.person_id,
        )
        * WEEKS_IN_YEAR
    )
    pe_person["esa_contrib_reported"] = (
        sum_to_entity(
            benefits.benamt * (benefits.var2.isin((1, 3))) * (benefits.benefit == 16),
            benefits.person_id,
            person.person_id,
        )
        * WEEKS_IN_YEAR
    )
    pe_person["esa_income_reported"] = (
        sum_to_entity(
            benefits.benamt * (benefits.var2.isin((2, 4))) * (benefits.benefit == 16),
            benefits.person_id,
            person.person_id,
        )
        * WEEKS_IN_YEAR
    )
    pe_person["receives_benefits_in_own_right"] = derive_receives_benefits_in_own_right(
        pe_person
    )

    pe_person["bsp_reported"] = (
        sum_to_entity(
            benefits.benamt * (benefits.benefit.isin((6, 9))),
            benefits.person_id,
            person.person_id,
        )
        * WEEKS_IN_YEAR
    )

    pe_person["winter_fuel_allowance_reported"] /= WEEKS_IN_YEAR

    pe_person["statutory_sick_pay"] = person.sspadj * WEEKS_IN_YEAR
    pe_person["statutory_maternity_pay"] = person.smpadj * WEEKS_IN_YEAR

    pe_person["student_loans"] = np.maximum(person.tuborr, 0)
    if "adema" not in person.columns:
        person["adema"] = person.eduma
        person["ademaamt"] = person.edumaamt
    pe_person["adult_ema"] = fill_with_mean(person, "adema", "ademaamt")
    pe_person["child_ema"] = fill_with_mean(person, "chema", "chemaamt")

    pe_person["access_fund"] = np.maximum(person.accssamt, 0) * WEEKS_IN_YEAR

    pe_person["education_grants"] = np.maximum(
        person[["grtdir1", "grtdir2"]].sum(axis=1), 0
    )

    pe_person["council_tax_benefit_reported"] = np.maximum(
        (person.hrpid == 1)
        * pd.Series(
            household.ctrebamt[person.household_id.values].values,
            index=person.person_id,
        )
        .fillna(0)
        .values
        * WEEKS_IN_YEAR,
        0,
    )

    pe_person["healthy_start_vouchers"] = person.heartval * WEEKS_IN_YEAR

    pe_person["free_school_breakfasts"] = person.fsbval * WEEKS_IN_YEAR
    pe_person["free_school_fruit_veg"] = person.fsfvval * WEEKS_IN_YEAR
    pe_person["free_school_meals"] = person.fsmval * WEEKS_IN_YEAR

    pe_person["maintenance_expenses"] = (
        pd.Series(
            np.where(maintenance.mrus == 2, maintenance.mruamt, maintenance.mramt)
        )
        .groupby(maintenance.person_id)
        .sum()
        .reindex(person.person_id)
        .fillna(0)
        .values
        * WEEKS_IN_YEAR
    )
    pe_household["rent"] = household.hhrent.fillna(0).values * WEEKS_IN_YEAR
    pe_household["mortgage_interest_repayment"] = (
        household.mortint.fillna(0).values * WEEKS_IN_YEAR
    )
    mortgage_capital = np.where(mortgage.rmort == 1, mortgage.rmamt, mortgage.borramt)
    mortgage_capital_repayment = sum_to_entity(
        mortgage_capital / mortgage.mortend,
        mortgage.household_id,
        household.index,
    )
    pe_household["mortgage_capital_repayment"] = mortgage_capital_repayment

    pe_person["childcare_expenses"] = (
        sum_to_entity(
            childcare.chamt * (childcare.cost == 1) * (childcare.registrd == 1),
            childcare.person_id,
            person.person_id,
        )
        * 52
    )

    pe_person["personal_pension_contributions"] = np.maximum(
        0,
        sum_to_entity(
            pen_prov.penamt[pen_prov.stemppen.isin((5, 6))],
            pen_prov.person_id,
            person.person_id,
        ).clip(0, pen_prov.penamt.quantile(0.95))
        * WEEKS_IN_YEAR,
    )
    pe_person["employee_pension_contributions"] = np.maximum(
        0,
        sum_to_entity(job.deduc1.fillna(0), job.person_id, person.person_id)
        * WEEKS_IN_YEAR,
    )
    pe_person["employer_pension_contributions"] = (
        pe_person["employee_pension_contributions"] * 3
    )  # Rough estimate based on aggregates.
    # Salary sacrifice pension contributions from FRS Job table (SPNAMT field)
    # SPNAMT represents employer pension contributions made via salary sacrifice
    # arrangements where employees forego salary in exchange for increased pension
    # contributions. This is separate from regular employee pension contributions
    # (deduc1) and provides tax advantages for both employer and employee.
    # Uses same pattern as employee_pension_contributions (deduc1) without outlier
    # clipping, as job-level data is generally cleaner than pension provider data.
    # Source: https://datacatalogue.ukdataservice.ac.uk/datasets/dataset/630d4a8d-ba6a-82b3-f33d-c713c66efcb3
    # Note: Values are annualized from weekly amounts reported in the survey.
    pe_person["pension_contributions_via_salary_sacrifice"] = np.maximum(
        0,
        sum_to_entity(job.spnamt.fillna(0), job.person_id, person.person_id)
        * WEEKS_IN_YEAR,
    )

    # Salary sacrifice participation indicator from SALSAC field
    # Used for imputation: 1 = Yes, 0 = No, -1 = not asked (skip)
    # This allows distinguishing between explicit No responses and
    # respondents who were not asked the question (imputation candidates)
    if "salsac_raw" in job.columns:
        salsac_numeric = (
            job["salsac_raw"].map({"1": 1, "2": 0, " ": -1}).fillna(-1).astype(int)
        )
        # Aggregate to person level: take max (any job with SS = person has SS)
        pe_person["salary_sacrifice_reported"] = np.clip(
            sum_to_entity(
                (salsac_numeric == 1).astype(int),
                job.person_id,
                person.person_id,
            ),
            0,
            1,
        )
        # Track if person was asked about SS in any job (for imputation)
        pe_person["salary_sacrifice_asked"] = np.clip(
            sum_to_entity(
                (salsac_numeric >= 0).astype(int),
                job.person_id,
                person.person_id,
            ),
            0,
            1,
        )
    else:
        # If SALSAC not available, mark all as not asked
        pe_person["salary_sacrifice_reported"] = 0
        pe_person["salary_sacrifice_asked"] = 0

    pe_household["housing_service_charges"] = (
        pd.DataFrame(
            [
                household[f"chrgamt{i}"] * (household[f"chrgamt{i}"] > 0)
                for i in range(1, 10)
            ]
        )
        .sum()
        .values
        * WEEKS_IN_YEAR
    )
    pe_household["structural_insurance_payments"] = (
        household.struins.values * WEEKS_IN_YEAR
    )
    pe_household["water_and_sewerage_charges"] = (
        pd.Series(
            np.where(
                household.gvtregno == 12,
                household.csewamt + household.cwatamtd,
                household.watsewrt,
            )
        )
        .fillna(0)
        .values
        * WEEKS_IN_YEAR
    )

    pe_household["external_child_payments"] = sum_to_entity(
        extchild.nhhamt * WEEKS_IN_YEAR,
        extchild.household_id,
        household.index,
    )

    dataset = UKSingleYearDataset(
        person=pe_person,
        benunit=pe_benunit,
        household=pe_household,
        fiscal_year=year,
    )

    # Randomly select broad rental market areas from regions.
    from policyengine_uk import Microsimulation

    sim = Microsimulation(dataset=dataset)
    region = sim.populations["benunit"].household("region", dataset.time_period)
    lha_category = np.asarray(sim.calculate("LHA_category", year))

    # Draw each benefit unit's BRMA in proportion to the private-rented
    # households in each of its region's BRMAs with the matching number of
    # bedrooms. Use a seeded generator so the assignment is reproducible.
    brma_rng = np.random.default_rng(0)
    brma = assign_brmas(region, lha_category, brma_rng)

    household_brma = pick_household_brmas(
        brma,
        sim.populations["benunit"].household("household_id", dataset.time_period),
        brma_rng,
    )
    pe_household["brma"] = household_brma[sim.calculate("household_id")].values

    pe_person = add_disability_benefit_flags_from_reported_amounts(
        pe_person,
        year,
        inplace=True,
    )

    # Dataset-side claimant-state approximations for future legacy ESA/JSA
    # modelling. These are explicit proxies based on observed survey
    # conditions, not legislative determinations.
    pe_person = attach_legacy_benefit_proxies_from_frs_person(
        pe_person, person, sim, year
    )

    if (pe_person["education_grants"] > 0).any():
        student_support_dataset = UKSingleYearDataset(
            person=pe_person,
            benunit=pe_benunit,
            household=pe_household,
            fiscal_year=year,
        )
        student_support_sim = Microsimulation(dataset=student_support_dataset)
        dsa_maximum = student_support_sim.tax_benefit_system.parameters(
            policy_year
        ).gov.dfe.disabled_students_allowance.maximum
        pe_person = split_reported_education_grants(
            pe_person,
            student_support_sim,
            year,
            dsa_maximum,
            policy_year=policy_year,
        )

    # Generate stochastic take-up decisions
    # All randomness is generated here in the data package using take-up rates
    # stored in YAML parameter files. This keeps the country package purely
    # deterministic.

    generator = np.random.default_rng(seed=100)

    # Load take-up rates from parameter files
    marriage_allowance_rate = load_take_up_rate("marriage_allowance", year)
    child_benefit_opts_out_rate = load_take_up_rate("child_benefit_opts_out_rate", year)
    tfc_rate = load_take_up_rate("tax_free_childcare", year)
    tfc_spend_routed_share = load_parameter(
        "stochastic", "tax_free_childcare_spend_routed_share", year
    )
    extended_childcare_rate = load_take_up_rate("extended_childcare", year)
    universal_childcare_rate = load_take_up_rate("universal_childcare", year)
    targeted_childcare_rate = load_take_up_rate("targeted_childcare", year)
    scp_under_6_rate = load_take_up_rate("scp_under_6", year)
    scp_6_plus_rate = load_take_up_rate("scp_6_plus", year)

    # Generate take-up decisions by comparing random draws to take-up rates,
    # anchored to reported receipts where the FRS captures them. Respondents
    # who report positive receipt of a benefit are assigned takeup=True with
    # certainty; the remaining non-reporters are filled probabilistically to
    # hit the aggregate target rate. See policyengine_uk_data/utils/takeup.py.
    # Person-level
    pe_person["would_claim_marriage_allowance"] = (
        generator.random(len(pe_person)) < marriage_allowance_rate
    )

    # Benefit unit-level — anchor on any adult in the benefit unit having
    # reported positive receipt in the FRS benefits table.
    pe_benunit["would_claim_child_benefit"] = assign_reported_takeup(
        pe_person,
        pe_benunit,
        "would_claim_child_benefit",
        year,
        generator.random(len(pe_benunit)),
    )
    pe_benunit["child_benefit_opts_out"] = (
        generator.random(len(pe_benunit)) < child_benefit_opts_out_rate
    )
    # The enhanced dataset redraws this once entitlement can be computed
    # (datasets/pension_credit_takeup.py).
    pe_benunit["would_claim_pc"] = assign_reported_takeup(
        pe_person, pe_benunit, "would_claim_pc", year, generator.random(len(pe_benunit))
    )
    # A benefit unit whose claimant and any partner have all reached State
    # Pension age cannot claim Universal Credit, so it never gets
    # would_claim_uc, even if it reports UC. The draw still covers every unit,
    # so the random stream and every other unit's value are unchanged. Ages
    # are not rolled forward, so if State Pension age rises above a claimant's
    # survey age in a later year, that unit stays without would_claim_uc there.
    pe_benunit["would_claim_uc"] = assign_reported_takeup(
        pe_person, pe_benunit, "would_claim_uc", year, generator.random(len(pe_benunit))
    ) & ~derive_all_claimants_over_state_pension_age(
        person_benunit_ids=sim.calculate("person_benunit_id", year).values,
        is_claimant_or_partner=sim.calculate(
            claimant_or_partner_variable(sim.tax_benefit_system.variables), year
        ).values,
        is_over_state_pension_age=sim.calculate("is_SP_age", year).values,
        benunit_ids=pe_benunit.benunit_id,
    )
    pe_benunit["would_claim_tfc"] = generator.random(len(pe_benunit)) < tfc_rate

    pe_benunit["would_claim_extended_childcare"] = (
        generator.random(len(pe_benunit)) < extended_childcare_rate
    )
    pe_benunit["would_claim_universal_childcare"] = (
        generator.random(len(pe_benunit)) < universal_childcare_rate
    )
    pe_benunit["would_claim_targeted_childcare"] = (
        generator.random(len(pe_benunit)) < targeted_childcare_rate
    )

    # Scottish Child Payment take-up at child level:
    # 97% for children under 6, 85% for children 6+
    # Source: gov.scot take-up rates publication Nov 2024
    # Note: We apply rates to all ages (not just <16) so that policy reforms
    # extending the age limit will have reasonable takeup assumptions.
    ages = pe_person["age"]
    scp_rate_per_person = np.where(ages < 6, scp_under_6_rate, scp_6_plus_rate)
    pe_person["would_claim_scp"] = (
        generator.random(len(pe_person)) < scp_rate_per_person
    )

    # Generate other stochastic variables using rates from parameter files
    tv_ownership_rate = load_parameter("stochastic", "tv_ownership_rate", year)
    tv_evasion_rate = load_parameter("stochastic", "tv_licence_evasion_rate", year)
    first_time_buyer_rate = load_parameter("stochastic", "first_time_buyer_rate", year)

    # Household-level: TV ownership
    pe_household["household_owns_tv"] = (
        generator.random(len(pe_household)) < tv_ownership_rate
    )

    # Household-level: TV licence evasion
    pe_household["would_evade_tv_licence_fee"] = (
        generator.random(len(pe_household)) < tv_evasion_rate
    )

    # Household-level: First home purchase
    pe_household["main_residential_property_purchased_is_first_home"] = (
        generator.random(len(pe_household)) < first_time_buyer_rate
    )

    # Person-level: Tie-breaking for higher earner (uniform random)
    # Tax-Free Childcare tops up money paid through the account, not a family's
    # whole childcare bill, and childcare_expenses is annual. Person-level
    # because a Tax-Free Childcare account is held for one child (Childcare
    # Payments Act 2014 section 15(2)) and the variable it feeds is per child.
    # See parameters/stochastic/tax_free_childcare_spend_routed_share.yaml.
    #
    # Checked rather than assumed: policyengine-uk's dataset loader skips
    # columns it does not recognise, so a model predating the variable would
    # silently drop this correction and fall back to the model default of 1 —
    # a build that looks clean and ships uncorrected Tax-Free Childcare
    # spending. The pyproject floor cannot express this yet because the
    # release carrying the variable does not exist at the time of writing, so
    # fail loudly here instead.
    require_variable(
        "tax_free_childcare_spend_routed_share",
        "Tax-Free Childcare routed-spend adjustment",
    )
    pe_person["tax_free_childcare_spend_routed_share"] = tfc_spend_routed_share

    pe_person["higher_earner_tie_break"] = generator.random(len(pe_person))

    # Person-level: Private school attendance random draw
    pe_person["attends_private_school_random_draw"] = generator.random(len(pe_person))

    # Generate extended childcare hours usage values. The mean and sd are a
    # modelling assumption, not a fitted result: without an extended spending
    # target the calibration objective cannot identify them. See
    # childcare/assumptions.py.
    extended_hours_values = generator.normal(
        EXTENDED_HOURS_MEAN, EXTENDED_HOURS_SD, len(pe_benunit)
    )
    # Clip values to be between 0 and 30 hours
    extended_hours_values = np.clip(extended_hours_values, 0, 30)

    # Add the maximum extended childcare hours usage
    pe_benunit["maximum_extended_childcare_hours_usage"] = extended_hours_values

    # Add marital status at the benefit unit level

    pe_benunit["is_married"] = benunit.famtypb2.isin([5, 7])

    # Pension Credit capital as the FRS records it for the benefit unit, in
    # place of the household wealth proxy (policyengine-uk
    # `pension_credit_reported_capital`).
    pe_benunit["pension_credit_reported_capital"] = (
        derive_pension_credit_reported_capital(benunit)
    )

    # Assign property_purchased to a share of households matching the UK
    # housing transaction rate, so only genuine purchasers are charged SDLT.
    #
    # This MUST be deterministic: a rules engine's inputs have to be
    # reproducible across builds. Use a seeded Generator (not global
    # np.random, whose state depends on whatever ran earlier in the build)
    # so the same FRS input always yields the same assignment. An unseeded
    # draw previously made the build non-reproducible and intermittently
    # spiked the first decile's effective tax rate.
    #
    # Sources:
    # - Transactions: HMRC 2024 - 1.1m/year
    #   https://www.gov.uk/government/statistics/monthly-property-transactions-completed-in-the-uk-with-value-40000-or-above
    # - Households: ONS 2024 - 28.6m
    #   https://www.ons.gov.uk/peoplepopulationandcommunity/birthsdeathsandmarriages/families/bulletins/familiesandhouseholds/2024
    # - Rate: 1.1m / 28.6m = 3.85%
    #
    # Verification against official SDLT revenue (2024-25):
    # - Official SDLT: £13.9bn (https://www.gov.uk/government/statistics/uk-stamp-tax-statistics)
    # - With 3.85% purchasers: £15.7bn (close to official)
    # - With every household a purchaser: £370bn (26x too high)
    PROPERTY_PURCHASE_RATE = 0.0385
    PROPERTY_PURCHASE_SEED = 0
    purchase_rng = np.random.default_rng(PROPERTY_PURCHASE_SEED)
    pe_household["property_purchased"] = (
        purchase_rng.random(len(pe_household)) < PROPERTY_PURCHASE_RATE
    )

    if not include_internal_disability_reported_amounts:
        pe_person = drop_internal_disability_reported_amounts(pe_person)

    dataset = UKSingleYearDataset(
        person=pe_person,
        benunit=pe_benunit,
        household=pe_household,
        fiscal_year=year,
    )

    return dataset


if __name__ == "__main__":
    from policyengine_uk_data.datasets.frs_release import CURRENT_FRS_RELEASE

    frs = create_frs(
        raw_frs_folder=STORAGE_FOLDER / CURRENT_FRS_RELEASE.name,
        year=CURRENT_FRS_RELEASE.survey_year,
        policy_year=CURRENT_FRS_RELEASE.calibration_year,
    )
    frs.save(STORAGE_FOLDER / CURRENT_FRS_RELEASE.base_dataset_file)
