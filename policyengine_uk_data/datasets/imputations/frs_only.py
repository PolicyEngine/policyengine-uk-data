"""Second-stage QRF imputation of FRS-only variables on SPI-donor rows.

The enhanced-FRS pipeline in :mod:`income` creates a zero-weight subsample
of the FRS that will be upweighted during calibration to fit SPI-derived
high-income targets. The first-stage QRF (trained on SPI) replaces only
the six core income components (plus ``gift_aid`` and
``charitable_investment_gifts``) on those rows. Every other FRS column —
benefit ``_reported`` values, pension contributions, savings, rent,
mortgage, council tax — stays at whatever the middle-income FRS donor
whose row was sampled happened to report.

That produces implausible joint distributions on the synthetic
high-income side. A row with imputed £2 M self-employment income carries
its donor's £120 UC ``_reported`` value, its donor's tiny pension
contribution, and its donor's typical rent. Under calibration upweight
these cascade into false benefit aggregates, depressed allowances, and
distorted housing-cost totals.

This second-stage QRF trains on the original FRS with predictors =
[demographics + first-stage income outputs] and outputs = a curated list
of FRS-only variables. For each SPI-donor row, it substitutes the
predicted value drawn from FRS respondents with similar demographics and
post-stage-1 incomes. Benefit ``_reported`` flags for high earners
naturally collapse to zero (no high-earner FRS respondent reports UC),
pension contributions rescale, and savings interest / rent correlate
with income instead of with the random FRS donor's draw.

Mirrors the US ``_impute_cps_only_variables`` approach introduced in
``policyengine-us-data#589`` but targets UK-specific FRS variables.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from policyengine_uk.data import UKSingleYearDataset
from policyengine_uk_data.datasets.disability_benefits import (
    add_disability_benefit_categories_from_reported_amounts,
    add_disability_benefit_flags_from_reported_amounts,
)
from policyengine_uk_data.datasets.frs import (
    BENEFITS_IN_OWN_RIGHT_REPORTED_COLUMNS,
    REPORTED_TAKEUP_ANCHORS,
    reported_benunit_mask,
)
from policyengine_uk_data.parameters import load_take_up_rate
from policyengine_uk_data.utils.takeup import assign_takeup_with_reported_anchors

logger = logging.getLogger(__name__)


STAGE2_DEMOGRAPHIC_PREDICTORS = [
    "age",
    "gender",
    "region",
]

# Predictors drawn from the first-stage QRF output columns. They are the
# same six income components that the first stage imputes from SPI.
STAGE2_INCOME_PREDICTORS = [
    "employment_income",
    "self_employment_income",
    "savings_interest_income",
    "dividend_income",
    "private_pension_income",
    "property_income",
]

# FRS-only variables the second stage replaces on SPI-donor rows. Kept
# conservative: benefit ``_reported`` columns and pension contributions
# are the leading sources of cross-income inconsistency, and are
# well-populated in the base FRS build so training is stable.
FRS_ONLY_PERSON_VARIABLES = [
    # Pension contributions
    "employee_pension_contributions",
    "employer_pension_contributions",
    "personal_pension_contributions",
    "pension_contributions_via_salary_sacrifice",
    # Savings-related
    "tax_free_savings_income",
    # Benefit `_reported` columns
    "universal_credit_reported",
    "pension_credit_reported",
    "child_benefit_reported",
    "housing_benefit_reported",
    "income_support_reported",
    "working_tax_credit_reported",
    "child_tax_credit_reported",
    "attendance_allowance_reported",
    "state_pension_reported",
    "dla_sc_reported",
    "dla_m_reported",
    "pip_m_reported",
    "pip_dl_reported",
    "sda_reported",
    "carers_allowance_reported",
    "iidb_reported",
    "afcs_reported",
    "bsp_reported",
    "incapacity_benefit_reported",
    "maternity_allowance_reported",
    "winter_fuel_allowance_reported",
    "council_tax_benefit_reported",
    "jsa_contrib_reported",
    "jsa_income_reported",
    "esa_contrib_reported",
    "esa_income_reported",
]

# Benefit reports set to zero on SPI-donor rows once the QRF has drawn them.
# The QRF draws each person's reports from their age, gender, region and
# incomes; it sees nothing of their benefit unit (partner, children, rent,
# capital) or of their health. A report says the person is an existing
# claimant, and policyengine-uk acts on that, so these are zeroed:
#
# - Income-related awards. Entitlement turns on the benefit unit's joint
#   means and make-up, which on these rows come from the imputed incomes, so
#   the model's own means test is the only coherent source. A report would
#   instead be a continuing-award gate (housing benefit, income support, tax
#   credits, income-related ESA and JSA), the paid amount (income-related
#   ESA and JSA; council tax reduction where the model has no scheme) or a
#   certain-claim anchor (UC, Pension Credit). Sure Start Maternity Grant
#   needs one of these awards and is copied from the donor, not drawn.
# - Benefits paid only to people out of work or incapable of it: ESA and
#   JSA (contributory), incapacity benefit and severe disablement allowance.
#   On the 2024-25 build, 76% of SPI-row ESA (contributory) reporters earned
#   more than ESA's permitted-work limit, against 2% of FRS reporters.
# - Child Benefit, whose only use is the take-up anchor. The draw ignores
#   the children: on the 2024-25 build, 44% of SPI-row reports were in
#   benefit units without a child under 16 (FRS rows: 8%).
#
# State pension (driven by age), winter fuel payment (not read by the
# model), the disability benefits, carer's allowance (the draws respect its
# earnings limit) and IIDB, AFCS and bereavement support (payable at any
# income) keep their drawn values. The zeroed columns stay in the QRF chain
# above so that the values kept do not change.
SPI_DONOR_ZEROED_PERSON_VARIABLES = [
    "universal_credit_reported",
    "pension_credit_reported",
    "housing_benefit_reported",
    "council_tax_benefit_reported",
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
]

# Seed for the take-up draws on SPI-donor rows; create_frs uses 100.
SPI_DONOR_TAKEUP_SEED = 101


def apply_spi_donor_benefit_rules(
    dataset: UKSingleYearDataset,
) -> UKSingleYearDataset:
    """Zero SPI-donor benefit reports and re-derive the flags built from them.

    ``create_frs`` sets ``receives_benefits_in_own_right`` and the
    report-anchored take-up flags from the donor's own reports, before the
    SPI rows exist. Here they are rebuilt from the rows' own reports: a unit
    reporting receipt claims, and the rest draw at the take-up rate. With the
    anchoring reports zeroed, every SPI-donor unit draws at the rate, since
    whether a synthetic family claims is unobserved.
    """
    dataset = dataset.copy()
    person, benunit = dataset.person, dataset.benunit
    for column in SPI_DONOR_ZEROED_PERSON_VARIABLES:
        if column in person.columns:
            person[column] = 0.0

    own_right = [c for c in BENEFITS_IN_OWN_RIGHT_REPORTED_COLUMNS if c in person]
    if "receives_benefits_in_own_right" in person.columns:
        person["receives_benefits_in_own_right"] = (
            person[own_right].fillna(0).sum(axis=1) > 0
        )

    year = int(str(dataset.time_period)[:4])
    generator = np.random.default_rng(seed=SPI_DONOR_TAKEUP_SEED)
    for flag, (rate_name, column) in REPORTED_TAKEUP_ANCHORS.items():
        if flag not in benunit.columns or column not in person.columns:
            continue
        benunit[flag] = assign_takeup_with_reported_anchors(
            generator.random(len(benunit)),
            load_take_up_rate(rate_name, year),
            reported_mask=reported_benunit_mask(person, benunit, column),
        )
    return dataset


def _one_hot_encode(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Return ``df`` with object-typed ``columns`` one-hot encoded.

    QRF predictors must be numeric. Uses ``pandas.get_dummies`` so
    identical category sets are produced from the same input data.
    """
    return pd.get_dummies(df, columns=columns, drop_first=False, dtype=float)


def _align_columns(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Ensure train/test share the same columns in the same order.

    After independent ``get_dummies`` calls on train and test one-hot
    expansions can diverge if a category appears in one set and not the
    other. Reindex both to the union of columns, filling missing cells
    with zero.
    """
    columns = sorted(set(train_df.columns) | set(test_df.columns))
    return (
        train_df.reindex(columns=columns, fill_value=0.0),
        test_df.reindex(columns=columns, fill_value=0.0),
    )


def _build_predictor_frame(dataset: UKSingleYearDataset) -> pd.DataFrame:
    """Return a person-indexed DataFrame of stage-2 predictor columns.

    ``region`` lives on the household frame in the enhanced-FRS build,
    so it is joined onto each person row via ``person_household_id``.
    Remaining predictors (age, gender, the six income components) are
    read directly from the person frame. If the person frame already
    carries ``region`` (as in some test fixtures and the standalone SPI
    build) that value wins and no join is performed.
    """
    person = dataset.person
    predictors = STAGE2_DEMOGRAPHIC_PREDICTORS + STAGE2_INCOME_PREDICTORS

    if "region" in person.columns:
        frame = person[predictors].copy()
    elif (
        "region" in dataset.household.columns
        and "person_household_id" in person.columns
    ):
        hh_region = dataset.household.set_index("household_id")["region"]
        person_region = person["person_household_id"].map(hh_region)
        frame = person[[c for c in predictors if c != "region"]].copy()
        frame["region"] = person_region.values
        frame = frame[predictors]
    else:
        raise KeyError(
            "Stage-2 imputation needs 'region' either on the person frame "
            "or on the household frame with a 'person_household_id' join key."
        )
    return frame


def impute_frs_only_variables(
    train_dataset: UKSingleYearDataset,
    target_dataset: UKSingleYearDataset,
) -> UKSingleYearDataset:
    """Impute FRS-only person variables onto ``target_dataset``.

    ``train_dataset`` must be a full FRS build (before income
    imputation) so the training rows preserve the original co-occurrence
    of income and every FRS-only variable. ``target_dataset`` is the
    SPI-donor subsample after the first-stage QRF has overwritten its
    income columns.

    A single multi-output QRF is fitted on the training data and used
    to predict values for every row of ``target_dataset``; predictions
    replace the existing (donor-leaked) values in
    ``FRS_ONLY_PERSON_VARIABLES`` only. Variables absent from either
    frame are skipped silently. ``apply_spi_donor_benefit_rules`` then
    zeroes ``SPI_DONOR_ZEROED_PERSON_VARIABLES`` and rebuilds the flags
    derived from reports, before the disability categories and flags are
    derived from the remaining reports.
    """
    from policyengine_uk_data.utils.qrf import QRF

    target_dataset = target_dataset.copy()

    train_person = train_dataset.person
    target_person = target_dataset.person

    # Use only variables present in both frames.
    outputs = [
        v
        for v in FRS_ONLY_PERSON_VARIABLES
        if v in train_person.columns and v in target_person.columns
    ]
    missing = set(FRS_ONLY_PERSON_VARIABLES) - set(outputs)
    if missing:
        logger.warning(
            "Stage-2 FRS-only imputation: %d variables absent from "
            "train/target frames, skipped: %s",
            len(missing),
            sorted(missing),
        )
    if not outputs:
        logger.warning(
            "Stage-2 FRS-only imputation: no output variables available; "
            "applying only the SPI-donor benefit rules."
        )
        return apply_spi_donor_benefit_rules(target_dataset)

    train_inputs_raw = _build_predictor_frame(train_dataset)
    target_inputs_raw = _build_predictor_frame(target_dataset)

    train_inputs = _one_hot_encode(train_inputs_raw, columns=["gender", "region"])
    target_inputs = _one_hot_encode(target_inputs_raw, columns=["gender", "region"])
    train_inputs, target_inputs = _align_columns(train_inputs, target_inputs)

    # Replace NaNs in outputs with 0 so the QRF trains on clean targets;
    # FRS-only variables are almost all zero-heavy "amount if eligible"
    # columns that default to zero when unreported.
    train_outputs = train_person[outputs].fillna(0).astype(float)

    logger.info(
        "Stage-2 FRS-only imputation: %d outputs, training on %d FRS "
        "persons, predicting for %d SPI-donor persons",
        len(outputs),
        len(train_inputs),
        len(target_inputs),
    )

    model = QRF()
    model.fit(train_inputs, train_outputs)
    predictions = model.predict(target_inputs)

    # The QRF occasionally returns NaN for extreme predictor combos;
    # clamp to zero (the population-typical value for these variables).
    predictions = predictions.fillna(0.0)

    for column in outputs:
        # Clamp negative predictions — these columns represent receipted
        # amounts or contributions and are non-negative by construction.
        values = np.maximum(predictions[column].values, 0.0)
        target_dataset.person[column] = values

    target_dataset = apply_spi_donor_benefit_rules(target_dataset)
    target_dataset.person = add_disability_benefit_categories_from_reported_amounts(
        target_dataset.person,
        int(str(target_dataset.time_period)[:4]),
    )
    target_dataset.person = add_disability_benefit_flags_from_reported_amounts(
        target_dataset.person,
        int(str(target_dataset.time_period)[:4]),
    )

    return target_dataset
