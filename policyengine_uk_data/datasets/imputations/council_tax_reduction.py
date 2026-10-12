"""Learn household CTR receipt without manufacturing reported awards.

SPI incomes replace personal income records, while CTR is a household award.
A separate household receipt model learns from the original FRS and supplies
an explicit claim flag only to the household reference person's benefit unit.
PolicyEngine then calculates eligibility and the award from current policy.

An alternative is to impute the household award itself. That would require an
award override, or synthetic reported CTR, and would make the imputed amount
less responsive to reforms. Receipt probabilities retain the FRS income
gradient without choosing a uniform take-up rate or a majority-label cutoff.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from policyengine_uk.data import UKSingleYearDataset

HOUSEHOLD_CTR_INCOME_PREDICTORS = (
    "employment_income",
    "self_employment_income",
    "savings_interest_income",
    "dividend_income",
    "private_pension_income",
    "property_income",
)
HOUSEHOLD_CTR_RECEIPT_TARGET = "household_receives_council_tax_reduction"
HOUSEHOLD_CTR_MODEL_SEED = 42
HOUSEHOLD_CTR_DRAW_SEED = 102


def _require_columns(frame: pd.DataFrame, name: str, columns) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise KeyError(f"Household CTR imputation needs {name} columns: {missing}")


def _require_unique_ids(frame: pd.DataFrame, column: str) -> None:
    if frame[column].isna().any() or frame[column].duplicated().any():
        raise ValueError(f"Household CTR imputation needs unique, non-null {column}")


def _household_predictors(dataset: UKSingleYearDataset) -> pd.DataFrame:
    """Aggregate persons by household ID, independently of table row order.

    Adults are people aged 18 or older; children are younger than 18. For a
    household with no adult, oldest adult age falls back to its oldest member.
    """
    person, household = dataset.person, dataset.household
    _require_columns(
        person,
        "person",
        ("person_household_id", "age", *HOUSEHOLD_CTR_INCOME_PREDICTORS),
    )
    _require_columns(
        household,
        "household",
        ("household_id", "region", "tenure_type", "council_tax"),
    )
    _require_unique_ids(household, "household_id")
    household_ids = pd.Index(household.household_id, name="household_id").sort_values()
    person_households = person.person_household_id
    if not person_households.isin(household_ids).all():
        raise ValueError("Household CTR imputation found a person without a household")
    if not household_ids.isin(person_households).all():
        raise ValueError("Household CTR imputation found a household without a person")

    incomes = person[list(HOUSEHOLD_CTR_INCOME_PREDICTORS)].astype(float).fillna(0)
    if not np.isfinite(incomes.to_numpy()).all():
        raise ValueError("Household CTR imputation needs finite personal incomes")
    frame = incomes.groupby(person_households).sum().reindex(household_ids)

    ages = person.age.astype(float)
    if not np.isfinite(ages.to_numpy()).all():
        raise ValueError("Household CTR imputation needs finite ages")
    adults = ages.ge(18)
    frame["adult_count"] = adults.groupby(person_households).sum()
    frame["child_count"] = (~adults).groupby(person_households).sum()
    frame["oldest_adult_age"] = (
        ages.where(adults)
        .groupby(person_households)
        .max()
        .fillna(ages.groupby(person_households).max())
    )

    household_by_id = household.set_index("household_id")
    gross_tax = household_by_id.council_tax.astype(float).fillna(0)
    if not np.isfinite(gross_tax.to_numpy()).all():
        raise ValueError("Household CTR imputation needs finite gross council tax")
    frame["council_tax"] = gross_tax.clip(lower=0)
    for column in ("region", "tenure_type"):
        frame[column] = household_by_id[column].fillna("UNKNOWN").astype(str)
    return frame


def _reference_person_benunits(dataset: UKSingleYearDataset) -> pd.Series:
    """Return one benefit-unit ID per household, using explicit entity keys."""
    person, benunit = dataset.person, dataset.benunit
    _require_columns(person, "person", ("person_benunit_id", "is_household_head"))
    _require_columns(benunit, "benunit", ("benunit_id",))
    _require_unique_ids(benunit, "benunit_id")
    if not person.person_benunit_id.isin(benunit.benunit_id).all():
        raise ValueError(
            "Household CTR imputation found a person without a benefit unit"
        )
    if not benunit.benunit_id.isin(person.person_benunit_id).all():
        raise ValueError(
            "Household CTR imputation found a benefit unit without a person"
        )
    if person.groupby("person_benunit_id").person_household_id.nunique().gt(1).any():
        raise ValueError(
            "Household CTR imputation found a benefit unit spanning households"
        )
    if not person.is_household_head.isin((False, True)).all():
        raise ValueError("Household CTR imputation needs boolean is_household_head")
    heads = person.loc[
        person.is_household_head.astype(bool),
        ["person_household_id", "person_benunit_id"],
    ]
    head_counts = heads.groupby("person_household_id").size()
    if (
        not head_counts.reindex(dataset.household.household_id, fill_value=0)
        .eq(1)
        .all()
    ):
        raise ValueError(
            "Household CTR imputation needs exactly one household reference person "
            "per household"
        )
    return heads.set_index("person_household_id").person_benunit_id


def _receipt_probabilities(fitted_model, inputs: pd.DataFrame) -> pd.Series:
    """Read the binary classifier probabilities exposed by MicroImpute QRF.

    The locked QRF implementation uses a random-forest classifier for binary
    targets. Its default predictions are majority labels, so request the class
    probabilities and draw receipt separately with our own local generator.
    """
    prediction = fitted_model.predict(inputs, return_probs=True)
    if isinstance(prediction, pd.DataFrame):
        # MicroImpute returns a frame directly for constant training targets.
        if not prediction.index.equals(inputs.index):
            raise ValueError("Household CTR predictions lost their household IDs")
        probabilities = prediction[HOUSEHOLD_CTR_RECEIPT_TARGET].astype(float)
    else:
        frames = [
            value for value in prediction.values() if isinstance(value, pd.DataFrame)
        ]
        if len(frames) != 1 or not frames[0].index.equals(inputs.index):
            raise ValueError("Household CTR predictions lost their household IDs")
        info = prediction["probabilities"][HOUSEHOLD_CTR_RECEIPT_TARGET]
        classes = np.asarray(info["classes"])
        class_probabilities = np.asarray(info["probabilities"])
        if class_probabilities.shape != (len(inputs), len(classes)):
            raise ValueError(
                "Household CTR classifier returned invalid probability dimensions"
            )
        probabilities = pd.Series(
            class_probabilities[:, classes == True].sum(axis=1),
            index=inputs.index,
        )
    if (
        not np.isfinite(probabilities.to_numpy()).all()
        or not probabilities.between(0, 1).all()
    ):
        raise ValueError(
            "Household CTR classifier returned invalid receipt probabilities"
        )
    return probabilities


def impute_household_council_tax_reduction(
    train_dataset: UKSingleYearDataset,
    target_dataset: UKSingleYearDataset,
) -> UKSingleYearDataset:
    """Return SPI households with learned CTR claim flags, leaving reports alone.

    Training uses the original FRS, before income imputation. The target is
    the SPI-donor subsample after income imputation and reported CTR zeroing.
    Its predictors sum the six first-stage income components and include
    household composition, region, tenure and the already-built gross bill.
    Receipt is learned from a positive household sum of reported FRS CTR.

    Household IDs are sorted before fitting and drawing to keep the seeded
    results stable when input frames are shuffled. Each household can claim
    only through its reference person's benefit unit, and a zero gross bill
    forces non-receipt. Explicit flags on every target benefit unit leave
    eligibility and the award calculation to the current policy model.
    """
    from microimpute.models import QRF

    train_inputs = _household_predictors(train_dataset)
    target_inputs = _household_predictors(target_dataset)
    reference_benunits = _reference_person_benunits(target_dataset)
    _require_columns(
        train_dataset.person, "training person", ("council_tax_benefit_reported",)
    )
    reported = train_dataset.person.council_tax_benefit_reported.astype(float).fillna(0)
    if not np.isfinite(reported.to_numpy()).all():
        raise ValueError("Household CTR imputation needs finite reported CTR")
    train_outputs = (
        reported.groupby(train_dataset.person.person_household_id).sum().gt(0)
    )
    train_outputs = train_outputs.reindex(train_inputs.index).rename(
        HOUSEHOLD_CTR_RECEIPT_TARGET
    )

    categories = ["region", "tenure_type"]
    train_encoded = pd.get_dummies(train_inputs, columns=categories, dtype=float)
    target_encoded = pd.get_dummies(target_inputs, columns=categories, dtype=float)
    columns = sorted(set(train_encoded.columns) | set(target_encoded.columns))
    train_encoded = train_encoded.reindex(columns=columns, fill_value=0)
    target_encoded = target_encoded.reindex(columns=columns, fill_value=0)

    model = QRF()
    model.seed = HOUSEHOLD_CTR_MODEL_SEED
    fitted = model.fit(
        pd.concat([train_encoded, train_outputs], axis=1),
        predictors=list(train_encoded.columns),
        imputed_variables=[HOUSEHOLD_CTR_RECEIPT_TARGET],
    )
    probabilities = _receipt_probabilities(fitted, target_encoded)
    generator = np.random.default_rng(HOUSEHOLD_CTR_DRAW_SEED)
    receives = probabilities.gt(generator.random(len(probabilities)))
    receives &= target_inputs.council_tax.gt(0)
    benunit_receives = pd.Series(
        receives.reindex(reference_benunits.index).to_numpy(),
        index=reference_benunits.to_numpy(),
    )
    result = target_dataset.copy()
    result.benunit["would_claim_council_tax_reduction"] = (
        result.benunit.benunit_id.map(benunit_receives).fillna(False).astype(bool)
    )
    return result
