"""Household CTR receipt is learned and routed by entity IDs, without reports."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from policyengine_uk_data.datasets.imputations.council_tax_reduction import (
    HOUSEHOLD_CTR_INCOME_PREDICTORS,
    HOUSEHOLD_CTR_MODEL_SEED,
    HOUSEHOLD_CTR_RECEIPT_TARGET,
    impute_household_council_tax_reduction,
)


def _dataset():
    from policyengine_uk.data import UKSingleYearDataset

    person = pd.DataFrame(
        {
            "person_id": [91, 92, 93, 94, 95, 96, 97, 98],
            "person_household_id": [710, 710, 710, 710, 101, 101, 330, 330],
            "person_benunit_id": [7100, 7100, 7109, 7100, 1017, 1017, 3300, 3308],
            "is_household_head": [True, False, False, False, True, False, True, False],
            "age": [45, 43, 22, 12, 70, 66, 40, 8],
            "employment_income": [10_000, 5_000, 7_000, 0, 0, 0, 15_000, 0],
            "council_tax_benefit_reported": [40, 60, 0, 0, 0, 0, 0, 15],
        },
        index=[18, 4, 77, 29, 55, 88, 12, 99],
    )
    for i, column in enumerate(HOUSEHOLD_CTR_INCOME_PREDICTORS[1:], start=1):
        person[column] = np.arange(1, len(person) + 1) * i
    benunit = pd.DataFrame(
        {
            "benunit_id": [3308, 7109, 1017, 7100, 3300],
            "would_claim_council_tax_reduction": True,
        },
        index=[44, 11, 76, 32, 92],
    )
    household = pd.DataFrame(
        {
            "household_id": [710, 101, 330],
            "region": ["LONDON", "SCOTLAND", "WALES"],
            "tenure_type": ["RENT_FROM_COUNCIL", "OWNED_OUTRIGHT", "RENT_PRIVATELY"],
            "council_tax": [1_500, 0, 1_700],
        },
        index=[37, 11, 88],
    )
    return UKSingleYearDataset(
        person=person,
        benunit=benunit,
        household=household,
        fiscal_year=2023,
    )


def _single_person_dataset(employment_incomes, reports=None):
    from policyengine_uk.data import UKSingleYearDataset

    n = len(employment_incomes)
    household_ids = np.arange(n) * 7 + 10
    benunit_ids = household_ids * 10 + 3
    person = pd.DataFrame(
        {
            "person_id": household_ids * 100 + 8,
            "person_household_id": household_ids,
            "person_benunit_id": benunit_ids,
            "is_household_head": True,
            "age": 40,
            "employment_income": employment_incomes,
            "council_tax_benefit_reported": np.zeros(n) if reports is None else reports,
        }
    )
    for column in HOUSEHOLD_CTR_INCOME_PREDICTORS[1:]:
        person[column] = 0.0
    return UKSingleYearDataset(
        person=person,
        benunit=pd.DataFrame({"benunit_id": benunit_ids}),
        household=pd.DataFrame(
            {
                "household_id": household_ids,
                "region": "LONDON",
                "tenure_type": "RENT_PRIVATELY",
                "council_tax": 2_000.0,
            }
        ),
        fiscal_year=2023,
    )


def _fake_qrf(monkeypatch, probability):
    """Match MicroImpute's fit and classifier probability result contract."""
    import microimpute.models

    fitted_models = []

    class FakeQRF:
        def fit(self, data, predictors, imputed_variables):
            assert imputed_variables == [HOUSEHOLD_CTR_RECEIPT_TARGET]
            assert HOUSEHOLD_CTR_RECEIPT_TARGET not in predictors
            self.data = data.copy(deep=True)
            self.predictors = predictors
            fitted_models.append(self)
            return self

        def predict(self, inputs, return_probs=False):
            assert return_probs
            self.inputs = inputs.copy(deep=True)
            probabilities = np.asarray(probability(inputs), dtype=float)
            return {
                0.5: pd.DataFrame(
                    {HOUSEHOLD_CTR_RECEIPT_TARGET: probabilities > 0.5},
                    index=inputs.index,
                ),
                "probabilities": {
                    HOUSEHOLD_CTR_RECEIPT_TARGET: {
                        "classes": np.array([False, True]),
                        "probabilities": np.column_stack(
                            [1 - probabilities, probabilities]
                        ),
                    }
                },
            }

    monkeypatch.setattr(microimpute.models, "QRF", FakeQRF)
    return fitted_models


def test_training_aggregates_income_reports_and_composition_by_household(monkeypatch):
    models = _fake_qrf(monkeypatch, lambda x: np.zeros(len(x)))
    train = _dataset()
    target = _dataset()
    target.person.council_tax_benefit_reported = 0.0

    impute_household_council_tax_reduction(train, target)

    model = models[0]
    assert model.seed == HOUSEHOLD_CTR_MODEL_SEED
    assert model.data.index.tolist() == [101, 330, 710]
    assert model.data[HOUSEHOLD_CTR_RECEIPT_TARGET].tolist() == [False, True, True]
    assert model.data.loc[710, "employment_income"] == 22_000
    for i, column in enumerate(HOUSEHOLD_CTR_INCOME_PREDICTORS[1:], start=1):
        assert model.data.loc[710, column] == 10 * i
    assert model.data.loc[710, "adult_count"] == 3
    assert model.data.loc[710, "child_count"] == 1
    assert model.data.loc[710, "oldest_adult_age"] == 45
    assert model.data.loc[101, "oldest_adult_age"] == 70
    assert model.data.loc[710, "council_tax"] == 1_500
    assert model.data.loc[710, "region_LONDON"] == 1
    assert model.data.loc[330, "region_WALES"] == 1
    assert model.data.loc[710, "tenure_type_RENT_FROM_COUNCIL"] == 1
    assert list(model.inputs.columns) == model.predictors


def test_receipt_maps_only_to_reference_person_benunit_and_gross_bill(monkeypatch):
    _fake_qrf(monkeypatch, lambda x: np.ones(len(x)))
    train, target = _dataset(), _dataset()
    target.person.council_tax_benefit_reported = 0.0
    original_train = train.copy()
    original_target = target.copy()

    result = impute_household_council_tax_reduction(train, target)

    assert result.benunit.set_index(
        "benunit_id"
    ).would_claim_council_tax_reduction.to_dict() == {
        3308: False,
        7109: False,
        1017: False,  # This household has no gross council tax bill.
        7100: True,
        3300: True,
    }
    for name in ("person", "benunit", "household"):
        pd.testing.assert_frame_equal(
            getattr(train, name), getattr(original_train, name)
        )
        pd.testing.assert_frame_equal(
            getattr(target, name), getattr(original_target, name)
        )
    pd.testing.assert_frame_equal(result.person, original_target.person)
    pd.testing.assert_frame_equal(result.household, original_target.household)
    assert result.person.council_tax_benefit_reported.eq(0).all()


def test_seeded_receipt_is_stable_when_all_frames_are_shuffled(monkeypatch):
    _fake_qrf(monkeypatch, lambda x: np.full(len(x), 0.5))
    train = _single_person_dataset(np.arange(100) * 100, np.arange(100) % 2)
    target = _single_person_dataset(np.arange(100) * 100)
    shuffled_train, shuffled_target = train.copy(), target.copy()
    for name in ("person", "benunit", "household"):
        for dataset in (shuffled_train, shuffled_target):
            frame = getattr(dataset, name).sample(frac=1, random_state=3)
            # The arbitrary frame index must not serve as an entity join key.
            frame.index = np.arange(len(frame)) * 13 + 5
            setattr(dataset, name, frame)

    result = impute_household_council_tax_reduction(train, target)
    shuffled_result = impute_household_council_tax_reduction(
        shuffled_train, shuffled_target
    )

    flags = result.benunit.set_index(
        "benunit_id"
    ).would_claim_council_tax_reduction.sort_index()
    shuffled_flags = shuffled_result.benunit.set_index(
        "benunit_id"
    ).would_claim_council_tax_reduction.sort_index()
    pd.testing.assert_series_equal(flags, shuffled_flags)
    # A learned 50% probability produces both claimants and non-claimants.
    # Majority-label predictions from this fake model would produce none.
    assert 25 < flags.sum() < 75


@pytest.mark.parametrize(
    ("frame", "column", "match"),
    [
        ("household", "council_tax", "council_tax"),
        ("person", "is_household_head", "is_household_head"),
        ("person", "person_benunit_id", "person_benunit_id"),
    ],
)
def test_missing_inputs_raise_clear_errors(frame, column, match):
    train, target = _dataset(), _dataset()
    setattr(target, frame, getattr(target, frame).drop(columns=column))
    with pytest.raises(KeyError, match=match):
        impute_household_council_tax_reduction(train, target)


@pytest.mark.parametrize("reference_person_count", [0, 2])
def test_ambiguous_reference_person_is_rejected(reference_person_count):
    train, target = _dataset(), _dataset()
    household_mask = target.person.person_household_id.eq(710)
    target.person.loc[household_mask, "is_household_head"] = False
    target.person.loc[
        target.person.index[household_mask][:reference_person_count],
        "is_household_head",
    ] = True
    with pytest.raises(ValueError, match="exactly one household reference person"):
        impute_household_council_tax_reduction(train, target)


def test_real_qrf_receipt_probabilities_follow_training_income_gradient():
    """Exercise the locked model's probability API without a microsimulation."""
    incomes = np.r_[np.full(16, 1_000.0), np.full(16, 200_000.0)]
    reports = np.r_[np.full(16, 1_000.0), np.zeros(16)]
    train = _single_person_dataset(incomes, reports)
    target = _single_person_dataset(np.r_[np.full(8, 1_000.0), np.full(8, 200_000.0)])

    result = impute_household_council_tax_reduction(train, target)

    flags = result.benunit.would_claim_council_tax_reduction.to_numpy()
    assert flags[:8].all()
    assert not flags[8:].any()
    assert result.person.council_tax_benefit_reported.eq(0).all()


@pytest.mark.parametrize("reported_amount", [0.0, 100.0])
def test_real_qrf_supports_constant_receipt_targets(reported_amount):
    train = _single_person_dataset([1_000, 2_000], [reported_amount] * 2)
    target = _single_person_dataset([1_000, 2_000])

    result = impute_household_council_tax_reduction(train, target)

    assert result.benunit.would_claim_council_tax_reduction.eq(
        reported_amount > 0
    ).all()
