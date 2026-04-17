"""
Demographic ageing for the panel pipeline (step 3 of #345).

Given a base dataset and a number of years to advance, produces a new
dataset in which:

- every surviving person's ``age`` column is incremented by the step size,
- a fraction of persons die each year according to an age-indexed mortality
  table (their rows are removed),
- a fraction of women of reproductive age give birth each year according
  to an age-indexed fertility table (new person rows are appended, attached
  to the mother's existing benefit unit and household, with fresh
  ``person_id`` values that do not collide with any pre-existing ID).

The panel ID contract from step 1 is extended rather than broken: the
person IDs of *survivors* are preserved byte-for-byte, deaths remove IDs
and births add IDs. Consumers reasoning about the transition should use
``policyengine_uk_data.utils.panel_ids.classify_panel_ids``.

Out of scope for this module (tracked in #345):

- Real ONS life tables and fertility rates. The defaults shipped here are
  reasonable-shape **placeholders** and are clearly named as such. Callers
  supply the real rates via the ``mortality_rates`` and ``fertility_rates``
  arguments.
- Household and benefit unit formation beyond attaching newborns to their
  mother's existing unit. Children reaching adulthood, marriage, separation,
  leaving home — all deferred.
- Migration.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

from policyengine_uk.data import UKSingleYearDataset


AGE_COLUMN = "age"
SEX_COLUMN = "gender"
FEMALE_VALUE = "FEMALE"
MALE_VALUE = "MALE"

MIN_FERTILE_AGE = 15
MAX_FERTILE_AGE = 49

# Placeholder rates, NOT from ONS. They are shipped only so the module is
# usable end-to-end in tests and smoke-runs. A follow-up to #345 will plug
# in real ONS life tables and fertility rates by age of mother.
DEFAULT_MORTALITY_RATES_PLACEHOLDER: Mapping[int, float] = {
    # ages 0-14: very low
    **{age: 0.0005 for age in range(0, 15)},
    # ages 15-39: low and flat
    **{age: 0.001 for age in range(15, 40)},
    # ages 40-59: rising
    **{age: 0.001 * (1.08 ** (age - 40)) for age in range(40, 60)},
    # ages 60-99: exponential
    **{age: min(0.5, 0.01 * (1.1 ** (age - 60))) for age in range(60, 100)},
    # 100+: capped
    **{age: 0.5 for age in range(100, 121)},
}

DEFAULT_FERTILITY_RATES_PLACEHOLDER: Mapping[int, float] = {
    **{15: 0.005, 16: 0.01, 17: 0.02, 18: 0.03, 19: 0.04},
    **{age: 0.05 for age in range(20, 25)},
    **{age: 0.09 for age in range(25, 30)},
    **{age: 0.10 for age in range(30, 35)},
    **{age: 0.06 for age in range(35, 40)},
    **{age: 0.02 for age in range(40, 45)},
    **{age: 0.005 for age in range(45, 50)},
}


def age_dataset(
    base: UKSingleYearDataset,
    years: int,
    *,
    seed: int = 0,
    mortality_rates: Mapping[int, float] | None = None,
    fertility_rates: Mapping[int, float] | None = None,
) -> UKSingleYearDataset:
    """Return a demographically-aged copy of ``base``, ``years`` forward.

    The base dataset is not mutated. One year of ageing is applied at a
    time so that mortality and fertility are stochastic-independent per
    year, matching how real demographic models work.

    Args:
        base: the starting dataset.
        years: non-negative number of whole years to advance. ``0`` returns
            a straight copy.
        seed: random seed. Identical seeds produce identical outputs.
        mortality_rates: map from age to per-year death probability. Pass an
            empty dict to disable mortality. Pass ``None`` to use the
            placeholder defaults. Ages not in the table default to ``0``.
        fertility_rates: map from age of mother to per-year birth
            probability. Pass an empty dict to disable fertility. Pass
            ``None`` to use the placeholder defaults. Only applied to
            persons whose ``gender`` column equals ``FEMALE``.

    Returns:
        A new ``UKSingleYearDataset``. The ``time_period`` is not shifted —
        combine with ``uprate_dataset`` if monetary values also need to move
        forward.

    Raises:
        ValueError: if ``years`` is negative.
    """
    if years < 0:
        raise ValueError(f"years must be non-negative, got {years}.")

    if mortality_rates is None:
        mortality_rates = DEFAULT_MORTALITY_RATES_PLACEHOLDER
    if fertility_rates is None:
        fertility_rates = DEFAULT_FERTILITY_RATES_PLACEHOLDER

    rng = np.random.default_rng(seed)
    aged = base.copy()
    # ``copy`` returns shallow-copied DataFrames for the three tables, so
    # we still want deep copies to avoid surprising the caller if they
    # hold references to ``base.person`` etc.
    aged.person = aged.person.copy(deep=True)
    aged.benunit = aged.benunit.copy(deep=True)
    aged.household = aged.household.copy(deep=True)

    for _ in range(int(years)):
        aged = _apply_mortality(aged, mortality_rates, rng)
        aged = _apply_fertility(aged, fertility_rates, rng)
        aged.person[AGE_COLUMN] = aged.person[AGE_COLUMN].astype(int) + 1

    return aged


def _apply_mortality(
    dataset: UKSingleYearDataset,
    mortality_rates: Mapping[int, float],
    rng: np.random.Generator,
) -> UKSingleYearDataset:
    if not mortality_rates:
        return dataset
    person = dataset.person
    ages = person[AGE_COLUMN].astype(int).to_numpy()
    rates = np.array([mortality_rates.get(int(a), 0.0) for a in ages], dtype=float)
    draws = rng.random(size=ages.shape[0])
    survives = draws >= rates
    dataset.person = person.loc[survives].reset_index(drop=True)
    return dataset


def _apply_fertility(
    dataset: UKSingleYearDataset,
    fertility_rates: Mapping[int, float],
    rng: np.random.Generator,
) -> UKSingleYearDataset:
    if not fertility_rates:
        return dataset
    person = dataset.person
    if SEX_COLUMN not in person.columns:
        return dataset

    ages = person[AGE_COLUMN].astype(int).to_numpy()
    sexes = person[SEX_COLUMN].to_numpy()
    rates = np.array(
        [
            fertility_rates.get(int(a), 0.0)
            if (s == FEMALE_VALUE and MIN_FERTILE_AGE <= a <= MAX_FERTILE_AGE)
            else 0.0
            for a, s in zip(ages, sexes)
        ],
        dtype=float,
    )
    draws = rng.random(size=ages.shape[0])
    gives_birth = draws < rates
    n_births = int(gives_birth.sum())
    if n_births == 0:
        return dataset

    mother_rows = person.loc[gives_birth]
    max_existing_id = int(person["person_id"].max())
    new_ids = np.arange(max_existing_id + 1, max_existing_id + 1 + n_births, dtype=int)
    # Sex ratio at birth in the UK is ~1.05 boys per girl. A 50/50 split is
    # within sampling noise for this v1 placeholder. We leave the array in
    # plain Python-string form — pandas-extension dtypes (e.g. StringDtype)
    # are not valid numpy dtypes, so matching the column dtype via
    # ``numpy.ndarray.astype`` raises a TypeError. ``pd.concat`` will
    # coerce the column back to the right extension dtype at the end.
    new_sex = np.array(
        rng.choice([MALE_VALUE, FEMALE_VALUE], size=n_births), dtype=object
    )

    newborns = _build_newborn_rows(
        template=person,
        mother_rows=mother_rows,
        new_ids=new_ids,
        new_sex=new_sex,
    )
    dataset.person = pd.concat([person, newborns], ignore_index=True).reset_index(
        drop=True
    )
    return dataset


def _build_newborn_rows(
    template: pd.DataFrame,
    mother_rows: pd.DataFrame,
    new_ids: np.ndarray,
    new_sex: np.ndarray,
) -> pd.DataFrame:
    """Construct a DataFrame of newborn person rows matching ``template``.

    Every column from ``template`` is present on the output. Numeric columns
    default to ``0`` and object columns to empty string, except where we
    explicitly set values (IDs, age, gender, benunit and household links
    inherited from the mother).
    """
    n = len(new_ids)
    data: dict[str, np.ndarray] = {}
    for col in template.columns:
        series = template[col]
        if pd.api.types.is_numeric_dtype(series):
            data[col] = np.zeros(n, dtype=series.dtype)
        else:
            # Non-numeric columns may use pandas-extension dtypes (e.g.
            # StringDtype) that numpy cannot parse. Initialise with plain
            # object strings and let pandas coerce during concat below.
            data[col] = np.array([""] * n, dtype=object)

    data["person_id"] = new_ids.astype(template["person_id"].dtype)
    data[AGE_COLUMN] = np.zeros(n, dtype=template[AGE_COLUMN].dtype)
    data[SEX_COLUMN] = new_sex
    # Attach each newborn to its mother's benefit unit and household so
    # that downstream joins on the foreign keys still work.
    if "person_benunit_id" in template.columns:
        data["person_benunit_id"] = (
            mother_rows["person_benunit_id"]
            .to_numpy()
            .astype(template["person_benunit_id"].dtype)
        )
    if "person_household_id" in template.columns:
        data["person_household_id"] = (
            mother_rows["person_household_id"]
            .to_numpy()
            .astype(template["person_household_id"].dtype)
        )

    return pd.DataFrame(data, columns=template.columns)
