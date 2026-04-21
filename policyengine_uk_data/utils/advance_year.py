"""Top-level composer for the panel pipeline.

``advance_year`` chains all the dynamic mechanics already on the branch
— migration, separation, leaving-home, marriage, employment and
income-decile transitions, mortality, fertility, age increment — plus
deterministic uprating, into a single reproducible one-year step.

Ordering rationale (the sequence is load-bearing):

1. **Migration** first so immigrants joining this year can be exposed
   to every subsequent transition, and emigrants leave before they
   accidentally marry / retire / age.
2. **Separations** before marriages so a person who separates this
   year cannot marry in the same step — matches how ONS registers
   marriage / divorce by year.
3. **Children leaving home** next: adult children who move out this
   year are then exposed to marriage as their own benunit.
4. **Marriages**: draws on the (now-expanded) single pool.
5. **Employment transitions**: redraws labour market state for
   anyone who survived all the composition changes above.
6. **Income-decile transitions**: repositions each worker in the
   income distribution.
7. **Demographic ageing** (mortality → fertility → age increment)
   last so that:
     - women who just married can give birth in the same step,
     - newborns enter at age 0 and stay age 0 at year end.
8. **Uprating** closes the step so monetary values match the target
   year's price level / OBR index.

Every transition uses the same seed-derived generator so the whole
year is deterministic, and every function returns a fresh dataset so
the caller never sees mutation.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from policyengine_uk.data import UKSingleYearDataset

from policyengine_uk_data.utils.demographic_ageing import age_dataset
from policyengine_uk_data.utils.household_transitions import (
    apply_children_leaving_home,
    apply_employment_transitions,
    apply_income_decile_transitions,
    apply_marriages,
    apply_migration,
    apply_separations,
)
from policyengine_uk_data.utils.uprating import uprate_dataset


def advance_year(
    dataset: UKSingleYearDataset,
    *,
    target_year: int | None = None,
    seed: int = 0,
    mortality_rates: Any = None,
    fertility_rates: Any = None,
    marriage_rates: Any = None,
    separation_rates: Any = None,
    leaving_home_rates: Any = None,
    net_migration_rates: Any = None,
    ukhls_employment_rates: Mapping | None = None,
    ukhls_decile_rates: Mapping | None = None,
    state_pension_age: int = 66,
    job_loss_rate: float = 0.03,
    job_gain_rate: float = 0.05,
    wage_drift: float = 0.04,
    uprate: bool = True,
) -> UKSingleYearDataset:
    """Run one full year of the panel pipeline against ``dataset``.

    Args:
        dataset: the starting-year dataset. Not mutated.
        target_year: year to uprate monetary values to. ``None`` takes
            ``dataset.time_period + 1``.
        seed: reproducibility seed. Every transition draws from the
            same seeded generator sequence.
        mortality_rates, fertility_rates, marriage_rates,
        separation_rates, leaving_home_rates, net_migration_rates:
            pass through to the corresponding ``apply_*`` / ageing
            function. ``None`` uses that function's default table.
            Pass ``{}`` to disable a specific transition.
        ukhls_employment_rates: if provided, the rule-based job
            loss/gain path in ``apply_employment_transitions`` is
            replaced by an age_band × sex × state draw from this
            mapping. Output of
            ``utils.ukhls_transitions.load_employment_transitions``.
        ukhls_decile_rates: if provided, adds a decile-transition
            step that repositions workers in the income distribution.
            Output of
            ``utils.ukhls_transitions.load_income_decile_transitions``.
        state_pension_age, job_loss_rate, job_gain_rate, wage_drift:
            employment-transition knobs.
        uprate: whether to run ``uprate_dataset`` at the end. Disable
            when composing multiple years and uprating in bulk.

    Returns:
        A new ``UKSingleYearDataset`` one year forward.
    """
    if target_year is None:
        target_year = int(dataset.time_period) + 1

    rng = np.random.default_rng(seed)

    ds = apply_migration(
        dataset,
        net_migration_rates=net_migration_rates,
        rng=rng,
    )
    ds = apply_separations(
        ds,
        separation_rates=separation_rates,
        rng=rng,
    )
    ds = apply_children_leaving_home(
        ds,
        leaving_home_rates=leaving_home_rates,
        rng=rng,
    )
    ds = apply_marriages(
        ds,
        marriage_rates=marriage_rates,
        rng=rng,
    )
    ds = apply_employment_transitions(
        ds,
        state_pension_age=state_pension_age,
        job_loss_rate=job_loss_rate,
        job_gain_rate=job_gain_rate,
        wage_drift=wage_drift,
        ukhls_rates=ukhls_employment_rates,
        rng=rng,
    )
    if ukhls_decile_rates:
        ds = apply_income_decile_transitions(
            ds,
            decile_rates=ukhls_decile_rates,
            rng=rng,
        )
    ds = age_dataset(
        ds,
        years=1,
        seed=int(rng.integers(0, 2**31 - 1)),
        mortality_rates=mortality_rates,
        fertility_rates=fertility_rates,
    )
    if uprate:
        ds = uprate_dataset(ds, target_year=target_year)
    return ds
