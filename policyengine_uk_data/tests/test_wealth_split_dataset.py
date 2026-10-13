"""The built dataset carries the WAS wealth split, and the locked engine reads it.

The engine checks need a policyengine-uk release with the share-like
components and share_like_capital (PolicyEngine/policyengine-uk#1974) and the
secured-debt inputs and reg 49 valuation (#1969).
"""

import numpy as np
import pytest

from policyengine_uk_data.datasets.frs_release import CURRENT_FRS_RELEASE
from policyengine_uk_data.datasets.imputations.wealth import (
    CORPORATE_WEALTH_COMPONENTS,
    SECURED_DEBT_ASSETS,
)

NEW_INPUTS = (
    "private_pension_wealth",
    *CORPORATE_WEALTH_COMPONENTS,
    "cash_isa",
    *SECURED_DEBT_ASSETS,
)


def test_corporate_wealth_is_the_sum_of_its_components(enhanced_frs):
    household = enhanced_frs.household
    components = household[list(CORPORATE_WEALTH_COMPONENTS)].sum(axis=1)
    assert np.abs(household.corporate_wealth - components).max() < 0.01


def test_secured_debt_only_where_the_asset_is_held(enhanced_frs):
    household = enhanced_frs.household
    for debt, asset in SECURED_DEBT_ASSETS.items():
        assert (household.loc[household[asset] <= 0, debt] == 0).all(), debt
        assert (household[debt] >= 0).all(), debt


@pytest.mark.parametrize("variable", NEW_INPUTS)
def test_the_engine_reads_each_new_input(baseline, enhanced_frs, variable):
    assert variable in baseline.tax_benefit_system.variables, (
        f"the locked policyengine-uk has no {variable}"
    )
    household = enhanced_frs.household
    stored = (household[variable] * household.household_weight).sum()
    simulated = baseline.calculate(variable, CURRENT_FRS_RELEASE.base_year).sum()
    assert simulated == pytest.approx(stored, rel=1e-4)


def test_the_engine_values_capital_with_these_inputs(baseline):
    from policyengine_uk.utils.capital_valuation import SECURED_DEBT

    # The engine deducts each imputed debt from the asset it is imputed for.
    assert {debt: asset for asset, debt in SECURED_DEBT.items()} == SECURED_DEBT_ASSETS
    year = CURRENT_FRS_RELEASE.base_year
    # corporate_wealth is the components' exact sum here, so the capital
    # block keeps it.
    assert baseline.calculate("share_like_capital", year).sum() == pytest.approx(
        baseline.calculate("corporate_wealth", year).sum(), rel=1e-6
    )
