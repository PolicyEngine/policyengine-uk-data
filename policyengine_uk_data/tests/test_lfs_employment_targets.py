"""ONS LFS employee and self-employed calibration targets.

Invariants:

1. Two national count targets, with the ONS values (held here independently
   of the source module, so a wrong value is caught) for every year listed.
2. The calibration and base years resolve to a value.
3. The household column counts the household's members whose main-job status
   is in the target's group, for any statuses and household layout
   (Hypothesis), so the weighted column total is the weighted head count.
4. On a built enhanced FRS, the weighted counts are near the targets.
   This also catches the local HMRC employment counts going back into
   training (``VALIDATION_ONLY_LOCAL_TARGETS``), which moves employees up.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from policyengine_uk_data.datasets.frs_release import CURRENT_FRS_RELEASE
from policyengine_uk_data.targets.build_loss_matrix import _resolve_value
from policyengine_uk_data.targets.registry import discover_source_modules
from policyengine_uk_data.targets.sources.ons_labour_market import get_targets
from policyengine_uk_data.utils.employment_status import (
    EMPLOYEE_STATUSES,
    SELF_EMPLOYED_STATUSES,
)

# ONS Labour market overview, 15 September 2026: MGRN and MGRQ, annual
# four-quarter averages, thousands.
ONS_THOUSANDS = {
    "ons/lfs_employees": {2022: 28_564, 2023: 28_821, 2024: 29_126, 2025: 29_590},
    "ons/lfs_self_employed": {2022: 4_244, 2023: 4_380, 2024: 4_340, 2025: 4_395},
}
STATUS_GROUPS = {
    "ons/lfs_employees": EMPLOYEE_STATUSES,
    "ons/lfs_self_employed": SELF_EMPLOYED_STATUSES,
}
ALL_STATUSES = (
    EMPLOYEE_STATUSES
    + SELF_EMPLOYED_STATUSES
    + ("CHILD", "UNEMPLOYED", "RETIRED", "STUDENT", "CARER", "OTHER_INACTIVE")
)
# Calibration only partly pulls national counts in (see the public sector
# employment test); the built-data check guards against the 4m drift seen
# without these targets, not against small misses.
BUILT_RELATIVE_TOLERANCE = 0.08


def _by_name():
    return {target.name: target for target in get_targets()}


def test_targets_and_values():
    targets = _by_name()
    assert set(targets) == set(ONS_THOUSANDS)
    for name, values in ONS_THOUSANDS.items():
        target = targets[name]
        assert target.is_count
        assert target.source == "ons"
        assert target.variable == "employment_status"
        assert target.values == {year: v * 1e3 for year, v in values.items()}


def test_source_module_is_discovered():
    # Discovery only imports the source modules; collecting every target
    # would download the other sources' tables.
    modules = {module.__name__ for module in discover_source_modules()}
    assert "policyengine_uk_data.targets.sources.ons_labour_market" in modules


@pytest.mark.parametrize(
    "year",
    sorted({CURRENT_FRS_RELEASE.base_year, CURRENT_FRS_RELEASE.calibration_year}),
)
def test_model_years_resolve(year):
    for name, target in _by_name().items():
        assert _resolve_value(target, year) == ONS_THOUSANDS[name][year] * 1e3


class _FakeContext:
    def __init__(self, status, household):
        self.status = np.array(status, dtype=object)
        self.household = np.array(household)

    def pe_person(self, variable):
        assert variable == "employment_status"
        return self.status

    def household_from_person(self, values):
        return np.bincount(self.household, weights=values, minlength=10)


@settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    st.lists(
        st.tuples(st.sampled_from(ALL_STATUSES), st.integers(0, 9)),
        min_size=1,
        max_size=60,
    )
)
def test_column_counts_household_members_in_group(people):
    status, household = zip(*people)
    ctx = _FakeContext(status, household)
    for name, target in _by_name().items():
        column = target.custom_compute(ctx, target, 2025)
        expected = np.zeros(10)
        for s, h in people:
            expected[h] += s in STATUS_GROUPS[name]
        np.testing.assert_array_equal(column, expected)


def test_built_enhanced_frs_near_lfs(enhanced_frs, baseline):
    year = CURRENT_FRS_RELEASE.calibration_year
    status = baseline.calculate("employment_status", year).values.astype(str)
    weight = baseline.calculate("person_weight", year).values
    for name, statuses in STATUS_GROUPS.items():
        estimate = weight[np.isin(status, statuses)].sum()
        target = ONS_THOUSANDS[name][year] * 1e3
        assert abs(estimate / target - 1) < BUILT_RELATIVE_TOLERANCE, (
            f"{name}: {estimate / 1e6:.2f}m against {target / 1e6:.2f}m"
        )
