"""Build calibration loss matrices from the targets registry.

Bridges the targets system to the calibration pipeline by converting
each Target into a household-level column vector and a scalar target
value, producing the (matrix, targets) pair that the weight optimiser
expects.

For most targets the column is a straightforward simulation query
(sum a variable, count recipients, filter by region/age/income band).
For targets requiring custom logic (counterfactuals, cross-tabs), the
Target's custom_compute callable is invoked instead.
"""

import logging

import numpy as np
import pandas as pd

from policyengine_uk_data.targets import get_all_targets
from policyengine_uk_data.targets.schema import (
    GeographicLevel,
    Target,
    Unit,
)
from policyengine_uk_data.targets.compute import (
    compute_benefit_cap,
    compute_council_tax_band,
    compute_esa,
    compute_gender_age,
    compute_household_type,
    compute_housing,
    compute_income_band,
    compute_obr_council_tax,
    compute_pip_claimants,
    compute_regional_age,
    compute_savings_interest,
    compute_scotland_demographics,
    compute_scotland_uc_child,
    compute_scottish_child_payment,
    compute_student_loan_plan,
    compute_ss_contributions,
    compute_ss_headcount,
    compute_ss_it_relief,
    compute_ss_ni_relief,
    compute_tenure,
    compute_two_child_limit,
    compute_uc_by_children,
    compute_uc_by_family_type,
    compute_uc_jobseeker,
    compute_uc_outside_cap,
    compute_uc_payment_dist,
    compute_uk_population,
    compute_vehicles,
)

logger = logging.getLogger(__name__)


def create_target_matrix(
    dataset,
    time_period: str = None,
    reform=None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Create (matrix, target_values) for calibration.

    Args:
        dataset: a UKSingleYearDataset instance.
        time_period: calendar year as string; defaults to dataset year.
        reform: optional PolicyEngine reform.

    Returns:
        (df, targets) where df has one column per target and one row
        per household, and targets is a Series of scalar target values
        indexed by target name.
    """
    from policyengine_uk import Microsimulation

    if time_period is None:
        time_period = dataset.time_period

    year = int(time_period)
    sim = Microsimulation(dataset=dataset, reform=reform)
    sim.default_calculation_period = time_period

    ctx = _SimContext(sim, time_period, dataset, reform)

    all_targets = []
    seen = set()
    for level in (
        GeographicLevel.NATIONAL,
        GeographicLevel.REGION,
        GeographicLevel.COUNTRY,
    ):
        for t in get_all_targets(geographic_level=level):
            if t.name not in seen:
                seen.add(t.name)
                all_targets.append(t)

    df = pd.DataFrame()
    target_names = []
    target_values = []

    for target in all_targets:
        try:
            val = _resolve_value(target, year)
            if val is None:
                continue
            col = _compute_column(target, ctx, year)
            if col is None:
                continue
            df[target.name] = col
            target_names.append(target.name)
            target_values.append(val)
        except Exception as e:
            logger.warning("Skipping target %s: %s", target.name, e)

    return df, pd.Series(target_values, index=target_names)


def _resolve_value(target: Target, year: int) -> float | None:
    """Get the target value for a year, falling back to nearest year.

    VOA council tax targets are population-uprated when extrapolating
    from their base year (2024).
    """
    if year in target.values:
        return target.values[year]
    available = sorted(target.values.keys())
    if not available:
        return None
    closest = min(available, key=lambda y: abs(y - year))
    if abs(closest - year) > 3:
        return None
    if closest > year:
        return None
    base_value = target.values[closest]
    # VOA council tax counts scale with population
    if target.source == "voa" and year != closest:
        from policyengine_uk_data.targets.sources.local_age import (
            get_uk_total_population,
        )

        pop_target = get_uk_total_population(year)
        pop_base = get_uk_total_population(closest)
        if pop_base > 0:
            base_value *= pop_target / pop_base
    return base_value


class _SimContext:
    """Holds the simulation and lazily computed intermediate arrays."""

    def __init__(self, sim, time_period, dataset, reform):
        self.sim = sim
        self.time_period = time_period
        self.dataset = dataset
        self.reform = reform
        self._cache = {}

    def pe(self, variable: str):
        """Calculate variable mapped to household level."""
        key = ("pe", variable)
        if key not in self._cache:
            self._cache[key] = self.sim.calculate(
                variable, map_to="household"
            ).values
        return self._cache[key]

    def pe_person(self, variable: str):
        """Calculate variable at person level."""
        key = ("pe_person", variable)
        if key not in self._cache:
            self._cache[key] = self.sim.calculate(variable).values
        return self._cache[key]

    def pe_count(self, *variables):
        """Count people with variable > 0, mapped to household."""
        total = 0
        for variable in variables:
            entity = self.sim.tax_benefit_system.variables[variable].entity.key
            total += self.sim.map_result(
                self.sim.calculate(variable) > 0,
                entity,
                "household",
            )
        return total

    def household_from_person(self, values):
        return self.sim.map_result(values, "person", "household")

    def household_from_family(self, values):
        return self.sim.map_result(values, "benunit", "household")

    @property
    def region(self):
        if "region" not in self._cache:
            self._cache["region"] = self.sim.calculate(
                "region", map_to="person"
            )
        return self._cache["region"]

    @property
    def household_region(self):
        if "household_region" not in self._cache:
            self._cache["household_region"] = self.sim.calculate(
                "region", map_to="household"
            ).values
        return self._cache["household_region"]

    @property
    def age(self):
        if "age" not in self._cache:
            self._cache["age"] = self.sim.calculate("age").values
        return self._cache["age"]

    @property
    def country(self):
        if "country" not in self._cache:
            self._cache["country"] = self.sim.calculate("country").values
        return self._cache["country"]

    @property
    def counterfactual_sim(self):
        """Lazily create the salary sacrifice counterfactual."""
        if "counterfactual_sim" not in self._cache:
            from policyengine_uk import Microsimulation

            ss = self.sim.calculate(
                "pension_contributions_via_salary_sacrifice"
            )
            emp = self.sim.calculate("employment_income")
            cf_sim = Microsimulation(dataset=self.dataset, reform=self.reform)
            cf_sim.set_input(
                "pension_contributions_via_salary_sacrifice",
                self.time_period,
                np.zeros_like(ss),
            )
            cf_sim.set_input(
                "employment_income",
                self.time_period,
                emp + ss,
            )
            self._cache["counterfactual_sim"] = cf_sim
        return self._cache["counterfactual_sim"]


# ── Column computation dispatch ──────────────────────────────────────


def _compute_column(
    target: Target, ctx: _SimContext, year: int
) -> np.ndarray | None:
    """Compute the household-level column for a target.

    Dispatches to domain-specific compute modules.
    """
    if target.custom_compute is not None:
        return target.custom_compute(ctx, target, year)

    name = target.name

    # Demographics
    if name.startswith("ons/") and "_age_" in name:
        return compute_regional_age(target, ctx)
    if name.startswith("ons/female_") or name.startswith("ons/male_"):
        return compute_gender_age(target, ctx)
    if name == "ons/uk_population":
        return compute_uk_population(target, ctx)
    if name in (
        "ons/scotland_children_under_16",
        "ons/scotland_babies_under_1",
        "ons/scotland_households_3plus_children",
    ):
        return compute_scotland_demographics(target, ctx)

    # Households and tenure
    if target.variable == "family_type" and target.is_count:
        return compute_household_type(target, ctx)
    if target.variable == "tenure_type" and target.is_count:
        return compute_tenure(target, ctx)

    # Income bands (HMRC SPI)
    if target.breakdown_variable == "total_income":
        return compute_income_band(target, ctx)

    # Council tax
    if name.startswith("voa/council_tax/"):
        return compute_council_tax_band(target, ctx)
    if name.startswith("obr/council_tax"):
        return compute_obr_council_tax(target, ctx)

    # Vehicles
    if name.startswith("nts/households_"):
        return compute_vehicles(target, ctx)

    # Housing
    if name in ("housing/total_mortgage", "housing/rent_private"):
        return compute_housing(target, ctx)

    # Savings
    if name == "ons/savings_interest_income":
        return compute_savings_interest(target, ctx)

    # Scottish child payment
    if name == "sss/scottish_child_payment":
        return compute_scottish_child_payment(target, ctx)

    # Student loan plan borrower counts (SLC)
    if name.startswith("slc/plan_"):
        return compute_student_loan_plan(target, ctx)

    # PIP claimants
    if name in (
        "dwp/pip_dl_standard_claimants",
        "dwp/pip_dl_enhanced_claimants",
    ):
        return compute_pip_claimants(target, ctx)

    # Benefit cap
    if name in (
        "dwp/benefit_capped_households",
        "dwp/benefit_cap_total_reduction",
    ):
        return compute_benefit_cap(target, ctx)

    # Scotland UC + child under 1
    if name == "dwp/scotland_uc_households_child_under_1":
        return compute_scotland_uc_child(target, ctx)

    # UC claimants by children
    if name.startswith("dwp/uc/claimants_with_") and "_children" in name:
        return compute_uc_by_children(target, ctx)

    # UC claimants by family type
    if name.startswith("dwp/uc/claimants_") and not name.startswith(
        "dwp/uc/claimants_with_"
    ):
        return compute_uc_by_family_type(target, ctx)

    # UC payment distribution
    if name.startswith("dwp/uc_payment_dist/"):
        return compute_uc_payment_dist(target, ctx)

    # Salary sacrifice IT relief
    if name.startswith("hmrc/salary_sacrifice_it_relief_"):
        return compute_ss_it_relief(target, ctx)

    # Salary sacrifice contributions
    if name == "hmrc/salary_sacrifice_contributions":
        return compute_ss_contributions(target, ctx)

    # Salary sacrifice headcount
    if name.startswith("obr/salary_sacrifice_users_"):
        return compute_ss_headcount(target, ctx)

    # Salary sacrifice NI relief
    if name in (
        "hmrc/salary_sacrifice_employee_nics_relief",
        "obr/salary_sacrifice_employee_ni_relief",
        "hmrc/salary_sacrifice_employer_nics_relief",
        "obr/salary_sacrifice_employer_ni_relief",
    ):
        return compute_ss_ni_relief(target, ctx)

    # UC jobseeker splits
    if name in (
        "obr/universal_credit_jobseekers",
        "obr/universal_credit_non_jobseekers",
        "obr/universal_credit_jobseekers_count",
        "obr/universal_credit_non_jobseekers_count",
    ):
        return compute_uc_jobseeker(target, ctx)

    # UC outside benefit cap
    if name == "obr/universal_credit_outside_cap":
        return compute_uc_outside_cap(target, ctx)

    # Two-child limit
    if "two_child_limit" in name:
        return compute_two_child_limit(target, ctx)

    # ESA (combined income + contributory)
    if name == "obr/esa":
        return compute_esa(target, ctx)

    # Fallbacks: simple GBP sum / simple count
    if target.unit == Unit.GBP and not target.is_count:
        return _compute_simple_gbp(target, ctx)
    if target.is_count and target.unit == Unit.COUNT:
        return _compute_simple_count(target, ctx)

    logger.debug("No compute logic for target %s", name)
    return None


def _compute_simple_gbp(target: Target, ctx: _SimContext) -> np.ndarray | None:
    """Sum a variable at household level."""
    variable = target.variable
    try:
        entity = ctx.sim.tax_benefit_system.variables[variable].entity.key
    except KeyError:
        return None
    if entity == "household":
        return ctx.pe(variable)
    elif entity == "person":
        return ctx.household_from_person(ctx.sim.calculate(variable))
    elif entity == "benunit":
        return ctx.household_from_family(ctx.sim.calculate(variable))
    return None


def _compute_simple_count(target: Target, ctx: _SimContext) -> np.ndarray:
    """Count recipients of a variable, mapped to household."""
    return ctx.pe_count(target.variable)
