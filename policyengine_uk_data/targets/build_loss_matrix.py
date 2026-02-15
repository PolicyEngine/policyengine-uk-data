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
from policyengine_uk_data.targets.schema import GeographicLevel, Target, Unit

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

    # Helper closures for the simulation
    ctx = _SimContext(sim, time_period, dataset, reform)

    # Fetch all targets (no year filter — we resolve values below)
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
    """Get the target value for a year, falling back to nearest year."""
    if year in target.values:
        return target.values[year]
    # Use nearest available year
    available = sorted(target.values.keys())
    if not available:
        return None
    closest = min(available, key=lambda y: abs(y - year))
    # Only allow ±3 years of extrapolation
    if abs(closest - year) > 3:
        return None
    return target.values[closest]


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
            self._cache[key] = self.sim.calculate(variable, map_to="household").values
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
            self._cache["region"] = self.sim.calculate("region", map_to="person")
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
        """Lazily create the salary sacrifice counterfactual simulation."""
        if "counterfactual_sim" not in self._cache:
            from policyengine_uk import Microsimulation

            ss = self.sim.calculate("pension_contributions_via_salary_sacrifice")
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


# ── Region name mapping ──────────────────────────────────────────────

_REGION_MAP = {
    "NORTH_EAST": "north_east",
    "SOUTH_EAST": "south_east",
    "EAST_MIDLANDS": "east_midlands",
    "WEST_MIDLANDS": "west_midlands",
    "YORKSHIRE": "yorkshire_and_the_humber",
    "EAST_OF_ENGLAND": "east",
    "LONDON": "london",
    "SOUTH_WEST": "south_west",
    "NORTH_WEST": "north_west",
    "WALES": "wales",
    "SCOTLAND": "scotland",
    "NORTHERN_IRELAND": "northern_ireland",
}

_REGION_INV = {v: k for k, v in _REGION_MAP.items()}


# ── Column computation dispatch ──────────────────────────────────────


def _compute_column(target: Target, ctx: _SimContext, year: int) -> np.ndarray | None:
    """Compute the household-level column for a target.

    Returns None if the target can't be computed (e.g. missing
    custom_compute for a complex target).
    """
    # If the target has a custom compute function, use it
    if target.custom_compute is not None:
        return target.custom_compute(ctx, target, year)

    # Dispatch by target name patterns and metadata
    name = target.name

    # ── Regional age bands ────────────────────────────────────────
    # Names like "ons/north_east_age_0_9"
    if name.startswith("ons/") and "_age_" in name:
        return _compute_regional_age(target, ctx)

    # ── Gender × age bands ────────────────────────────────────────
    # Names like "ons/female_0_14"
    if name.startswith("ons/") and (
        name.startswith("ons/female_") or name.startswith("ons/male_")
    ):
        return _compute_gender_age(target, ctx)

    # ── UK total population ───────────────────────────────────────
    if name == "ons/uk_population":
        return ctx.household_from_person(ctx.age >= 0)

    # ── Scotland-specific demographics ────────────────────────────
    if name == "ons/scotland_children_under_16":
        return ctx.household_from_person(
            (ctx.region.values == "SCOTLAND") & (ctx.age < 16)
        )
    if name == "ons/scotland_babies_under_1":
        return ctx.household_from_person(
            (ctx.region.values == "SCOTLAND") & (ctx.age < 1)
        )
    if name == "ons/scotland_households_3plus_children":
        is_child = ctx.pe_person("is_child")
        children_per_hh = ctx.household_from_person(is_child)
        return ((ctx.household_region == "SCOTLAND") & (children_per_hh >= 3)).astype(
            float
        )

    # ── Household type targets ────────────────────────────────────
    if target.variable == "family_type" and target.is_count:
        return _compute_household_type(target, ctx)

    # ── Tenure targets ────────────────────────────────────────────
    if target.variable == "tenure_type" and target.is_count:
        return _compute_tenure(target, ctx)

    # ── Income band breakdowns (HMRC SPI) ─────────────────────────
    if target.breakdown_variable == "total_income":
        return _compute_income_band(target, ctx)

    # ── Council tax bands by region (VOA) ─────────────────────────
    if name.startswith("voa/council_tax/"):
        return _compute_council_tax_band(target, ctx)

    # ── Vehicle ownership (NTS) ───────────────────────────────────
    if name == "nts/households_no_vehicle":
        return (ctx.pe("num_vehicles") == 0).astype(float)
    if name == "nts/households_one_vehicle":
        return (ctx.pe("num_vehicles") == 1).astype(float)
    if name == "nts/households_two_plus_vehicles":
        return (ctx.pe("num_vehicles") >= 2).astype(float)

    # ── Housing targets ───────────────────────────────────────────
    if name == "housing/total_mortgage":
        return ctx.pe("mortgage_capital_repayment") + ctx.pe(
            "mortgage_interest_repayment"
        )
    if name == "housing/rent_private":
        tenure = ctx.sim.calculate("tenure_type", map_to="household").values
        return ctx.pe("rent") * (tenure == "RENT_PRIVATELY")

    # ── Savings interest (ONS) ────────────────────────────────────
    if name == "ons/savings_interest_income":
        savings = ctx.sim.calculate("savings_interest_income")
        return ctx.household_from_person(savings)

    # ── Scottish child payment ────────────────────────────────────
    if name == "sss/scottish_child_payment":
        scp = ctx.sim.calculate("scottish_child_payment")
        return ctx.household_from_person(scp)

    # ── DWP PIP claimant splits ───────────────────────────────────
    if name == "dwp/pip_dl_standard_claimants":
        pip_dl = ctx.sim.calculate("pip_dl_category")
        return ctx.sim.map_result(pip_dl == "STANDARD", "person", "household")
    if name == "dwp/pip_dl_enhanced_claimants":
        pip_dl = ctx.sim.calculate("pip_dl_category")
        return ctx.sim.map_result(pip_dl == "ENHANCED", "person", "household")

    # ── DWP benefit cap ───────────────────────────────────────────
    if name == "dwp/benefit_capped_households":
        reduction = ctx.sim.calculate(
            "benefit_cap_reduction", map_to="household"
        ).values
        return (reduction > 0).astype(float)
    if name == "dwp/benefit_cap_total_reduction":
        return ctx.sim.calculate(
            "benefit_cap_reduction", map_to="household"
        ).values.astype(float)

    # ── DWP Scotland UC + child under 1 ──────────────────────────
    if name == "dwp/scotland_uc_households_child_under_1":
        uc = ctx.sim.calculate("universal_credit")
        on_uc = ctx.household_from_family(uc > 0) > 0
        child_u1 = ctx.pe_person("is_child") & (ctx.age < 1)
        has_child_u1 = ctx.household_from_person(child_u1) > 0
        return ((ctx.household_region == "SCOTLAND") & on_uc & has_child_u1).astype(
            float
        )

    # ── UC claimants by number of children ─────────────────────────
    if name.startswith("dwp/uc/claimants_with_") and "_children" in name:
        return _compute_uc_by_children(target, ctx)

    # ── UC claimants by family type ──────────────────────────────
    if name.startswith("dwp/uc/claimants_") and not name.startswith(
        "dwp/uc/claimants_with_"
    ):
        return _compute_uc_by_family_type(target, ctx)

    # ── UC payment distribution ───────────────────────────────────
    if name.startswith("dwp/uc_payment_dist/"):
        return _compute_uc_payment_dist(target, ctx)

    # ── Salary sacrifice IT relief by tax band ────────────────────
    if name.startswith("hmrc/salary_sacrifice_it_relief_"):
        return _compute_ss_it_relief(target, ctx)

    # ── Salary sacrifice NI relief ────────────────────────────────
    if name in (
        "hmrc/salary_sacrifice_employee_nics_relief",
        "obr/salary_sacrifice_employee_ni_relief",
    ):
        ni_base = ctx.sim.calculate("ni_employee")
        ni_cf = ctx.counterfactual_sim.calculate("ni_employee", ctx.time_period)
        return ctx.household_from_person(ni_cf - ni_base)
    if name in (
        "hmrc/salary_sacrifice_employer_nics_relief",
        "obr/salary_sacrifice_employer_ni_relief",
    ):
        ni_base = ctx.sim.calculate("ni_employer")
        ni_cf = ctx.counterfactual_sim.calculate("ni_employer", ctx.time_period)
        return ctx.household_from_person(ni_cf - ni_base)

    # ── UC jobseeker / non-jobseeker splits ───────────────────────
    if name in (
        "obr/universal_credit_jobseekers",
        "obr/universal_credit_non_jobseekers",
        "obr/universal_credit_jobseekers_count",
        "obr/universal_credit_non_jobseekers_count",
    ):
        return _compute_uc_jobseeker(target, ctx)

    # ── OBR UC outside benefit cap ────────────────────────────────
    if name == "obr/universal_credit_outside_cap":
        uc = ctx.sim.calculate("universal_credit")
        uc_hh = ctx.household_from_family(uc)
        cap_reduction = ctx.sim.calculate(
            "benefit_cap_reduction", map_to="household"
        ).values
        not_capped = cap_reduction == 0
        return uc_hh * not_capped

    # ── Two-child limit targets ───────────────────────────────────
    if "two_child_limit" in name:
        return _compute_two_child_limit(target, ctx)

    # ── OBR council tax by country ────────────────────────────────
    if name.startswith("obr/council_tax"):
        return _compute_obr_council_tax(target, ctx)

    # ── Simple GBP sum targets ────────────────────────────────────
    if target.unit == Unit.GBP and not target.is_count:
        return _compute_simple_gbp(target, ctx)

    # ── Simple count targets ──────────────────────────────────────
    if target.is_count and target.unit == Unit.COUNT:
        return _compute_simple_count(target, ctx)

    logger.debug("No compute logic for target %s", name)
    return None


# ── Compute implementations ──────────────────────────────────────────


def _compute_simple_gbp(target: Target, ctx: _SimContext) -> np.ndarray:
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


def _compute_regional_age(target: Target, ctx: _SimContext) -> np.ndarray:
    """Compute person count in a region × age band."""
    # Parse "ons/{region_name}_age_{lower}_{upper}" from the name
    name = target.name.removeprefix("ons/")
    # Find the _age_ part
    idx = name.index("_age_")
    region_name = name[:idx]
    age_part = name[idx + 5 :]  # e.g. "0_9"
    lower, upper = age_part.split("_")
    lower, upper = int(lower), int(upper)

    pe_region = _REGION_INV.get(region_name)
    if pe_region is None:
        return None

    person_match = (
        (ctx.region.values == pe_region) & (ctx.age >= lower) & (ctx.age <= upper)
    )
    return ctx.household_from_person(person_match)


def _compute_gender_age(target: Target, ctx: _SimContext) -> np.ndarray:
    """Compute person count in a gender × age band."""
    name = target.name.removeprefix("ons/")
    # "female_0_14" or "male_75_90"
    parts = name.split("_")
    sex = parts[0]
    lower = int(parts[1])
    upper = int(parts[2])

    gender = ctx.sim.calculate("gender").values
    sex_match = gender == ("FEMALE" if sex == "female" else "MALE")
    age_match = (ctx.age >= lower) & (ctx.age <= upper)
    return ctx.household_from_person(sex_match & age_match)


def _compute_household_type(target: Target, ctx: _SimContext) -> np.ndarray | None:
    """Compute household type count from ONS families & households categories.

    Maps ONS household categories to PE family_type enum values and
    household composition conditions. family_type is a benunit variable
    so we map boolean comparisons to household level.
    """
    name = target.name.removeprefix("ons/")
    ft = ctx.sim.calculate("family_type").values  # benunit level
    is_child = ctx.pe_person("is_child")
    children_per_hh = ctx.household_from_person(is_child)
    age_hh_head = ctx.pe("age")  # head of household age

    def ft_hh(value):
        """Map family_type == value from benunit to household (any)."""
        return ctx.household_from_family(ft == value) > 0

    if name == "lone_households_under_65":
        return (ft_hh("SINGLE") & (children_per_hh == 0) & (age_hh_head < 65)).astype(
            float
        )
    if name == "lone_households_over_65":
        return (ft_hh("SINGLE") & (children_per_hh == 0) & (age_hh_head >= 65)).astype(
            float
        )
    if name == "unrelated_adult_households":
        people_per_hh = ctx.household_from_person(np.ones_like(is_child))
        return (ft_hh("SINGLE") & (children_per_hh == 0) & (people_per_hh > 1)).astype(
            float
        )
    if name == "couple_no_children_households":
        return ft_hh("COUPLE_NO_CHILDREN").astype(float)
    if name == "couple_under_3_children_households":
        return (
            ft_hh("COUPLE_WITH_CHILDREN")
            & (children_per_hh >= 1)
            & (children_per_hh <= 2)
        ).astype(float)
    if name == "couple_3_plus_children_households":
        return (ft_hh("COUPLE_WITH_CHILDREN") & (children_per_hh >= 3)).astype(float)
    if name == "couple_non_dependent_children_only_households":
        people_per_hh = ctx.household_from_person(np.ones_like(is_child))
        return (ft_hh("COUPLE_NO_CHILDREN") & (people_per_hh > 2)).astype(float)
    if name == "lone_parent_dependent_children_households":
        return (ft_hh("LONE_PARENT") & (children_per_hh > 0)).astype(float)
    if name == "lone_parent_non_dependent_children_households":
        people_per_hh = ctx.household_from_person(np.ones_like(is_child))
        return (
            ft_hh("SINGLE")
            & (children_per_hh == 0)
            & (people_per_hh > 1)
            & (age_hh_head >= 40)
        ).astype(float)
    if name == "multi_family_households":
        n_benunits = ctx.pe("household_num_benunits")
        return (n_benunits > 1).astype(float)

    return None


def _compute_tenure(target: Target, ctx: _SimContext) -> np.ndarray | None:
    """Compute dwelling count by tenure type."""
    # Map ONS target name suffixes to PE tenure_type enum values
    _TENURE_MAP = {
        "tenure_england_owned_outright": "OWNED_OUTRIGHT",
        "tenure_england_owned_with_mortgage": "OWNED_WITH_MORTGAGE",
        "tenure_england_rented_privately": "RENT_PRIVATELY",
        "tenure_england_social_rent": ["RENT_FROM_COUNCIL", "RENT_FROM_HA"],
        "tenure_england_total": None,  # all tenures
    }
    suffix = target.name.removeprefix("ons/")
    pe_values = _TENURE_MAP.get(suffix)
    if pe_values is None and suffix == "tenure_england_total":
        # Total dwellings in England
        return (ctx.country == "ENGLAND").astype(float)
    if pe_values is None:
        return None

    tenure = ctx.sim.calculate("tenure_type", map_to="household").values
    in_england = ctx.country == "ENGLAND"
    if isinstance(pe_values, list):
        match = np.zeros_like(tenure, dtype=bool)
        for v in pe_values:
            match = match | (tenure == v)
    else:
        match = tenure == pe_values
    return (match & in_england).astype(float)


def _compute_income_band(target: Target, ctx: _SimContext) -> np.ndarray:
    """Compute income variable within a total income band."""
    variable = target.variable
    lower = target.lower_bound
    upper = target.upper_bound

    income_df = ctx.sim.calculate_dataframe(["total_income", variable])
    in_band = (income_df.total_income >= lower) & (income_df.total_income < upper)

    if target.is_count:
        return ctx.household_from_person((income_df[variable] > 0) * in_band)
    else:
        return ctx.household_from_person(income_df[variable] * in_band)


def _compute_council_tax_band(target: Target, ctx: _SimContext) -> np.ndarray:
    """Compute council tax band count for a region."""
    # "voa/council_tax/{REGION}/{band}"
    parts = target.name.split("/")
    region = parts[2]
    band = parts[3]

    in_region = ctx.sim.calculate("region").values == region

    if band == "total":
        return in_region.astype(float)

    in_band = ctx.sim.calculate("council_tax_band") == band
    return (in_band * in_region).astype(float)


def _compute_obr_council_tax(target: Target, ctx: _SimContext) -> np.ndarray:
    """Compute OBR council tax receipts, optionally by country."""
    name = target.name
    ct = ctx.pe("council_tax")

    if name == "obr/council_tax":
        return ct
    if name == "obr/council_tax_england":
        return ct * (ctx.country == "ENGLAND")
    if name == "obr/council_tax_scotland":
        return ct * (ctx.country == "SCOTLAND")
    if name == "obr/council_tax_wales":
        return ct * (ctx.country == "WALES")
    return ct


def _compute_uc_jobseeker(target: Target, ctx: _SimContext) -> np.ndarray:
    """Compute UC jobseeker / non-jobseeker splits."""
    family = ctx.sim.populations["benunit"]
    uc = ctx.sim.calculate("universal_credit")
    on_uc = uc > 0
    unemployed = family.any(ctx.sim.calculate("employment_status") == "UNEMPLOYED")

    if "non_jobseekers" in target.name:
        mask = on_uc * ~unemployed
    else:
        mask = on_uc * unemployed

    if "_count" in target.name:
        return ctx.household_from_family(mask)
    else:
        return ctx.household_from_family(uc * mask)


def _compute_uc_payment_dist(target: Target, ctx: _SimContext) -> np.ndarray:
    """Compute UC payment distribution band × family type."""
    # Parse from name: "dwp/uc_payment_dist/{family_type}_annual_payment_{lower}_to_{upper}"
    name = target.name.removeprefix("dwp/uc_payment_dist/")
    # Find the _annual_payment_ separator
    idx = name.index("_annual_payment_")
    family_type = name[:idx]
    lower = target.lower_bound
    upper = target.upper_bound

    uc_payments = ctx.sim.calculate("universal_credit", map_to="benunit").values
    uc_family_type = ctx.sim.calculate("family_type", map_to="benunit").values

    in_band = (
        (uc_payments >= lower) & (uc_payments < upper) & (uc_family_type == family_type)
    )
    return ctx.household_from_family(in_band)


def _compute_ss_it_relief(target: Target, ctx: _SimContext) -> np.ndarray:
    """Compute salary sacrifice IT relief by tax band."""
    it_base = ctx.sim.calculate("income_tax")
    it_cf = ctx.counterfactual_sim.calculate("income_tax", ctx.time_period)
    it_relief = it_cf - it_base

    adj_net_income_cf = ctx.counterfactual_sim.calculate(
        "adjusted_net_income", ctx.time_period
    )

    params = ctx.sim.tax_benefit_system.parameters.gov.hmrc.income_tax.rates.uk
    basic_thresh = params[0].threshold(ctx.time_period)
    higher_thresh = params[1].threshold(ctx.time_period)
    additional_thresh = params[2].threshold(ctx.time_period)

    name = target.name
    if "basic" in name:
        mask = (adj_net_income_cf > basic_thresh) & (adj_net_income_cf <= higher_thresh)
    elif "higher" in name:
        mask = (adj_net_income_cf > higher_thresh) & (
            adj_net_income_cf <= additional_thresh
        )
    elif "additional" in name:
        mask = adj_net_income_cf > additional_thresh
    else:
        # Total — no mask
        mask = np.ones_like(it_relief, dtype=bool)

    return ctx.household_from_person(it_relief * mask)


def _compute_two_child_limit(target: Target, ctx: _SimContext) -> np.ndarray | None:
    """Compute two-child limit targets.

    These involve cross-tabulations of UC eligibility, child count,
    disability status, etc. Complex enough to need specific logic
    per target name.
    """
    name = target.name
    sim = ctx.sim

    is_child = sim.calculate("is_child").values
    child_is_affected = (
        sim.map_result(
            sim.calculate("uc_is_child_limit_affected", map_to="household"),
            "household",
            "person",
        )
        > 0
    ) * is_child
    child_in_uc = sim.calculate("universal_credit", map_to="person").values > 0
    children_in_capped = sim.map_result(
        child_is_affected * child_in_uc, "person", "household"
    )
    capped_hh = (children_in_capped > 0) * 1.0

    if name == "dwp/uc/two_child_limit/households_affected":
        return capped_hh
    if name == "dwp/uc/two_child_limit/children_affected":
        return children_in_capped
    if name == "dwp/uc/two_child_limit/children_in_affected_households":
        # Total children (not just affected ones) in capped households
        total_children = sim.map_result(is_child * child_in_uc, "person", "household")
        return total_children * capped_hh

    # By number of children: "dwp/uc/two_child_limit/{n}_children_households"
    if "_children_households_total_children" in name:
        n = int(name.split("/")[-1].split("_")[0])
        children_count = sim.map_result(is_child, "person", "household")
        return (capped_hh * (children_count == n) * children_count).astype(float)
    if "_children_households" in name and "total" not in name:
        n = int(name.split("/")[-1].split("_")[0])
        children_count = sim.map_result(is_child, "person", "household")
        match = n if n < 6 else slice(6, None)
        if isinstance(match, int):
            return (capped_hh * (children_count == n)).astype(float)
        else:
            return (capped_hh * (children_count >= 6)).astype(float)

    # Disability cross-tabs
    if "adult_pip_households" in name:
        pip = sim.calculate("pip", map_to="household").values
        return (capped_hh * (pip > 0)).astype(float)
    if "adult_pip_children" in name:
        pip = sim.calculate("pip", map_to="household").values
        return (children_in_capped * (pip > 0)).astype(float)
    if "disabled_child_element_households" in name:
        dce = sim.calculate(
            "uc_individual_disabled_child_element", map_to="household"
        ).values
        return (capped_hh * (dce > 0)).astype(float)
    if "disabled_child_element_children" in name:
        dce = sim.calculate(
            "uc_individual_disabled_child_element", map_to="household"
        ).values
        return (children_in_capped * (dce > 0)).astype(float)

    return None


def _compute_uc_by_children(target: Target, ctx: _SimContext) -> np.ndarray:
    """Compute UC claimant households filtered by number of dependent children."""
    # Parse "dwp/uc/claimants_with_{n}_children"
    name = target.name
    n_str = name.split("claimants_with_")[1].split("_children")[0]

    uc = ctx.sim.calculate("universal_credit")
    on_uc = ctx.household_from_family(uc > 0) > 0

    is_child = ctx.pe_person("is_child")
    children_per_hh = ctx.household_from_person(is_child)

    if n_str.endswith("+"):
        n = int(n_str[:-1])
        match = children_per_hh >= n
    else:
        n = int(n_str)
        match = children_per_hh == n

    return (on_uc & match).astype(float)


def _compute_uc_by_family_type(target: Target, ctx: _SimContext) -> np.ndarray:
    """Compute UC claimant households filtered by family type."""
    name = target.name
    ft_str = name.split("dwp/uc/claimants_")[1]

    uc = ctx.sim.calculate("universal_credit")
    on_uc = ctx.household_from_family(uc > 0) > 0

    ft = ctx.sim.calculate("family_type").values  # benunit level

    def ft_hh(value):
        return ctx.household_from_family(ft == value) > 0

    is_child = ctx.pe_person("is_child")
    children_per_hh = ctx.household_from_person(is_child)

    if ft_str == "single_no_children":
        match = ft_hh("SINGLE") & (children_per_hh == 0)
    elif ft_str == "single_with_children":
        match = (ft_hh("SINGLE") | ft_hh("LONE_PARENT")) & (children_per_hh > 0)
    elif ft_str == "couple_no_children":
        match = ft_hh("COUPLE_NO_CHILDREN")
    elif ft_str == "couple_with_children":
        match = ft_hh("COUPLE_WITH_CHILDREN")
    else:
        return None

    return (on_uc & match).astype(float)
