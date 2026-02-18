"""Benefit-related compute functions (UC, PIP, benefit cap, etc)."""

import numpy as np


def compute_pip_claimants(target, ctx) -> np.ndarray:
    """Compute PIP daily living standard/enhanced claimant counts."""
    pip_dl = ctx.sim.calculate("pip_dl_category")
    if "standard" in target.name:
        return ctx.sim.map_result(pip_dl == "STANDARD", "person", "household")
    return ctx.sim.map_result(pip_dl == "ENHANCED", "person", "household")


def compute_benefit_cap(target, ctx) -> np.ndarray:
    """Compute benefit cap targets."""
    if "total_reduction" in target.name:
        return ctx.sim.calculate(
            "benefit_cap_reduction", map_to="household"
        ).values.astype(float)
    reduction = ctx.sim.calculate(
        "benefit_cap_reduction", map_to="household"
    ).values
    return (reduction > 0).astype(float)


def compute_scotland_uc_child(target, ctx) -> np.ndarray:
    """Compute Scotland UC households with child under 1."""
    uc = ctx.sim.calculate("universal_credit")
    on_uc = ctx.household_from_family(uc > 0) > 0
    child_u1 = ctx.pe_person("is_child") & (ctx.age < 1)
    has_child_u1 = ctx.household_from_person(child_u1) > 0
    return (
        (ctx.household_region == "SCOTLAND") & on_uc & has_child_u1
    ).astype(float)


def compute_uc_by_children(target, ctx) -> np.ndarray:
    """Compute UC claimant households by number of children."""
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


def compute_uc_by_family_type(target, ctx) -> np.ndarray | None:
    """Compute UC claimant households by family type."""
    name = target.name
    ft_str = name.split("dwp/uc/claimants_")[1]

    uc = ctx.sim.calculate("universal_credit")
    on_uc = ctx.household_from_family(uc > 0) > 0

    ft = ctx.sim.calculate("family_type").values

    def ft_hh(value):
        return ctx.household_from_family(ft == value) > 0

    is_child = ctx.pe_person("is_child")
    children_per_hh = ctx.household_from_person(is_child)

    if ft_str == "single_no_children":
        match = ft_hh("SINGLE") & (children_per_hh == 0)
    elif ft_str == "single_with_children":
        match = (ft_hh("SINGLE") | ft_hh("LONE_PARENT")) & (
            children_per_hh > 0
        )
    elif ft_str == "couple_no_children":
        match = ft_hh("COUPLE_NO_CHILDREN")
    elif ft_str == "couple_with_children":
        match = ft_hh("COUPLE_WITH_CHILDREN")
    else:
        return None

    return (on_uc & match).astype(float)


def compute_uc_payment_dist(target, ctx) -> np.ndarray:
    """Compute UC payment distribution band x family type."""
    name = target.name.removeprefix("dwp/uc_payment_dist/")
    idx = name.index("_annual_payment_")
    family_type = name[:idx]
    lower = target.lower_bound
    upper = target.upper_bound

    uc_payments = ctx.sim.calculate(
        "universal_credit", map_to="benunit"
    ).values
    uc_family_type = ctx.sim.calculate("family_type", map_to="benunit").values

    in_band = (
        (uc_payments >= lower)
        & (uc_payments < upper)
        & (uc_family_type == family_type)
    )
    return ctx.household_from_family(in_band)


def compute_uc_jobseeker(target, ctx) -> np.ndarray:
    """Compute UC jobseeker / non-jobseeker splits."""
    family = ctx.sim.populations["benunit"]
    uc = ctx.sim.calculate("universal_credit")
    on_uc = uc > 0
    unemployed = family.any(
        ctx.sim.calculate("employment_status") == "UNEMPLOYED"
    )

    if "non_jobseekers" in target.name:
        mask = on_uc * ~unemployed
    else:
        mask = on_uc * unemployed

    if "_count" in target.name:
        return ctx.household_from_family(mask)
    else:
        return ctx.household_from_family(uc * mask)


def compute_uc_outside_cap(target, ctx) -> np.ndarray:
    """Compute OBR UC outside benefit cap."""
    uc = ctx.sim.calculate("universal_credit")
    uc_hh = ctx.household_from_family(uc)
    cap_reduction = ctx.sim.calculate(
        "benefit_cap_reduction", map_to="household"
    ).values
    not_capped = cap_reduction == 0
    return uc_hh * not_capped


def compute_two_child_limit(target, ctx) -> np.ndarray | None:
    """Compute two-child limit targets."""
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
        total_children = sim.map_result(
            is_child * child_in_uc, "person", "household"
        )
        return total_children * capped_hh

    if "_children_households_total_children" in name:
        n = int(name.split("/")[-1].split("_")[0])
        children_count = sim.map_result(is_child, "person", "household")
        return (capped_hh * (children_count == n) * children_count).astype(
            float
        )
    if "_children_households" in name and "total" not in name:
        n = int(name.split("/")[-1].split("_")[0])
        children_count = sim.map_result(is_child, "person", "household")
        if n < 6:
            return (capped_hh * (children_count == n)).astype(float)
        else:
            return (capped_hh * (children_count >= 6)).astype(float)

    if "adult_pip_households" in name:
        pip = sim.calculate("pip", map_to="household").values
        return (capped_hh * (pip > 0)).astype(float)
    if "adult_pip_children" in name:
        pip = sim.calculate("pip", map_to="household").values
        return (children_in_capped * (pip > 0)).astype(float)
    if "disabled_child_element_households" in name:
        dce = sim.calculate(
            "uc_individual_disabled_child_element",
            map_to="household",
        ).values
        return (capped_hh * (dce > 0)).astype(float)
    if "disabled_child_element_children" in name:
        dce = sim.calculate(
            "uc_individual_disabled_child_element",
            map_to="household",
        ).values
        return (children_in_capped * (dce > 0)).astype(float)

    return None
