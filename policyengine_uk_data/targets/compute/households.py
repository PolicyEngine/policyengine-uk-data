"""Household type and tenure compute functions."""

import numpy as np


def compute_household_type(target, ctx) -> np.ndarray | None:
    """Compute household type count from ONS categories."""
    name = target.name.removeprefix("ons/")
    ft = ctx.sim.calculate("family_type").values
    is_child = ctx.pe_person("is_child")
    children_per_hh = ctx.household_from_person(is_child)
    age_hh_head = ctx.pe("age")

    def ft_hh(value):
        return ctx.household_from_family(ft == value) > 0

    if name == "lone_households_under_65":
        return (
            ft_hh("SINGLE") & (children_per_hh == 0) & (age_hh_head < 65)
        ).astype(float)
    if name == "lone_households_over_65":
        return (
            ft_hh("SINGLE") & (children_per_hh == 0) & (age_hh_head >= 65)
        ).astype(float)
    if name == "unrelated_adult_households":
        people_per_hh = ctx.household_from_person(np.ones_like(is_child))
        return (
            ft_hh("SINGLE") & (children_per_hh == 0) & (people_per_hh > 1)
        ).astype(float)
    if name == "couple_no_children_households":
        return ft_hh("COUPLE_NO_CHILDREN").astype(float)
    if name == "couple_under_3_children_households":
        return (
            ft_hh("COUPLE_WITH_CHILDREN")
            & (children_per_hh >= 1)
            & (children_per_hh <= 2)
        ).astype(float)
    if name == "couple_3_plus_children_households":
        return (ft_hh("COUPLE_WITH_CHILDREN") & (children_per_hh >= 3)).astype(
            float
        )
    if name == "couple_non_dependent_children_only_households":
        people_per_hh = ctx.household_from_person(np.ones_like(is_child))
        return (ft_hh("COUPLE_NO_CHILDREN") & (people_per_hh > 2)).astype(
            float
        )
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


def compute_tenure(target, ctx) -> np.ndarray | None:
    """Compute dwelling count by tenure type."""
    _TENURE_MAP = {
        "tenure_england_owned_outright": "OWNED_OUTRIGHT",
        "tenure_england_owned_with_mortgage": "OWNED_WITH_MORTGAGE",
        "tenure_england_rented_privately": "RENT_PRIVATELY",
        "tenure_england_social_rent": [
            "RENT_FROM_COUNCIL",
            "RENT_FROM_HA",
        ],
        "tenure_england_total": None,
    }
    suffix = target.name.removeprefix("ons/")
    pe_values = _TENURE_MAP.get(suffix)
    if pe_values is None and suffix == "tenure_england_total":
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
