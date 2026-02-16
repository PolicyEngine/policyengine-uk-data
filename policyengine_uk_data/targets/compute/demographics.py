"""Demographic target compute functions."""

import numpy as np

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


def compute_regional_age(target, ctx) -> np.ndarray | None:
    """Compute person count in a region x age band."""
    name = target.name.removeprefix("ons/")
    idx = name.index("_age_")
    region_name = name[:idx]
    age_part = name[idx + 5 :]
    lower, upper = age_part.split("_")
    lower, upper = int(lower), int(upper)

    pe_region = _REGION_INV.get(region_name)
    if pe_region is None:
        return None

    person_match = (
        (ctx.region.values == pe_region)
        & (ctx.age >= lower)
        & (ctx.age <= upper)
    )
    return ctx.household_from_person(person_match)


def compute_gender_age(target, ctx) -> np.ndarray:
    """Compute person count in a gender x age band."""
    name = target.name.removeprefix("ons/")
    parts = name.split("_")
    sex = parts[0]
    lower = int(parts[1])
    upper = int(parts[2])

    gender = ctx.sim.calculate("gender").values
    sex_match = gender == ("FEMALE" if sex == "female" else "MALE")
    age_match = (ctx.age >= lower) & (ctx.age <= upper)
    return ctx.household_from_person(sex_match & age_match)


def compute_uk_population(target, ctx) -> np.ndarray:
    """Compute UK total population column."""
    return ctx.household_from_person(ctx.age >= 0)


def compute_scotland_demographics(target, ctx) -> np.ndarray | None:
    """Compute Scotland-specific demographic targets."""
    name = target.name
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
        return (
            (ctx.household_region == "SCOTLAND") & (children_per_hh >= 3)
        ).astype(float)
    return None
