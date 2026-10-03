"""Broad Rental Market Area (BRMA) assignment for FRS households.

The FRS identifies only a household's region, but Local Housing Allowance
rates vary by BRMA. Each benefit unit is given a BRMA drawn within its
region, in proportion to the number of private-rented households in each
BRMA with the number of bedrooms matching its LHA category (shared
accommodation and one-bedroom categories both use one-bedroom homes).

The counts come from the censuses (England and Wales 2021, Scotland 2022,
Northern Ireland 2021) mapped to BRMAs; ``storage/BRMA_DATA_SOURCES.md``
gives the sources, method and validation. A BRMA that crosses a region
boundary appears once per region, holding the households on that side.

These weights replace the row counts of the 2019-20 LHA list of rents. Its
Scottish, Welsh and Northern Ireland rows were copies of English BRMAs'
lists, so their counts said nothing about those nations' rental markets.

The base FRS dataset gets one draw per household (``assign_brmas`` and
``pick_household_brmas``). One draw is noisy: a few heavily weighted private
renters carry most of the variation, so a different seed moved UK Universal
Credit by about £0.3bn. The enhanced dataset therefore splits each
private-renting household into ``k`` records (``split_private_renters_across_brmas``),
each with ``1/k`` of its weight and one of ``k`` evenly spaced quantiles of its
BRMA distribution, walking its region's BRMAs from the lowest LHA rate to the
highest. Each record keeps exactly the distribution of a single draw; the
records' average follows the expectation far more closely. Calibration ties
the ``k`` records of a household (``BRMA_SPLIT_GROUP_COLUMN``) so that they
keep equal weights.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from policyengine_uk_data.storage import STORAGE_FOLDER

BRMA_HOUSEHOLDS_PATH = STORAGE_FOLDER / "brma_private_rented_households.csv"

# Census bedroom band whose private-rented households weight each LHA category.
LHA_CATEGORY_BEDROOMS = {"A": "1", "B": "1", "C": "2", "D": "3", "E": "4+"}

# Household column holding the id of the household a split record came from.
BRMA_SPLIT_GROUP_COLUMN = "brma_split_group"


def load_brma_weights(path: Path = BRMA_HOUSEHOLDS_PATH) -> pd.DataFrame:
    """Return BRMA sampling weights by region and LHA category.

    Columns: ``region``, ``lha_category``, ``brma``, ``weight`` (private-rented
    households). Every row has a positive weight. Northern Ireland's census
    has no bedrooms question, so its rows (bedrooms ``all``) weight every
    category.
    """
    households = pd.read_csv(path, dtype={"bedrooms": str})
    bands = pd.DataFrame(
        LHA_CATEGORY_BEDROOMS.items(), columns=["lha_category", "bedrooms"]
    )
    by_band = bands.merge(households, on="bedrooms")
    any_band = households[households.bedrooms == "all"].merge(
        bands[["lha_category"]], how="cross"
    )
    weights = pd.concat([by_band, any_band], ignore_index=True)
    weights = weights[weights.households > 0]
    return weights.rename(columns={"households": "weight"})[
        ["region", "lha_category", "brma", "weight"]
    ].reset_index(drop=True)


def assign_brmas(
    region: np.ndarray,
    lha_category: np.ndarray,
    rng: np.random.Generator,
    weights: pd.DataFrame | None = None,
) -> np.ndarray:
    """Draw a BRMA for each benefit unit from its region × LHA category cell.

    Args:
        region: Region name of each benefit unit's household.
        lha_category: LHA category (A-E) of each benefit unit.
        rng: Generator for the draws.
        weights: Output of ``load_brma_weights``; loaded from storage if omitted.

    Raises:
        ValueError: If a benefit unit's cell has no BRMA with a positive weight.
    """
    if weights is None:
        weights = load_brma_weights()
    if hasattr(region, "decode_to_str"):  # EnumArray of region codes
        region = region.decode_to_str()
    region = np.asarray(region).astype(str)
    lha_category = np.asarray(lha_category).astype(str)
    brma = np.empty(len(region), dtype=object)
    for (cell_region, category), cell in weights.groupby(
        ["region", "lha_category"], sort=True
    ):
        mask = (region == cell_region) & (lha_category == category)
        if mask.any():
            p = cell.weight.to_numpy(dtype=float)
            brma[mask] = rng.choice(
                cell.brma.to_numpy(), size=mask.sum(), p=p / p.sum()
            )
    missing = pd.isna(brma)
    if missing.any():
        cells = sorted(set(zip(region[missing], lha_category[missing])))
        raise ValueError(f"No BRMA weights for region × LHA category cells {cells}.")
    return brma


def pick_household_brmas(
    brma: np.ndarray, household_id: np.ndarray, rng: np.random.Generator
) -> pd.Series:
    """Give each household the BRMA of one of its benefit units, chosen at random.

    ``brma`` and ``household_id`` are per benefit unit; returns a Series
    indexed by household ID.
    """
    units = pd.DataFrame({"brma": brma, "household_id": household_id})
    return units.groupby("household_id").brma.aggregate(
        lambda x: x.sample(n=1, random_state=rng).iloc[0]
    )


def household_brma_probabilities(
    region: np.ndarray,
    benunit_household: np.ndarray,
    benunit_category: np.ndarray,
    weights: pd.DataFrame | None = None,
) -> tuple[dict[str, list[str]], np.ndarray]:
    """Each household's BRMA distribution under ``assign_brmas`` and ``pick_household_brmas``.

    Each benefit unit draws from its region × LHA category cell and the
    household takes one unit's BRMA at random, so the household's distribution
    is the mean of its units' cell distributions.

    Args:
        region: Region name of each household.
        benunit_household: Index into ``region`` of each benefit unit's household.
        benunit_category: LHA category (A-E) of each benefit unit.
        weights: Output of ``load_brma_weights``; loaded from storage if omitted.

    Returns:
        ``(brmas, probabilities)``: ``brmas`` maps each region to its BRMA names
        in alphabetical order; ``probabilities[h, j]`` is household h's
        probability of the j-th BRMA of its region (zero past the region's count).

    Raises:
        ValueError: If a household has no benefit unit, or a unit's cell has no
            BRMA with a positive weight.
    """
    if weights is None:
        weights = load_brma_weights()
    region = np.asarray(region).astype(str)
    benunit_household = np.asarray(benunit_household)
    benunit_category = np.asarray(benunit_category).astype(str)
    brmas = {r: sorted(g.brma.unique()) for r, g in weights.groupby("region")}
    width = max(len(names) for names in brmas.values())
    probabilities = np.zeros((len(region), width))
    units = np.bincount(benunit_household, minlength=len(region))
    unit_region = region[benunit_household]
    covered = np.zeros(len(benunit_household), bool)
    for (cell_region, category), cell in weights.groupby(["region", "lha_category"]):
        mask = (unit_region == cell_region) & (benunit_category == category)
        if not mask.any():
            continue
        p = cell.set_index("brma").weight.reindex(brmas[cell_region]).fillna(0)
        if not (p >= 0).all() or p.sum() <= 0:
            raise ValueError(
                f"No BRMA weights for region × LHA category cells "
                f"{[(cell_region, category)]}."
            )
        p = np.pad(p.to_numpy(float) / p.sum(), (0, width - len(p)))
        np.add.at(probabilities, benunit_household[mask], p)
        covered |= mask
    if (units == 0).any():
        raise ValueError("Every household needs at least one benefit unit.")
    if not covered.all():
        cells = sorted(set(zip(unit_region[~covered], benunit_category[~covered])))
        raise ValueError(f"No BRMA weights for region × LHA category cells {cells}.")
    return brmas, probabilities / units[:, None]


def lha_rate_keys(
    simulation,
    year: int,
    brmas: dict[str, list[str]],
    region: np.ndarray,
    benunit_household: np.ndarray,
) -> np.ndarray:
    """Each household's LHA rate in each BRMA of its region.

    The mean, over the household's benefit units, of PolicyEngine UK's
    ``uncapped_BRMA_LHA_rate`` (``BRMA_LHA_rate`` in versions without it) with
    the household moved to that BRMA; ``inf`` past the region's count. It only
    orders BRMAs for the spaced draws: any order keeps every household's
    distribution exact.
    """
    region = np.asarray(region).astype(str)
    count = np.array([len(brmas[r]) for r in region])
    width = max(len(names) for names in brmas.values())
    units = np.maximum(np.bincount(benunit_household, minlength=len(region)), 1)
    keys = np.full((len(region), width), np.inf)
    variable = "uncapped_BRMA_LHA_rate"
    if variable not in simulation.tax_benefit_system.variables:
        variable = "BRMA_LHA_rate"
    for j in range(width):
        slot = np.minimum(j, count - 1)
        names = np.array([brmas[r][i] for r, i in zip(region, slot)])
        simulation.set_input("brma", year, names)
        simulation.delete_arrays(variable)
        rate = np.asarray(simulation.calculate(variable, year))
        mean = np.bincount(benunit_household, weights=rate, minlength=len(region))
        keys[:, j] = np.where(j < count, mean / units, np.inf)
    simulation.delete_arrays(variable)
    return keys


def brmas_at_quantiles(
    probabilities: np.ndarray, keys: np.ndarray | None, quantiles: np.ndarray
) -> np.ndarray:
    """Column of each row's distribution at its quantile in [0, 1).

    Columns are walked in ascending ``keys`` order (ties, and ``keys=None``, by
    column). A column is returned only where its probability is positive: the
    cumulative probability is set to exactly 1 from the last positive column
    on, so rounding cannot carry a quantile past it.
    """
    quantiles = np.asarray(quantiles, float)
    if np.any((quantiles < 0) | (quantiles >= 1)):
        raise ValueError("Quantiles must lie in [0, 1).")
    rows, width = probabilities.shape
    if keys is None:
        order = np.broadcast_to(np.arange(width), probabilities.shape)
    else:
        order = np.argsort(
            np.where(probabilities > 0, keys, np.inf), axis=1, kind="stable"
        )
    p = np.take_along_axis(probabilities, order, axis=1)
    cumulative = np.cumsum(p, axis=1)
    last = width - 1 - np.argmax(p[:, ::-1] > 0, axis=1)
    cumulative[np.arange(width)[None, :] >= last[:, None]] = 1.0
    position = (cumulative <= quantiles[:, None]).sum(axis=1)
    return order[np.arange(rows), position]


def spaced_quantiles(rows: int, k: int, rng: np.random.Generator) -> np.ndarray:
    """``(rows, k)`` quantiles ``(u + i / k) mod 1`` with one uniform ``u`` per row.

    Each quantile is uniform on [0, 1), and a row's ``k`` quantiles fall one
    in each interval of width ``1 / k``.
    """
    if k < 1:
        raise ValueError("k must be at least 1.")
    u = rng.random(rows)
    return (u[:, None] + np.arange(k)[None, :] / k) % 1.0


def split_private_renters_across_brmas(
    dataset, k: int, seed: int = 0, weights: pd.DataFrame | None = None
):
    """Split each private-renting household into ``k`` records across its BRMAs.

    Each record has ``1/k`` of the household's weight and the BRMA at one of
    ``k`` evenly spaced quantiles of the household's distribution
    (``household_brma_probabilities``), walking its region's BRMAs from the
    lowest LHA rate to the highest. Other households keep their record and
    BRMA. Every household gets ``BRMA_SPLIT_GROUP_COLUMN``: the id of the
    household its record came from.

    Copies offset household ids by ``i * step`` (``step`` a power of ten above
    every household id), benefit unit ids by ``i * step * 100`` and person ids
    by ``i * step * 1000``, each raised to a power of ten above that table's
    largest id if needed so no id collides. When no raise is needed,
    ``benunit_id // 100 == household_id`` still holds wherever it held. Every
    benefit unit must have at least one person, and the household and benefit
    unit tables must be sorted by id (as PolicyEngine UK needs). Each record also gets
    ``source_household_id``, the household it came from, which the OA clone
    step keeps, so split records count as one source household.

    Returns:
        The new dataset. With ``k == 1`` it is the input plus the group column.
    """
    from policyengine_uk import Microsimulation
    from policyengine_uk.data import UKSingleYearDataset

    household = dataset.household.copy()
    household[BRMA_SPLIT_GROUP_COLUMN] = household.household_id.to_numpy()
    household["source_household_id"] = household.household_id.to_numpy()
    if k == 1:
        return UKSingleYearDataset(
            person=dataset.person,
            benunit=dataset.benunit,
            household=household,
            fiscal_year=dataset.time_period,
        )
    person, benunit = dataset.person, dataset.benunit
    owner = (
        person.drop_duplicates("person_benunit_id")
        .set_index("person_benunit_id")
        .person_household_id
    )
    empty = ~benunit.benunit_id.isin(owner.index)
    if empty.any():
        raise ValueError(
            f"{int(empty.sum())} benefit units have no people; every benefit "
            "unit needs one to place it in a household."
        )
    benunit_owner = owner.loc[benunit.benunit_id].to_numpy()
    year = int(dataset.time_period)
    simulation = Microsimulation(dataset=dataset)
    for entity, ids in (
        ("household", household.household_id),
        ("benunit", benunit.benunit_id),
    ):
        # PolicyEngine UK orders each entity by id but reads input columns in
        # table order, so the two orders must agree.
        if not np.array_equal(simulation.populations[entity].ids, ids.to_numpy()):
            raise ValueError(
                f"The {entity} table is not sorted by {entity}_id, so the "
                "simulation would misalign its rows."
            )
    tenure = np.asarray(simulation.calculate("tenure_type", year)).astype(str)
    renter = tenure == "RENT_PRIVATELY"
    if not renter.any():
        return UKSingleYearDataset(
            person=person,
            benunit=benunit,
            household=household,
            fiscal_year=dataset.time_period,
        )
    region = np.asarray(simulation.calculate("region", year)).astype(str)
    position = pd.Series(
        np.arange(len(household)), index=household.household_id.to_numpy()
    )
    benunit_household = position.loc[benunit_owner].to_numpy()
    category = np.asarray(simulation.calculate("LHA_category", year)).astype(str)
    brmas, probabilities = household_brma_probabilities(
        region, benunit_household, category, weights
    )
    keys = lha_rate_keys(simulation, year, brmas, region, benunit_household)
    columns = brmas_at_quantiles(
        np.repeat(probabilities[renter], k, axis=0),
        np.repeat(keys[renter], k, axis=0),
        spaced_quantiles(int(renter.sum()), k, np.random.default_rng(seed)).ravel(),
    ).reshape(-1, k)
    names = np.array(
        [[brmas[r][j] for j in row] for r, row in zip(region[renter], columns)],
        dtype=object,
    ).reshape(-1, k)

    renter_ids = household.household_id.to_numpy()[renter]
    renter_benunits = benunit[np.isin(benunit_owner, renter_ids)]
    renter_people = person[person.person_household_id.isin(renter_ids)]
    step = 10 ** len(str(int(household.household_id.max())))
    benunit_step = max(step * 100, 10 ** len(str(int(benunit.benunit_id.max()))))
    person_step = max(step * 1000, 10 ** len(str(int(person.person_id.max()))))

    base = household.copy()
    base.loc[renter, "brma"] = names[:, 0]
    base.loc[renter, "household_weight"] = household.household_weight[renter] / k
    households, benunits, people = [base], [benunit], [person]
    for i in range(1, k):
        copy = base[renter].copy()
        copy["brma"] = names[:, i]
        copy["household_id"] += i * step
        households.append(copy)
        units = renter_benunits.copy()
        units["benunit_id"] += i * benunit_step
        benunits.append(units)
        members = renter_people.copy()
        members["person_household_id"] += i * step
        members["person_benunit_id"] += i * benunit_step
        members["person_id"] += i * person_step
        people.append(members)
    result = UKSingleYearDataset(
        person=pd.concat(people, ignore_index=True),
        benunit=pd.concat(benunits, ignore_index=True),
        household=pd.concat(households, ignore_index=True),
        fiscal_year=dataset.time_period,
    )
    for table, column in (
        (result.household, "household_id"),
        (result.benunit, "benunit_id"),
        (result.person, "person_id"),
    ):
        if not table[column].is_unique:
            raise ValueError(f"Splitting private renters duplicated {column}.")
    return result
