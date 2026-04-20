"""Household transition mechanics for the panel pipeline.

Complements ``utils.demographic_ageing`` (which handles mortality,
fertility and the age increment) with the life-cycle events that
change household and benefit-unit composition:

- ``apply_marriages``: pair up single adults and merge their benunits.
- ``apply_separations``: split married benunits into two singles; children
  attach to the mother by default.
- ``apply_children_leaving_home``: move adult dependents out of their
  parents' benunit + household into a new one of their own.
- ``apply_migration``: add immigrant rows and drop emigrant rows.

Within-household income / employment change lives in
``employment_transitions.py``; see ``advance_year`` for the composed
pipeline.

Design notes
------------

Benefit-unit semantics. ``is_married`` in pe-uk is derived from
``benunit_count_adults == 2``. Marriage therefore means merging two
single benunits into one, and separation means splitting a two-adult
benunit into two. Neither requires rewriting any stored boolean — only
``person_benunit_id`` (and, for leaving-home and separation,
``person_household_id``). Benefit-unit weights are preserved by
summing the weights of the merging units.

Randomness. All functions take an explicit ``numpy.random.Generator``
so the panel pipeline stays fully reproducible given a seed. They are
pure w.r.t. the input dataset: ``UKSingleYearDataset.copy`` is used at
the top of each function and never propagates writes back.

Out of scope for this module: same-sex pairing (we match by opposite
sex for marriages), multi-benunit households not assembled from a
parent couple, and second / subsequent marriages. These are listed in
the docstring of each function where they matter.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

from policyengine_uk.data import UKSingleYearDataset

from policyengine_uk_data.utils.demographic_ageing import (
    AGE_COLUMN,
    FEMALE_VALUE,
    MALE_VALUE,
    SEX_COLUMN,
)


# -- Age-indexed marriage rates, per unmarried adult per year. ---------------
# Sourced from ONS Marriage Statistics for England and Wales, historic
# post-2000 averages. They are shipped so the module is usable end-to-end;
# callers can override with a fresher or region-specific table via the
# ``marriage_rates`` argument.
#
# Reference:
# https://www.ons.gov.uk/peoplepopulationandcommunity/birthsdeathsandmarriages/marriagecohabitationandcivilpartnerships/datasets/marriagesinenglandandwales2013
DEFAULT_MARRIAGE_RATES: dict[str, dict[int, float]] = {
    MALE_VALUE: {
        **{age: 0.0 for age in range(0, 18)},
        **{age: 0.002 for age in range(18, 20)},
        **{age: 0.010 for age in range(20, 25)},
        **{age: 0.030 for age in range(25, 30)},
        **{age: 0.035 for age in range(30, 35)},
        **{age: 0.025 for age in range(35, 40)},
        **{age: 0.015 for age in range(40, 50)},
        **{age: 0.008 for age in range(50, 65)},
        **{age: 0.003 for age in range(65, 121)},
    },
    FEMALE_VALUE: {
        **{age: 0.0 for age in range(0, 18)},
        **{age: 0.004 for age in range(18, 20)},
        **{age: 0.018 for age in range(20, 25)},
        **{age: 0.035 for age in range(25, 30)},
        **{age: 0.032 for age in range(30, 35)},
        **{age: 0.020 for age in range(35, 40)},
        **{age: 0.010 for age in range(40, 50)},
        **{age: 0.004 for age in range(50, 65)},
        **{age: 0.001 for age in range(65, 121)},
    },
}


# ── Marriage ────────────────────────────────────────────────────────────────


def apply_marriages(
    dataset: UKSingleYearDataset,
    marriage_rates: Mapping[str, Mapping[int, float]] | None = None,
    rng: np.random.Generator | None = None,
) -> UKSingleYearDataset:
    """Pair up single adults and merge their benunits.

    For each single adult, draw a Bernoulli(rate[age, sex]) — those who
    succeed enter a "want to marry" pool and are matched to a partner
    of the opposite sex in the same region, preferring partners of
    similar age. The two partners' benunits are merged into a single
    joint benunit; the household of the first partner is kept and the
    second partner (plus any dependents of their benunit) moves in.

    Args:
        dataset: input panel-ready dataset. Not mutated.
        marriage_rates: ``{"MALE": {age: p}, "FEMALE": {age: p}}``. ``None``
            uses :data:`DEFAULT_MARRIAGE_RATES`. Pass an empty dict (``{}``)
            to disable marriages entirely.
        rng: random number generator. ``None`` constructs a default one.

    Returns:
        A new dataset with ``person_benunit_id`` and
        ``person_household_id`` rewritten for newlywed movers, updated
        benunit and household tables. Weights on the surviving units
        are preserved by summing incoming weights.

    Out of scope: same-sex marriage (we match M↔F only), pre-existing
    cohabiting couples outside a benunit (not a shape the FRS carries),
    and adult children leaving with a spouse (handled by the
    leaving-home transition).
    """
    if rng is None:
        rng = np.random.default_rng()

    if marriage_rates is None:
        marriage_rates = DEFAULT_MARRIAGE_RATES
    if not marriage_rates:
        return dataset

    ds = dataset.copy()
    ds.person = ds.person.copy(deep=True)
    ds.benunit = ds.benunit.copy(deep=True)
    ds.household = ds.household.copy(deep=True)

    singles = _identify_single_adults(ds)
    if singles.empty:
        return ds

    singles = _draw_want_to_marry(singles, marriage_rates, rng)
    if singles["wants_to_marry"].sum() == 0:
        return ds

    pairs = _match_pairs(singles, rng)
    if not pairs:
        return ds

    ds = _merge_pairs(ds, pairs)
    return ds


def _identify_single_adults(ds: UKSingleYearDataset) -> pd.DataFrame:
    """Return a frame of single adults keyed by ``person_id`` with their
    ``person_benunit_id``, ``person_household_id``, age, gender and region.

    A benunit is considered single if it has exactly one adult. Adults
    in multi-adult benunits (i.e. existing couples) are excluded from
    the marriage pool.
    """
    person = ds.person
    benunit_adult_counts = (
        person.assign(_is_adult=(person[AGE_COLUMN].astype(int) >= 18).astype(int))
        .groupby("person_benunit_id")["_is_adult"]
        .sum()
        .rename("benunit_adults")
    )
    single_benunits = benunit_adult_counts[benunit_adult_counts == 1].index

    adults = person[
        (person[AGE_COLUMN].astype(int) >= 18)
        & person["person_benunit_id"].isin(single_benunits)
    ].copy()

    # Pull household region so we can prefer in-region matches. Missing
    # region falls back to a placeholder so the groupby still works.
    if "region" in ds.household.columns:
        hh_region = ds.household.set_index("household_id")["region"]
        adults["_region"] = (
            adults["person_household_id"].map(hh_region).fillna("_UNKNOWN").values
        )
    else:
        adults["_region"] = "_UNKNOWN"
    return adults


def _draw_want_to_marry(
    singles: pd.DataFrame,
    rates: Mapping[str, Mapping[int, float]],
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Draw Bernoulli(rate[age, sex]) for each single; mark winners."""
    probs = np.array(
        [
            float(rates.get(str(sex), {}).get(int(age), 0.0))
            for age, sex in zip(singles[AGE_COLUMN].astype(int), singles[SEX_COLUMN])
        ],
        dtype=float,
    )
    draws = rng.random(size=len(singles))
    singles = singles.copy()
    singles["wants_to_marry"] = draws < probs
    return singles


def _match_pairs(
    singles: pd.DataFrame, rng: np.random.Generator
) -> list[tuple[int, int]]:
    """Return (male_person_id, female_person_id) pairs for matched singles.

    Greedy match within (region, same-sex draw order): for each male who
    wants to marry, pick the closest-age unmatched female in the same
    region. Remaining unmatched draws are discarded — one-sided excess
    of male or female marriage intent is not carried forward.
    """
    wants = singles[singles["wants_to_marry"]]
    if wants.empty:
        return []

    pairs: list[tuple[int, int]] = []
    for region, group in wants.groupby("_region"):
        males = group[group[SEX_COLUMN] == MALE_VALUE].copy()
        females = group[group[SEX_COLUMN] == FEMALE_VALUE].copy()
        if males.empty or females.empty:
            continue

        # Shuffle males so the greedy matcher doesn't deterministically
        # prefer younger men; matches still go to nearest age but the
        # iteration order is randomised.
        males = males.sample(frac=1.0, random_state=int(rng.integers(2**31 - 1)))

        female_ages = females[AGE_COLUMN].astype(int).to_numpy()
        female_ids = females["person_id"].to_numpy()
        female_taken = np.zeros(len(females), dtype=bool)

        for _, male_row in males.iterrows():
            male_age = int(male_row[AGE_COLUMN])
            # Closest available female.
            candidate_scores = np.where(
                female_taken, np.inf, np.abs(female_ages - male_age).astype(float)
            )
            best = int(np.argmin(candidate_scores))
            if candidate_scores[best] == np.inf:
                break
            pairs.append((int(male_row["person_id"]), int(female_ids[best])))
            female_taken[best] = True

    return pairs


def _merge_pairs(
    ds: UKSingleYearDataset, pairs: list[tuple[int, int]]
) -> UKSingleYearDataset:
    """For each (male_id, female_id) pair, merge their benunits.

    The male's benunit absorbs the female's benunit: every person whose
    ``person_benunit_id`` was the female's benunit gets reassigned to
    the male's benunit. Similarly the male's household absorbs the
    female's household — her previous household members (if any) move
    in with the male. If the female's benunit or household becomes
    empty as a result, its row is dropped and its weight is transferred
    to the surviving unit.

    This is a simplification: in practice a newlywed couple typically
    forms a new household rather than one party moving into the other.
    Expressing it this way keeps `benunit_id` / `household_id` stable
    for panel continuity of the male partner and avoids minting
    brand-new IDs each marriage; the downstream calibration step will
    smooth weight artefacts.
    """
    person = ds.person
    benunit = ds.benunit
    household = ds.household

    person_by_id = person.set_index("person_id")
    bu_weight_col = _find_weight_col(benunit)
    hh_weight_col = _find_weight_col(household)

    # Build lookup of affected benunit/household transitions.
    bu_reassign: dict[int, int] = {}
    hh_reassign: dict[int, int] = {}

    for male_id, female_id in pairs:
        m = person_by_id.loc[male_id]
        f = person_by_id.loc[female_id]
        m_bu, f_bu = int(m["person_benunit_id"]), int(f["person_benunit_id"])
        m_hh, f_hh = int(m["person_household_id"]), int(f["person_household_id"])

        # Chase transitive reassignments so that if a benunit was already
        # redirected (e.g. across two separate pairs touching the same
        # unit), we end up on the final destination.
        dest_bu = _resolve(bu_reassign, m_bu)
        src_bu = _resolve(bu_reassign, f_bu)
        if dest_bu != src_bu:
            bu_reassign[src_bu] = dest_bu

        dest_hh = _resolve(hh_reassign, m_hh)
        src_hh = _resolve(hh_reassign, f_hh)
        if dest_hh != src_hh:
            hh_reassign[src_hh] = dest_hh

    # Flatten transitive chains.
    bu_final = {k: _resolve(bu_reassign, k) for k in bu_reassign}
    hh_final = {k: _resolve(hh_reassign, k) for k in hh_reassign}

    if bu_final:
        person["person_benunit_id"] = person["person_benunit_id"].replace(bu_final)
    if hh_final:
        person["person_household_id"] = person["person_household_id"].replace(hh_final)

    # Fold weights from the absorbed rows into their new destination, then
    # drop the now-empty rows.
    if bu_final and bu_weight_col is not None:
        benunit = _fold_weights(benunit, "benunit_id", bu_final, bu_weight_col)
    if hh_final and hh_weight_col is not None:
        household = _fold_weights(household, "household_id", hh_final, hh_weight_col)

    ds.person = person.reset_index(drop=True)
    ds.benunit = benunit.reset_index(drop=True)
    ds.household = household.reset_index(drop=True)
    return ds


def _resolve(reassign: dict[int, int], key: int) -> int:
    while key in reassign:
        key = reassign[key]
    return key


def _find_weight_col(df: pd.DataFrame) -> str | None:
    for candidate in ("benunit_weight", "household_weight"):
        if candidate in df.columns:
            return candidate
    return None


# -- Age-indexed annual divorce (split) rates, per married person. -----------
# Sourced from ONS divorce statistics for England and Wales, smoothed
# approximations from the 1990-2020 period. They apply to the average
# age of the two adults in each benunit.
#
# Reference:
# https://www.ons.gov.uk/peoplepopulationandcommunity/birthsdeathsandmarriages/divorce/datasets/divorcesinenglandandwales
DEFAULT_SEPARATION_RATES: dict[int, float] = {
    **{age: 0.000 for age in range(0, 20)},
    **{age: 0.012 for age in range(20, 25)},
    **{age: 0.020 for age in range(25, 35)},
    **{age: 0.017 for age in range(35, 45)},
    **{age: 0.010 for age in range(45, 55)},
    **{age: 0.005 for age in range(55, 65)},
    **{age: 0.002 for age in range(65, 121)},
}


# ── Separation / divorce ────────────────────────────────────────────────────


def apply_separations(
    dataset: UKSingleYearDataset,
    separation_rates: Mapping[int, float] | None = None,
    rng: np.random.Generator | None = None,
    *,
    children_stay_with_female: bool = True,
) -> UKSingleYearDataset:
    """Split married benunits with probability ``separation_rates[age]``.

    For each benunit that contains exactly two adults, draw Bernoulli at
    the rate indexed by the mean of the two adults' ages. On a success,
    the benunit is split: one adult stays on the existing benunit ID,
    the other moves to a new benunit ID and a new household. Children
    remain by default with the female adult (the conventional HBAI
    treatment); set ``children_stay_with_female=False`` to keep them
    with the male.

    Args:
        dataset: panel-ready dataset. Not mutated.
        separation_rates: ``{age: probability}`` indexed by mean age of
            the adult couple. ``None`` uses :data:`DEFAULT_SEPARATION_RATES`;
            ``{}`` disables separations.
        rng: generator. ``None`` constructs a default one.
        children_stay_with_female: controls child attribution. Default
            True reflects HBAI's "resident parent" convention.

    Out of scope: partial-splits (e.g. separation without forming a new
    household, couples living apart together), and any asset division —
    income and consumption columns pass through unchanged on each side.
    """
    if rng is None:
        rng = np.random.default_rng()
    if separation_rates is None:
        separation_rates = DEFAULT_SEPARATION_RATES
    if not separation_rates:
        return dataset

    ds = dataset.copy()
    ds.person = ds.person.copy(deep=True)
    ds.benunit = ds.benunit.copy(deep=True)
    ds.household = ds.household.copy(deep=True)

    person = ds.person
    adult_mask = person[AGE_COLUMN].astype(int) >= 18
    adults = person[adult_mask]

    adult_counts = adults.groupby("person_benunit_id").size()
    couple_benunits = adult_counts[adult_counts == 2].index.tolist()
    if not couple_benunits:
        return ds

    mean_ages = (
        adults[adults["person_benunit_id"].isin(couple_benunits)]
        .groupby("person_benunit_id")[AGE_COLUMN]
        .mean()
        .astype(int)
    )
    probs = mean_ages.map(lambda a: float(separation_rates.get(int(a), 0.0)))
    draws = rng.random(size=len(probs))
    to_split = mean_ages.index[draws < probs.to_numpy()].tolist()
    if not to_split:
        return ds

    ds = _execute_separations(
        ds,
        couple_benunits_to_split=to_split,
        children_stay_with_female=children_stay_with_female,
        rng=rng,
    )
    return ds


def _execute_separations(
    ds: UKSingleYearDataset,
    *,
    couple_benunits_to_split: list[int],
    children_stay_with_female: bool,
    rng: np.random.Generator,
) -> UKSingleYearDataset:
    """Apply the actual row moves for the drawn separations."""
    person = ds.person
    benunit = ds.benunit
    household = ds.household

    next_bu_id = int(person["person_benunit_id"].max()) + 1
    next_hh_id = int(person["person_household_id"].max()) + 1

    bu_weight_col = _find_weight_col(benunit)
    hh_weight_col = _find_weight_col(household)

    # Stash extra rows we'll append.
    new_bu_rows: list[dict] = []
    new_hh_rows: list[dict] = []

    for bu_id in couple_benunits_to_split:
        bu_rows = person[person["person_benunit_id"] == bu_id]
        adults_in_bu = bu_rows[bu_rows[AGE_COLUMN].astype(int) >= 18]
        if len(adults_in_bu) != 2:
            continue

        sexes = adults_in_bu[SEX_COLUMN].to_numpy()
        ids = adults_in_bu["person_id"].to_numpy()

        # Decide which adult moves out. Whichever does NOT keep the children
        # leaves. If both adults are same sex or the rule disagrees, default
        # to the younger adult moving out (a common simplification).
        if children_stay_with_female and FEMALE_VALUE in sexes and MALE_VALUE in sexes:
            mover_id = int(ids[np.where(sexes != FEMALE_VALUE)[0][0]])
        elif (
            (not children_stay_with_female)
            and FEMALE_VALUE in sexes
            and MALE_VALUE in sexes
        ):
            mover_id = int(ids[np.where(sexes == FEMALE_VALUE)[0][0]])
        else:
            younger = adults_in_bu.sort_values(AGE_COLUMN).iloc[0]
            mover_id = int(younger["person_id"])

        original_hh_id = int(adults_in_bu.iloc[0]["person_household_id"])

        # Assign mover to a fresh benunit + household.
        person.loc[person["person_id"] == mover_id, "person_benunit_id"] = next_bu_id
        person.loc[person["person_id"] == mover_id, "person_household_id"] = next_hh_id

        if bu_weight_col is not None:
            original_bu_weight = float(
                benunit.loc[benunit["benunit_id"] == bu_id, bu_weight_col].iloc[0]
            )
            new_bu_rows.append(
                {"benunit_id": next_bu_id, bu_weight_col: original_bu_weight}
            )

        if hh_weight_col is not None:
            original_hh_weight = float(
                household.loc[
                    household["household_id"] == original_hh_id, hh_weight_col
                ].iloc[0]
            )
            new_hh_row = {"household_id": next_hh_id, hh_weight_col: original_hh_weight}
            # Preserve region if present.
            if "region" in household.columns:
                new_hh_row["region"] = str(
                    household.loc[
                        household["household_id"] == original_hh_id, "region"
                    ].iloc[0]
                )
            new_hh_rows.append(new_hh_row)

        next_bu_id += 1
        next_hh_id += 1

    if new_bu_rows:
        benunit = pd.concat(
            [benunit, pd.DataFrame(new_bu_rows, columns=benunit.columns)],
            ignore_index=True,
        )
    if new_hh_rows:
        household = pd.concat(
            [household, pd.DataFrame(new_hh_rows, columns=household.columns)],
            ignore_index=True,
        )

    ds.person = person.reset_index(drop=True)
    ds.benunit = benunit.reset_index(drop=True)
    ds.household = household.reset_index(drop=True)
    return ds


# -- Age-indexed rates of adults leaving the parental home. ------------------
# Sourced from ONS LFS "Young adults living with parents" series. Rates
# peak around age 18-22 as people leave for work or university and then
# decline sharply.
#
# Reference:
# https://www.ons.gov.uk/peoplepopulationandcommunity/birthsdeathsandmarriages/families/datasets/youngadultslivingwiththeirparents
DEFAULT_LEAVING_HOME_RATES: dict[int, float] = {
    **{age: 0.0 for age in range(0, 16)},
    16: 0.05,
    17: 0.08,
    **{age: 0.12 for age in range(18, 22)},
    **{age: 0.10 for age in range(22, 25)},
    **{age: 0.08 for age in range(25, 30)},
    **{age: 0.05 for age in range(30, 35)},
    **{age: 0.02 for age in range(35, 40)},
    **{age: 0.005 for age in range(40, 121)},
}


# ── Children leaving home ───────────────────────────────────────────────────


def apply_children_leaving_home(
    dataset: UKSingleYearDataset,
    leaving_home_rates: Mapping[int, float] | None = None,
    rng: np.random.Generator | None = None,
    *,
    min_age: int = 16,
) -> UKSingleYearDataset:
    """Move adult children out of their parents' benunit and household.

    For each person aged ``min_age`` or older who is currently attached
    to a benunit that contains another adult (their parent's couple, or
    a lone-parent benunit), draw Bernoulli at the age-indexed rate. On
    a success, the person leaves: a fresh benunit and household are
    minted, weights on both are seeded from the original household.

    Args:
        dataset: panel-ready dataset. Not mutated.
        leaving_home_rates: ``{age: probability}``. ``None`` uses
            :data:`DEFAULT_LEAVING_HOME_RATES`; ``{}`` disables.
        rng: generator. ``None`` constructs a default one.
        min_age: lower bound for eligibility. Default 16 matches the
            LFS "young adults living with parents" series.

    The mover carries their own income columns; the original household
    retains the same weight (households are not split in the weight
    sense — both sides represent independent units and are weighted
    as such).
    """
    if rng is None:
        rng = np.random.default_rng()
    if leaving_home_rates is None:
        leaving_home_rates = DEFAULT_LEAVING_HOME_RATES
    if not leaving_home_rates:
        return dataset

    ds = dataset.copy()
    ds.person = ds.person.copy(deep=True)
    ds.benunit = ds.benunit.copy(deep=True)
    ds.household = ds.household.copy(deep=True)

    person = ds.person
    eligible_ids = _identify_adult_dependents(person, min_age=min_age)
    if eligible_ids.empty:
        return ds

    probs = eligible_ids["age"].map(
        lambda a: float(leaving_home_rates.get(int(a), 0.0))
    )
    draws = rng.random(size=len(eligible_ids))
    leaving = eligible_ids[draws < probs.to_numpy()]
    if leaving.empty:
        return ds

    ds = _execute_leaving_home(ds, leaving["person_id"].tolist())
    return ds


def _identify_adult_dependents(person: pd.DataFrame, *, min_age: int) -> pd.DataFrame:
    """Return adults who appear to be living in a parent's home.

    The FRS / HBAI treatment has two shapes:

    1. **Young dependent adult on parents' benunit** — a benunit with
       three or more adults. The two oldest are assumed to be the
       parent couple; any additional adult is an adult dependent
       candidate.
    2. **Adult child with own benunit inside parental household** — a
       single-adult benunit whose household also contains at least one
       other benunit with adults. The single adult is the candidate;
       we only flag them when the household's *other* benunit has
       adults (so flatmate-shares and lone adults are excluded).

    Couples (two-adult benunits) are never flagged — those are the
    resident parental pair.
    """
    adult_mask = person[AGE_COLUMN].astype(int) >= min_age
    adults = person[adult_mask].copy()
    if adults.empty:
        return pd.DataFrame(columns=["person_id", "age"])

    # Pre-compute per-benunit adult counts.
    bu_adult_counts = adults.groupby("person_benunit_id").size()

    # ---- Case 1: benunit has 3+ adults; dependents are all but the two
    # oldest.
    case1_ids: list[int] = []
    multi_adult_bus = bu_adult_counts[bu_adult_counts >= 3].index
    for bu in multi_adult_bus:
        bu_adults = adults[adults["person_benunit_id"] == bu].sort_values(
            by=AGE_COLUMN, ascending=False
        )
        # Skip the two oldest (the parental couple).
        case1_ids.extend(bu_adults["person_id"].iloc[2:].tolist())

    # ---- Case 2: single-adult benunit sharing a household with a
    # different benunit that also has adults.
    benunit_is_single_adult = bu_adult_counts == 1
    single_adult_bu_ids = set(benunit_is_single_adult[benunit_is_single_adult].index)

    case2_ids: list[int] = []
    for hh_id, group in adults.groupby("person_household_id"):
        benunits_in_hh = set(group["person_benunit_id"].unique())
        if len(benunits_in_hh) < 2:
            continue
        for bu_id in benunits_in_hh:
            if bu_id not in single_adult_bu_ids:
                continue
            # Other benunits in this household must carry at least one
            # adult, not just children in other benunits (flatmates).
            others_with_adults = [
                other
                for other in benunits_in_hh
                if other != bu_id and bu_adult_counts.get(other, 0) >= 1
            ]
            if not others_with_adults:
                continue
            case2_ids.extend(
                group[group["person_benunit_id"] == bu_id]["person_id"].tolist()
            )

    eligible_ids = set(case1_ids) | set(case2_ids)
    if not eligible_ids:
        return pd.DataFrame(columns=["person_id", "age"])

    eligible = adults[adults["person_id"].isin(eligible_ids)].copy()
    eligible = eligible.rename(columns={AGE_COLUMN: "age"})
    return eligible[["person_id", "age"]].reset_index(drop=True)


def _execute_leaving_home(
    ds: UKSingleYearDataset, leaving_person_ids: list[int]
) -> UKSingleYearDataset:
    """Move each leaver to a fresh benunit + household."""
    person = ds.person
    benunit = ds.benunit
    household = ds.household

    next_bu_id = int(person["person_benunit_id"].max()) + 1
    next_hh_id = int(person["person_household_id"].max()) + 1

    bu_weight_col = _find_weight_col(benunit)
    hh_weight_col = _find_weight_col(household)

    new_bu_rows: list[dict] = []
    new_hh_rows: list[dict] = []

    for person_id in leaving_person_ids:
        row = person[person["person_id"] == person_id].iloc[0]
        original_hh_id = int(row["person_household_id"])
        original_bu_id = int(row["person_benunit_id"])

        person.loc[person["person_id"] == person_id, "person_benunit_id"] = next_bu_id
        person.loc[person["person_id"] == person_id, "person_household_id"] = next_hh_id

        if bu_weight_col is not None:
            bu_weight_source = benunit.loc[
                benunit["benunit_id"] == original_bu_id, bu_weight_col
            ]
            weight = float(bu_weight_source.iloc[0]) if len(bu_weight_source) else 1.0
            new_bu_rows.append({"benunit_id": next_bu_id, bu_weight_col: weight})

        if hh_weight_col is not None:
            hh_weight_source = household.loc[
                household["household_id"] == original_hh_id, hh_weight_col
            ]
            weight = float(hh_weight_source.iloc[0]) if len(hh_weight_source) else 1.0
            new_hh_row = {"household_id": next_hh_id, hh_weight_col: weight}
            if "region" in household.columns:
                region_source = household.loc[
                    household["household_id"] == original_hh_id, "region"
                ]
                new_hh_row["region"] = (
                    str(region_source.iloc[0]) if len(region_source) else ""
                )
            new_hh_rows.append(new_hh_row)

        next_bu_id += 1
        next_hh_id += 1

    if new_bu_rows:
        benunit = pd.concat(
            [benunit, pd.DataFrame(new_bu_rows, columns=benunit.columns)],
            ignore_index=True,
        )
    if new_hh_rows:
        household = pd.concat(
            [household, pd.DataFrame(new_hh_rows, columns=household.columns)],
            ignore_index=True,
        )

    ds.person = person.reset_index(drop=True)
    ds.benunit = benunit.reset_index(drop=True)
    ds.household = household.reset_index(drop=True)
    return ds


def _fold_weights(
    df: pd.DataFrame, id_col: str, reassign: dict[int, int], weight_col: str
) -> pd.DataFrame:
    """Sum absorbed rows' weights into their destination row, then drop."""
    df = df.copy()
    df[id_col] = df[id_col].astype(int)
    mapping = df.set_index(id_col)[weight_col]
    for src, dest in reassign.items():
        if src in mapping.index and dest in mapping.index:
            df.loc[df[id_col] == dest, weight_col] = float(mapping.loc[dest]) + float(
                mapping.loc[src]
            )
    df = df[~df[id_col].isin(reassign.keys())]
    return df


# -- Net migration rates (immigration minus emigration) by age, per capita. --
# Sourced from ONS Long-Term International Migration estimates, smoothed
# averages 2015-2023. Positive = net in, negative = net out. Migration is
# heavily concentrated at ages 18-35 (students and working-age migrants).
#
# Reference:
# https://www.ons.gov.uk/peoplepopulationandcommunity/populationandmigration/internationalmigration/datasets/longterminternationalmigrationprovisional
DEFAULT_NET_MIGRATION_RATES: dict[int, float] = {
    **{age: 0.001 for age in range(0, 15)},
    **{age: 0.003 for age in range(15, 18)},
    **{age: 0.012 for age in range(18, 25)},
    **{age: 0.010 for age in range(25, 35)},
    **{age: 0.003 for age in range(35, 50)},
    **{age: 0.001 for age in range(50, 65)},
    **{age: 0.000 for age in range(65, 121)},
}


# ── Migration ───────────────────────────────────────────────────────────────


def apply_migration(
    dataset: UKSingleYearDataset,
    net_migration_rates: Mapping[int, float] | None = None,
    rng: np.random.Generator | None = None,
) -> UKSingleYearDataset:
    """Add immigrants (positive net) or remove emigrants (negative net).

    Net migration is modelled as a per-capita age-indexed rate. For each
    age cohort we compute the expected net delta and either:

    - **immigrate** by cloning rows from the existing same-age cohort
      as donor "nearest-neighbour" immigrants, re-assigning fresh
      ``person_id`` / benunit / household IDs and a small starting
      weight;
    - **emigrate** by removing randomly-drawn rows of that age.

    The donor-bootstrap approach mirrors the US panel pipeline and keeps
    joint distributions (income, employment, region) realistic without
    needing an external immigrant synthetic population.

    Args:
        dataset: panel-ready dataset. Not mutated.
        net_migration_rates: ``{age: per_capita_delta}`` where positive
            means net inflow. ``None`` uses
            :data:`DEFAULT_NET_MIGRATION_RATES`; ``{}`` disables.
        rng: generator. ``None`` constructs a default one.

    Out of scope: asylum / refugee distinct flow (captured by the ONS
    LTIM aggregate), return-migration of citizens (absorbed into the
    same aggregate), and visa-type modelling.
    """
    if rng is None:
        rng = np.random.default_rng()
    if net_migration_rates is None:
        net_migration_rates = DEFAULT_NET_MIGRATION_RATES
    if not net_migration_rates:
        return dataset

    ds = dataset.copy()
    ds.person = ds.person.copy(deep=True)
    ds.benunit = ds.benunit.copy(deep=True)
    ds.household = ds.household.copy(deep=True)

    ages = ds.person[AGE_COLUMN].astype(int).to_numpy()
    cohort_sizes = pd.Series(ages).value_counts().to_dict()

    emigrate_ids: list[int] = []
    immigrate_donor_ids: list[int] = []

    for age, n_in_cohort in cohort_sizes.items():
        rate = float(net_migration_rates.get(int(age), 0.0))
        if rate == 0.0 or n_in_cohort == 0:
            continue
        expected = rate * n_in_cohort
        # Draw actual number as Poisson so small rates average correctly
        # over many panel runs.
        delta = int(rng.poisson(lam=abs(expected)))
        if delta == 0:
            continue
        if rate > 0:
            # Immigration: pick donor rows at this age.
            donors = ds.person[ds.person[AGE_COLUMN].astype(int) == int(age)]
            chosen = donors.sample(
                n=min(delta, len(donors)),
                random_state=int(rng.integers(2**31 - 1)),
                replace=True,
            )
            immigrate_donor_ids.extend(chosen["person_id"].tolist())
        else:
            # Emigration: remove random rows.
            leavers = ds.person[ds.person[AGE_COLUMN].astype(int) == int(age)]
            chosen = leavers.sample(
                n=min(delta, len(leavers)),
                random_state=int(rng.integers(2**31 - 1)),
                replace=False,
            )
            emigrate_ids.extend(chosen["person_id"].tolist())

    if emigrate_ids:
        ds = _drop_emigrants(ds, emigrate_ids)
    if immigrate_donor_ids:
        ds = _append_immigrants(ds, immigrate_donor_ids)

    return ds


def _drop_emigrants(
    ds: UKSingleYearDataset, emigrate_ids: list[int]
) -> UKSingleYearDataset:
    """Remove emigrant person rows. If the departure empties a benunit or
    household, drop the empty row so downstream joins stay valid.
    """
    person = ds.person[~ds.person["person_id"].isin(emigrate_ids)].reset_index(
        drop=True
    )
    remaining_bus = set(person["person_benunit_id"].unique())
    remaining_hhs = set(person["person_household_id"].unique())
    benunit = ds.benunit[ds.benunit["benunit_id"].isin(remaining_bus)].reset_index(
        drop=True
    )
    household = ds.household[
        ds.household["household_id"].isin(remaining_hhs)
    ].reset_index(drop=True)
    ds.person = person
    ds.benunit = benunit
    ds.household = household
    return ds


def _append_immigrants(
    ds: UKSingleYearDataset, donor_person_ids: list[int]
) -> UKSingleYearDataset:
    """Append immigrant rows cloned from donor rows with fresh IDs.

    Each donor becomes exactly one new person (we're modelling single
    migrants here, not family-unit migration). The new person lives in
    a fresh single-adult benunit and a fresh single-person household,
    inheriting the donor's age, sex and income attributes.
    """
    person = ds.person
    benunit = ds.benunit
    household = ds.household

    next_person_id = int(person["person_id"].max()) + 1
    next_bu_id = int(person["person_benunit_id"].max()) + 1
    next_hh_id = int(person["person_household_id"].max()) + 1

    bu_weight_col = _find_weight_col(benunit)
    hh_weight_col = _find_weight_col(household)

    new_people: list[dict] = []
    new_benunits: list[dict] = []
    new_households: list[dict] = []

    donor_rows = person.set_index("person_id")

    for donor_id in donor_person_ids:
        donor = donor_rows.loc[donor_id]
        row = donor.to_dict()
        row["person_id"] = next_person_id
        row["person_benunit_id"] = next_bu_id
        row["person_household_id"] = next_hh_id
        new_people.append(row)

        if bu_weight_col is not None:
            new_benunits.append({"benunit_id": next_bu_id, bu_weight_col: 1.0})

        if hh_weight_col is not None:
            hh_row: dict = {"household_id": next_hh_id, hh_weight_col: 1.0}
            if "region" in household.columns:
                # Seed region from donor's original household so immigrant
                # geography isn't uniformly placed in a single region.
                donor_hh_id = int(donor["person_household_id"])
                src = household.loc[household["household_id"] == donor_hh_id, "region"]
                hh_row["region"] = str(src.iloc[0]) if len(src) else ""
            new_households.append(hh_row)

        next_person_id += 1
        next_bu_id += 1
        next_hh_id += 1

    if new_people:
        person = pd.concat(
            [person, pd.DataFrame(new_people, columns=person.columns)],
            ignore_index=True,
        )
    if new_benunits:
        benunit = pd.concat(
            [benunit, pd.DataFrame(new_benunits, columns=benunit.columns)],
            ignore_index=True,
        )
    if new_households:
        household = pd.concat(
            [household, pd.DataFrame(new_households, columns=household.columns)],
            ignore_index=True,
        )

    ds.person = person
    ds.benunit = benunit
    ds.household = household
    return ds


# -- Rule-based employment and income transitions ----------------------------
# A placeholder for proper panel-estimated transitions. Without UKHLS we
# cannot observe true within-person year-on-year moves, so this path
# implements three simple rules that dominate the first-order dynamics:
#
# 1. Retirement at state-pension age: zero out labour income for workers
#    turning SPA.
# 2. Wage drift: scale existing labour income by a CPI-like factor.
# 3. Random job-loss and job-gain rates for working-age non-retired
#    adults.
#
# When UKHLS becomes available these rates can be replaced by observed
# transition probabilities. The signatures are stable so the caller does
# not need to change.

DEFAULT_STATE_PENSION_AGE = 66

DEFAULT_JOB_LOSS_RATE = 0.03
DEFAULT_JOB_GAIN_RATE = 0.05
DEFAULT_WAGE_DRIFT = 0.04  # nominal growth per year (CPI + small real)


def apply_employment_transitions(
    dataset: UKSingleYearDataset,
    *,
    state_pension_age: int = DEFAULT_STATE_PENSION_AGE,
    job_loss_rate: float = DEFAULT_JOB_LOSS_RATE,
    job_gain_rate: float = DEFAULT_JOB_GAIN_RATE,
    wage_drift: float = DEFAULT_WAGE_DRIFT,
    rng: np.random.Generator | None = None,
) -> UKSingleYearDataset:
    """Apply one year of labour-market transitions (rule-based).

    Three independent effects run in order:

    1. **Retirement**: people who reach SPA this year have their
       ``employment_income`` and ``self_employment_income`` zeroed, and
       their ``employment_status`` set to ``RETIRED`` if that column is
       present. No reverse transition (no "un-retirement"): the flow is
       one-way.
    2. **Wage drift**: labour income of remaining workers is multiplied
       by ``(1 + wage_drift)`` to track a nominal CPI-plus-real path.
    3. **Job loss / gain**: for working-age (< SPA) adults, each
       currently-employed row can lose their job with probability
       ``job_loss_rate`` (income zeroed, status → UNEMPLOYED if column
       present); each currently-non-employed row can gain a job with
       probability ``job_gain_rate``. New jobs draw a starting income
       from the distribution of the dataset's currently-employed
       earners at the same age (nearest-age donor).

    This is an explicit placeholder — proper year-on-year transitions
    belong on UKHLS-estimated rates. The rule-based defaults are small
    enough to be a low-noise starting point and surfaced as caller args
    so they can be tuned to observed flows in the meantime.
    """
    if rng is None:
        rng = np.random.default_rng()

    ds = dataset.copy()
    ds.person = ds.person.copy(deep=True)

    person = ds.person
    ages = person[AGE_COLUMN].astype(int).to_numpy()

    emp_col = "employment_income"
    self_col = (
        "self_employment_income" if "self_employment_income" in person.columns else None
    )
    status_col = "employment_status" if "employment_status" in person.columns else None

    has_labour_income = (
        person[emp_col] > 0
        if emp_col in person.columns
        else pd.Series([False] * len(person))
    )
    if self_col is not None:
        has_labour_income = has_labour_income | (person[self_col] > 0)

    # --- 1. Retirement at SPA -------------------------------------------------
    reaching_spa = ages >= state_pension_age
    if emp_col in person.columns:
        person.loc[reaching_spa, emp_col] = 0.0
    if self_col is not None:
        person.loc[reaching_spa, self_col] = 0.0
    if status_col is not None:
        person.loc[reaching_spa, status_col] = "RETIRED"

    # --- 2. Wage drift for remaining workers ---------------------------------
    active_workers = (ages < state_pension_age) & has_labour_income.to_numpy()
    if emp_col in person.columns:
        person.loc[active_workers, emp_col] = person.loc[active_workers, emp_col] * (
            1.0 + wage_drift
        )
    if self_col is not None:
        person.loc[active_workers, self_col] = person.loc[active_workers, self_col] * (
            1.0 + wage_drift
        )

    # --- 3. Job loss / job gain ----------------------------------------------
    # Working-age, non-retired.
    working_age = (ages >= 18) & (ages < state_pension_age)
    employed_mask = working_age & has_labour_income.to_numpy()
    unemployed_mask = working_age & ~has_labour_income.to_numpy()

    # Job loss draws.
    if job_loss_rate > 0 and employed_mask.any():
        draws = rng.random(size=len(person))
        loses = employed_mask & (draws < job_loss_rate)
        if emp_col in person.columns:
            person.loc[loses, emp_col] = 0.0
        if self_col is not None:
            person.loc[loses, self_col] = 0.0
        if status_col is not None:
            person.loc[loses, status_col] = "UNEMPLOYED"

    # Job gain draws — assign a new income from the same-age employed donor pool.
    if job_gain_rate > 0 and unemployed_mask.any() and emp_col in person.columns:
        draws = rng.random(size=len(person))
        gains = unemployed_mask & (draws < job_gain_rate)
        if gains.any():
            currently_employed = person[
                (person[emp_col] > 0) & (ages < state_pension_age)
            ]
            if not currently_employed.empty:
                age_values = currently_employed[AGE_COLUMN].astype(int).to_numpy()
                income_values = currently_employed[emp_col].to_numpy()
                gainer_ages = person.loc[gains, AGE_COLUMN].astype(int).to_numpy()
                new_incomes = []
                for ga in gainer_ages:
                    # Pick nearest age donor (ties broken randomly).
                    diffs = np.abs(age_values - ga)
                    best = np.where(diffs == diffs.min())[0]
                    chosen = int(rng.choice(best))
                    new_incomes.append(float(income_values[chosen]))
                person.loc[gains, emp_col] = new_incomes
                if status_col is not None:
                    person.loc[gains, status_col] = "FT_EMPLOYED"

    ds.person = person.reset_index(drop=True)
    return ds
