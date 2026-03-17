"""Adversarial offspring generation.

For each high-influence household record, generates synthetic
offspring via the generative model, runs them through PolicyEngine
to compute tax-benefit outputs, and assembles an expanded dataset
ready for recalibration.
"""

import logging

import numpy as np
import pandas as pd

from policyengine_uk_data.diagnostics.influence import (
    compute_influence_matrix,
    find_high_influence_records,
)
from policyengine_uk_data.diagnostics.generative_model import (
    extract_household_features,
    sample_offspring,
)

logger = logging.getLogger(__name__)


def _expand_household_to_dataset_records(
    synthetic_hh: pd.Series,
    source_dataset,
    source_hh_idx: int,
    new_hh_id_start: int,
    weight_per_offspring: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create person/benunit/household rows for one synthetic household.

    The synthetic household inherits the *structure* (number of
    persons, benefit units, relationships) from the source household,
    with attribute values replaced by the synthetic record where
    applicable.

    Args:
        synthetic_hh: Series of synthetic household-level features.
        source_dataset: the original UKSingleYearDataset.
        source_hh_idx: index of the source household in the dataset.
        new_hh_id_start: starting household_id for the new record.
        weight_per_offspring: weight to assign.

    Returns:
        (person_df, benunit_df, household_df) for the new household.
    """
    orig_hh = source_dataset.household
    orig_person = source_dataset.person
    orig_benunit = source_dataset.benunit

    source_hh_id = orig_hh.household_id.iloc[source_hh_idx]
    new_hh_id = new_hh_id_start

    # Copy the source household's structure
    hh_row = orig_hh.iloc[[source_hh_idx]].copy()
    hh_row["household_id"] = new_hh_id
    hh_row["household_weight"] = weight_per_offspring

    # Override household-level attributes from the synthetic record
    hh_attr_map = {
        "region": "region",
        "tenure_type": "tenure_type",
        "rent": "rent",
        "council_tax": "council_tax",
        "council_tax_band": "council_tax_band",
        "accommodation_type": "accommodation_type",
    }
    for synth_col, hh_col in hh_attr_map.items():
        if synth_col in synthetic_hh.index and hh_col in hh_row.columns:
            hh_row[hh_col] = synthetic_hh[synth_col]

    # Copy persons, remapping IDs
    person_mask = orig_person.person_household_id == source_hh_id
    new_persons = orig_person[person_mask].copy()
    new_persons["person_household_id"] = new_hh_id

    # Remap person IDs to avoid collisions
    person_id_offset = new_hh_id * 1000
    new_persons["person_id"] = np.arange(
        person_id_offset,
        person_id_offset + len(new_persons),
    )

    # Override head's income attributes from the synthetic record
    head_mask = new_persons.is_household_head.astype(bool)
    income_map = {
        "hh_employment_income": "employment_income",
        "hh_self_employment_income": "self_employment_income",
        "hh_private_pension_income": "private_pension_income",
        "hh_savings_interest_income": "savings_interest_income",
        "hh_dividend_income": "dividend_income",
        "hh_property_income": "property_income",
    }
    for synth_col, person_col in income_map.items():
        if synth_col in synthetic_hh.index and person_col in new_persons.columns:
            # Assign all household income to head (simplified)
            new_persons.loc[head_mask, person_col] = max(
                0, float(synthetic_hh[synth_col])
            )

    # Copy benefit units, remapping IDs
    old_bu_ids = new_persons.person_benunit_id.unique()
    bu_id_offset = new_hh_id * 100
    bu_id_map = {old: bu_id_offset + i for i, old in enumerate(old_bu_ids)}
    new_persons["person_benunit_id"] = new_persons["person_benunit_id"].map(bu_id_map)

    bu_mask = orig_benunit.benunit_id.isin(old_bu_ids)
    new_beunits = orig_benunit[bu_mask].copy()
    new_beunits["benunit_id"] = new_beunits["benunit_id"].map(bu_id_map)

    return new_persons, new_beunits, hh_row


def generate_offspring_for_record(
    dataset,
    record_idx: int,
    model,
    features: pd.DataFrame,
    n_offspring: int = 50,
    weight_target: float | None = None,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Generate synthetic offspring for a single household.

    Args:
        dataset: UKSingleYearDataset.
        record_idx: index of the household to split.
        model: trained generative model (TVAE).
        features: household features DataFrame (from
            extract_household_features).
        n_offspring: number of candidate offspring.
        weight_target: desired max weight per offspring.  If None,
            uses p90 of the current weight distribution.
        seed: random seed.

    Returns:
        (person_df, benunit_df, household_df) for all offspring
        combined.
    """
    weights = dataset.household.household_weight.values
    source_weight = weights[record_idx]

    if weight_target is None:
        weight_target = float(np.percentile(weights[weights > 0], 90))

    k = max(2, int(np.ceil(source_weight / weight_target)))
    n_candidates = max(n_offspring, k * 3)

    source_features = features.iloc[record_idx]
    synthetic = sample_offspring(
        model,
        source_features,
        n_samples=n_candidates,
        seed=seed,
    )

    weight_per = source_weight / n_candidates
    max_hh_id = dataset.household.household_id.max()

    all_persons = []
    all_beunits = []
    all_households = []

    for i in range(len(synthetic)):
        new_hh_id = int(max_hh_id + record_idx * 10_000 + i + 1)
        try:
            p, b, h = _expand_household_to_dataset_records(
                synthetic.iloc[i],
                dataset,
                record_idx,
                new_hh_id,
                weight_per,
            )
            all_persons.append(p)
            all_beunits.append(b)
            all_households.append(h)
        except Exception as e:
            logger.debug("Offspring %d failed: %s", i, e)

    if not all_persons:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    return (
        pd.concat(all_persons, ignore_index=True),
        pd.concat(all_beunits, ignore_index=True),
        pd.concat(all_households, ignore_index=True),
    )


def run_adversarial_loop(
    dataset,
    model,
    time_period: str = "2025",
    threshold: float = 0.05,
    max_rounds: int = 10,
    n_offspring: int = 50,
    weight_target: float | None = None,
    seed: int = 42,
) -> dict:
    """Run the full adversarial detect-spawn-recalibrate loop.

    Args:
        dataset: UKSingleYearDataset to expand.
        model: trained generative model (TVAE).
        time_period: calendar year as string.
        threshold: max allowable influence fraction.
        max_rounds: maximum number of adversarial rounds.
        n_offspring: offspring per flagged record.
        weight_target: desired max weight.
        seed: random seed.

    Returns:
        Dict with:
          - expanded_dataset: the expanded UKSingleYearDataset
          - rounds_completed: number of rounds run
          - influence_history: list of max-influence per round
          - records_expanded: total number of records added
    """
    from policyengine_uk import Microsimulation
    from policyengine_uk.data import UKSingleYearDataset
    from policyengine_uk_data.diagnostics.recalibrate import (
        recalibrate_with_regularisation,
    )

    working = dataset.copy()
    features = extract_household_features(working)
    influence_history = []
    total_added = 0

    for round_num in range(max_rounds):
        logger.info("Adversarial round %d/%d", round_num + 1, max_rounds)

        # Detect
        sim = Microsimulation(dataset=working)
        sim.default_calculation_period = time_period
        infl = compute_influence_matrix(sim, time_period)
        flagged = find_high_influence_records(infl, threshold)

        if flagged.empty:
            logger.info(
                "No records above threshold, stopping at round %d",
                round_num + 1,
            )
            break

        max_infl = flagged.max_influence.iloc[0]
        influence_history.append(float(max_infl))
        logger.info(
            "Round %d: %d flagged records, max influence %.3f",
            round_num + 1,
            len(flagged),
            max_infl,
        )

        # Spawn offspring for the worst offender
        worst_idx = int(flagged.record_idx.iloc[0])
        persons_new, beunits_new, hh_new = generate_offspring_for_record(
            working,
            worst_idx,
            model,
            features,
            n_offspring=n_offspring,
            weight_target=weight_target,
            seed=seed + round_num,
        )

        if hh_new.empty:
            logger.warning(
                "No offspring generated for record %d, skipping",
                worst_idx,
            )
            continue

        # Remove source record and add offspring
        orig_hh_id = working.household.household_id.iloc[worst_idx]

        new_person = pd.concat(
            [
                working.person[working.person.person_household_id != orig_hh_id],
                persons_new,
            ],
            ignore_index=True,
        )
        new_benunit = pd.concat(
            [
                working.benunit[
                    ~working.benunit.benunit_id.isin(
                        working.person[
                            working.person.person_household_id == orig_hh_id
                        ].person_benunit_id
                    )
                ],
                beunits_new,
            ],
            ignore_index=True,
        )
        new_household = pd.concat(
            [
                working.household[working.household.household_id != orig_hh_id],
                hh_new,
            ],
            ignore_index=True,
        )

        working = UKSingleYearDataset(
            person=new_person,
            benunit=new_benunit,
            household=new_household,
            fiscal_year=int(time_period),
        )
        features = extract_household_features(working)
        total_added += len(hh_new)

        # Recalibrate
        logger.info("Recalibrating expanded dataset...")
        working = recalibrate_with_regularisation(
            working,
            time_period=time_period,
        )

    return {
        "expanded_dataset": working,
        "rounds_completed": min(round_num + 1, max_rounds),
        "influence_history": influence_history,
        "records_expanded": total_added,
    }
