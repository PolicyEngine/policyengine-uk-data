"""Generative model for household attributes.

Trains a TVAE (Tabular Variational Autoencoder) on FRS input
attributes and provides conditional sampling for offspring
generation.  The model learns the joint distribution of household
demographics, income, housing, and geographic variables so that
synthetic records are plausible completions of partial attribute
sets.

Only *input* attributes are modelled.  Tax-benefit *outputs* (tax
liability, benefit entitlement, net income) are recomputed by
running offspring through PolicyEngine's calculator.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Input attributes to model.  These are the FRS variables that
# define a household before PolicyEngine calculates anything.
PERSON_INPUT_ATTRS = [
    "age",
    "gender",
    "employment_income",
    "self_employment_income",
    "private_pension_income",
    "savings_interest_income",
    "dividend_income",
    "property_income",
    "hours_worked",
    "employment_status",
    "is_disabled_for_benefits",
    "marital_status",
]

HOUSEHOLD_INPUT_ATTRS = [
    "region",
    "tenure_type",
    "rent",
    "council_tax",
    "council_tax_band",
    "accommodation_type",
    "household_weight",
]

BENUNIT_INPUT_ATTRS = [
    "would_claim_uc",
    "is_married",
]


def extract_household_features(
    dataset,
    use_design_weights: bool = True,
) -> pd.DataFrame:
    """Extract a flat feature table from a UKSingleYearDataset.

    Each row is one household.  Person-level variables are aggregated
    to household level (head's values for demographics, sums for
    incomes).

    Args:
        dataset: UKSingleYearDataset instance.
        use_design_weights: if True, use the original grossing
            weights (not calibrated) for training the generative
            model.

    Returns:
        DataFrame with one row per household.
    """
    person = dataset.person
    household = dataset.household
    benunit = dataset.benunit

    hh_ids = household.household_id.values
    features = pd.DataFrame({"household_id": hh_ids})

    # Household-level attributes
    for attr in HOUSEHOLD_INPUT_ATTRS:
        if attr in household.columns:
            features[attr] = household[attr].values

    # Person-level: head's demographics + income sums
    head_mask = person.is_household_head.astype(bool)
    heads = person[head_mask].set_index("person_household_id")

    for attr in ["age", "gender", "employment_status", "marital_status"]:
        if attr in heads.columns:
            features[f"head_{attr}"] = heads[attr].reindex(hh_ids).values

    # Income sums across all persons in household
    income_attrs = [
        "employment_income",
        "self_employment_income",
        "private_pension_income",
        "savings_interest_income",
        "dividend_income",
        "property_income",
    ]
    for attr in income_attrs:
        if attr in person.columns:
            summed = (
                person.groupby("person_household_id")[attr]
                .sum()
                .reindex(hh_ids)
                .fillna(0)
            )
            features[f"hh_{attr}"] = summed.values

    # Household size
    features["n_persons"] = (
        person.groupby("person_household_id")
        .size()
        .reindex(hh_ids)
        .fillna(1)
        .astype(int)
        .values
    )

    # Number of children (age < 18)
    if "age" in person.columns:
        features["n_children"] = (
            person[person.age < 18]
            .groupby("person_household_id")
            .size()
            .reindex(hh_ids)
            .fillna(0)
            .astype(int)
            .values
        )

    # Hours worked (head)
    if "hours_worked" in heads.columns:
        features["head_hours_worked"] = (
            heads["hours_worked"].reindex(hh_ids).fillna(0).values
        )

    # Disability flag (any person in household)
    if "is_disabled_for_benefits" in person.columns:
        features["has_disabled_member"] = (
            person.groupby("person_household_id")["is_disabled_for_benefits"]
            .max()
            .reindex(hh_ids)
            .fillna(0)
            .astype(int)
            .values
        )

    # Benunit: UC claim status (any benunit in household)
    if "would_claim_uc" in benunit.columns:
        person_bu = person[["person_household_id", "person_benunit_id"]]
        bu_hh = person_bu.drop_duplicates("person_benunit_id").set_index(
            "person_benunit_id"
        )["person_household_id"]
        uc_by_hh = benunit.set_index("benunit_id")["would_claim_uc"].reindex(
            bu_hh.index
        )
        uc_by_hh.index = bu_hh.values
        features["any_uc_claim"] = (
            uc_by_hh.groupby(level=0).max().reindex(hh_ids).fillna(0).astype(int).values
        )

    return features


def identify_column_types(
    df: pd.DataFrame,
) -> tuple[list[str], list[str]]:
    """Split columns into categorical and continuous.

    Returns:
        (categorical_columns, continuous_columns)
    """
    categorical = []
    continuous = []
    skip = {"household_id", "household_weight"}

    for col in df.columns:
        if col in skip:
            continue
        if df[col].dtype == object or df[col].nunique() < 20:
            categorical.append(col)
        else:
            continuous.append(col)

    return categorical, continuous


def train_generative_model(
    dataset,
    epochs: int = 300,
    seed: int = 42,
):
    """Train a TVAE on household features.

    Args:
        dataset: UKSingleYearDataset instance.
        epochs: training epochs for the TVAE.
        seed: random seed.

    Returns:
        Trained TVAE model (sdv SingleTableSynthesizer).
    """
    from sdv.single_table import TVAESynthesizer
    from sdv.metadata import Metadata

    features = extract_household_features(dataset)
    categorical_cols, continuous_cols = identify_column_types(features)

    # Drop household_id for training
    train_df = features.drop(columns=["household_id"])
    if "household_weight" in train_df.columns:
        sample_weights = train_df["household_weight"].values.copy()
        train_df = train_df.drop(columns=["household_weight"])
    else:
        sample_weights = None

    # Build metadata
    metadata = Metadata.detect_from_dataframe(data=train_df)

    model = TVAESynthesizer(
        metadata=metadata,
        epochs=epochs,
        verbose=True,
    )

    # Weight the training data by design weights if available
    if sample_weights is not None:
        # Resample proportional to weights for training
        rng = np.random.default_rng(seed)
        probs = sample_weights / sample_weights.sum()
        n_train = len(train_df)
        indices = rng.choice(len(train_df), size=n_train, replace=True, p=probs)
        train_df = train_df.iloc[indices].reset_index(drop=True)

    model.fit(train_df)
    logger.info("TVAE trained on %d records", len(train_df))

    return model


def sample_offspring(
    model,
    source_record: pd.Series,
    n_samples: int = 50,
    conditioning_fractions: list[float] | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic offspring conditioned on a source record.

    For each sample, a random subset of the source record's
    attributes are fixed, and the rest are sampled from the model.
    The conditioning fraction varies across samples to explore both
    close variants and broader alternatives.

    Args:
        model: trained TVAE model.
        source_record: Series of attribute values for the source
            household.
        n_samples: number of offspring to generate.
        conditioning_fractions: list of fractions of attributes to
            condition on.  Defaults to a spread from 0.2 to 0.8.
        seed: random seed.

    Returns:
        DataFrame of synthetic offspring (n_samples rows).
    """
    rng = np.random.default_rng(seed)

    if conditioning_fractions is None:
        conditioning_fractions = [0.2, 0.4, 0.5, 0.6, 0.8]

    all_cols = list(source_record.index)
    skip_cols = {"household_id", "household_weight"}
    usable_cols = [c for c in all_cols if c not in skip_cols]

    offspring = []
    samples_per_fraction = max(1, n_samples // len(conditioning_fractions))

    for frac in conditioning_fractions:
        n_cond = max(1, int(len(usable_cols) * frac))
        for _ in range(samples_per_fraction):
            cond_cols = rng.choice(usable_cols, size=n_cond, replace=False).tolist()
            conditions = {col: source_record[col] for col in cond_cols}
            try:
                from sdv.sampling import Condition

                condition = Condition(
                    num_rows=1,
                    column_values=conditions,
                )
                sample = model.sample_from_conditions(conditions=[condition])
                offspring.append(sample)
            except Exception:
                # Fall back to unconditional sampling and manually
                # override conditioned columns
                sample = model.sample(num_rows=1)
                for col, val in conditions.items():
                    if col in sample.columns:
                        sample[col] = val
                offspring.append(sample)

    if not offspring:
        return pd.DataFrame()

    result = pd.concat(offspring, ignore_index=True)

    # Top up if we're short
    if len(result) < n_samples:
        extra = model.sample(num_rows=n_samples - len(result))
        result = pd.concat([result, extra], ignore_index=True)

    return result.head(n_samples)


def validate_generative_model(
    model,
    original_features: pd.DataFrame,
    n_samples: int = 10_000,
) -> dict:
    """Compare synthetic samples against original data.

    Args:
        model: trained TVAE model.
        original_features: the training data.
        n_samples: number of synthetic samples to generate.

    Returns:
        Dict with validation metrics:
          - marginal_ks: Kolmogorov-Smirnov stats for continuous cols
          - categorical_tvd: total variation distance for cat cols
          - correlation_diff: max absolute difference in correlation
            matrix
    """
    from scipy import stats

    synthetic = model.sample(num_rows=n_samples)
    orig = original_features.drop(
        columns=["household_id", "household_weight"],
        errors="ignore",
    )

    categorical_cols, continuous_cols = identify_column_types(orig)

    # KS test for continuous columns
    ks_stats = {}
    for col in continuous_cols:
        if col in synthetic.columns and col in orig.columns:
            stat, _ = stats.ks_2samp(
                orig[col].dropna().values,
                synthetic[col].dropna().values,
            )
            ks_stats[col] = float(stat)

    # Total variation distance for categorical columns
    tvd = {}
    for col in categorical_cols:
        if col in synthetic.columns and col in orig.columns:
            orig_dist = orig[col].value_counts(normalize=True)
            synth_dist = synthetic[col].value_counts(normalize=True)
            all_vals = set(orig_dist.index) | set(synth_dist.index)
            tv = (
                sum(abs(orig_dist.get(v, 0) - synth_dist.get(v, 0)) for v in all_vals)
                / 2
            )
            tvd[col] = float(tv)

    # Correlation matrix difference (continuous only)
    shared_cont = [
        c for c in continuous_cols if c in synthetic.columns and c in orig.columns
    ]
    if len(shared_cont) >= 2:
        orig_corr = orig[shared_cont].corr().values
        synth_corr = synthetic[shared_cont].corr().values
        corr_diff = float(np.nanmax(np.abs(orig_corr - synth_corr)))
    else:
        corr_diff = None

    return {
        "marginal_ks": ks_stats,
        "categorical_tvd": tvd,
        "correlation_diff": corr_diff,
    }
