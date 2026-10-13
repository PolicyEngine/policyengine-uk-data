"""Report aggregate Round 8 WAS property reconciliation before imputation.

Run with a licensed household TAB file; stdout contains aggregate JSON only::

    python tools/reconcile_was_property.py --household-tab /licensed/was_r8.tab

No survey records, household identifiers or row examples are emitted. Compare
complete cases first. The separate cleaned report replaces invalid gross
values with zero, so its residuals can reflect missing data as well as a source
identity discrepancy. Neither report applies imputation or uprating.

The historical pre-#501 baseline has three source terms; the stacked #501
baseline has four because it already includes buy-to-let. Model summaries group
the seven raw terms into six inputs, combining second homes and buy-to-let once.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

# Independent transcription of DV_Spec_Property_R8, rows 49–59. Do not derive
# these from the imputation mapping: reconciliation must detect omissions there.
SOURCE_TOTAL = "DVPropertyR8"
SOURCE_COMPONENTS = {
    "main_residence": "DVhValueR8",
    "second_uk_houses": "DVHseValR8_sum",
    "uk_buy_to_let": "DVBltValR8_sum",
    "other_uk_buildings": "DVBlDValR8_sum",
    "uk_land": "DVLUKValR8_sum",
    "overseas_land_and_property": "DVLOSValR8_sum",
    "other_property": "DVOprValR8_sum",
}
# Historical pre-#501 baseline, retained to compare the earlier omissions.
ORIGINAL_COMPONENTS = (
    "main_residence",
    "second_uk_houses",
    "other_uk_buildings",
)
STACK_501_COMPONENTS = (
    "main_residence",
    "second_uk_houses",
    "uk_buy_to_let",
    "other_uk_buildings",
)
# Independent source-to-model transcription. Seven source terms produce six
# inputs because #501 combines second homes and buy-to-let exactly once.
SOURCE_TO_MODEL_INPUT = {
    "DVhValueR8": "main_residence_value",
    "DVHseValR8_sum": "other_residential_property_value",
    "DVBltValR8_sum": "other_residential_property_value",
    "DVBlDValR8_sum": "non_residential_property_value",
    "DVLUKValR8_sum": "owned_land",
    "DVLOSValR8_sum": "overseas_property_value",
    "DVOprValR8_sum": "other_property_value",
}
DEFAULT_WEIGHT_COLUMN = "R8xshhwgt"


def _column(frame: pd.DataFrame, name: str) -> pd.Series:
    matches = [column for column in frame if str(column).lower() == name.lower()]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one source column named {name}")
    return frame[matches[0]]


def _numeric_values(series: pd.Series, *, weight=False):
    numeric = pd.to_numeric(series, errors="coerce").astype(float)
    finite = pd.Series(np.isfinite(numeric), index=series.index)
    invalid = ~finite | (numeric <= 0 if weight else numeric < 0)
    counts = {
        "missing": int(series.isna().sum()),
        "non_numeric": int((series.notna() & numeric.isna()).sum()),
        "non_finite": int((numeric.notna() & ~finite).sum()),
        "negative": int((numeric < 0).sum()),
        "zero": int((numeric == 0).sum()),
        "invalid": int(invalid.sum()),
    }
    if not weight:
        counts.update(
            sentinel_minus_9=int((numeric == -9).sum()),
            sentinel_minus_8=int((numeric == -8).sum()),
            other_negative=int(((numeric < 0) & ~numeric.isin([-9, -8])).sum()),
        )
    return numeric.mask(invalid, 0), ~invalid, counts


def _summary(total, components, weights, valid_weights, tolerance):
    seven_sum = components.sum(axis=1)
    original_sum = components[list(ORIGINAL_COMPONENTS)].sum(axis=1)
    omitted = [name for name in components if name not in ORIGINAL_COMPONENTS]
    omitted_sum = components[omitted].sum(axis=1)
    stack_501_sum = components[list(STACK_501_COMPONENTS)].sum(axis=1)
    remaining = [name for name in components if name not in STACK_501_COMPONENTS]
    remaining_sum = components[remaining].sum(axis=1)
    model_inputs = pd.DataFrame(
        {
            model_input: components[
                [
                    name
                    for name, source_field in SOURCE_COMPONENTS.items()
                    if SOURCE_TO_MODEL_INPUT[source_field] == model_input
                ]
            ].sum(axis=1)
            for model_input in dict.fromkeys(SOURCE_TO_MODEL_INPUT.values())
        }
    )
    model_input_sum = model_inputs.sum(axis=1)
    residual = total - seven_sum

    def money(values):
        return {
            "unweighted_sum": float(values.sum()),
            "weighted_sum": float(
                (values[valid_weights] * weights[valid_weights]).sum()
            ),
        }

    def households(mask):
        return {
            "unweighted_count": int(mask.sum()),
            "weighted_count": float(weights[mask & valid_weights].sum()),
        }

    result = {
        "households": len(total),
        "households_with_valid_weight": int(valid_weights.sum()),
        "source_total": money(total),
        "original_three_component_sum": money(original_sum),
        "seven_component_sum": money(seven_sum),
        "source_total_minus_original_three": money(total - original_sum),
        "four_previously_omitted_components": money(omitted_sum),
        "stack_501_four_component_sum": money(stack_501_sum),
        "source_total_minus_stack_501_four": money(total - stack_501_sum),
        "three_remaining_components": money(remaining_sum),
        "model_input_sum": money(model_input_sum),
        "source_total_minus_model_inputs": money(total - model_input_sum),
        "source_total_minus_seven_components": money(residual),
        "absolute_seven_component_residual": money(residual.abs()),
        "households_with_residual_above_tolerance": households(
            residual.abs() > tolerance
        ),
        "households_owning_only_omitted_categories": households(
            (original_sum == 0) & (omitted_sum > 0)
        ),
        "households_owning_only_remaining_categories": households(
            (stack_501_sum == 0) & (remaining_sum > 0)
        ),
        "components": {},
        "model_inputs": {},
    }
    for name, values in components.items():
        result["components"][name] = {
            "source_field": SOURCE_COMPONENTS[name],
            **money(values),
            "owning_households": households(values > 0),
            "households_owning_only_this_category": households(
                (values > 0) & (seven_sum == values)
            ),
        }
    for name, values in model_inputs.items():
        result["model_inputs"][name] = {
            "source_fields": [
                source_field
                for source_field, model_input in SOURCE_TO_MODEL_INPUT.items()
                if model_input == name
            ],
            **money(values),
            "owning_households": households(values > 0),
            "households_owning_only_this_input": households(
                (values > 0) & (model_input_sum == values)
            ),
        }
    return result


def reconcile_property(
    frame: pd.DataFrame,
    *,
    weight_column: str = DEFAULT_WEIGHT_COLUMN,
    tolerance: float = 0.01,
) -> dict:
    """Summarise both complete-case and zero-filled raw-source identities."""
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("Tolerance must be finite and non-negative")
    source_fields = {"source_total": SOURCE_TOTAL, **SOURCE_COMPONENTS}
    cleaned, valid, invalid_counts = {}, {}, {}
    for name, source_field in source_fields.items():
        cleaned[name], valid[name], invalid_counts[source_field] = _numeric_values(
            _column(frame, source_field)
        )
    weights, valid_weights, weight_counts = _numeric_values(
        _column(frame, weight_column), weight=True
    )
    complete_cases = pd.DataFrame(valid).all(axis=1)
    components = pd.DataFrame({name: cleaned[name] for name in SOURCE_COMPONENTS})
    total = cleaned["source_total"]
    return {
        "round": 8,
        "stage": "raw source before imputation or regional HPI adjustment",
        "tolerance_pounds": tolerance,
        "baselines": {
            "historical_pre_501": {
                "description": (
                    "Historical three-source baseline before #501; "
                    "four omitted categories include buy-to-let"
                ),
                "source_fields": [
                    SOURCE_COMPONENTS[name] for name in ORIGINAL_COMPONENTS
                ],
            },
            "stack_501": {
                "description": (
                    "#501 residential/building aggregate covers four source terms; "
                    "UK land is already exported separately. #543 includes land, "
                    "overseas property and other real estate in the gross aggregate"
                ),
                "source_fields": [
                    SOURCE_COMPONENTS[name] for name in STACK_501_COMPONENTS
                ],
            },
        },
        "source_to_model_input": SOURCE_TO_MODEL_INPUT,
        "handling": {
            "complete_cases": "All eight gross value fields finite and non-negative",
            "cleaned": "Diagnostic only: missing, non-numeric, non-finite and all negative gross values set to zero",
            "known_negative_codes": "-9 not asked/applicable and -8 don't know/refusal (documented for overseas and other property); counted separately",
            "other_negative_values": "Excluded from complete cases and zeroed for the diagnostic; production rejects undocumented negative gross values",
            "weighted_statistics": "Only finite, strictly positive household weights included",
        },
        "gross_value_invalid_counts": invalid_counts,
        "weight_column": weight_column,
        "weight_invalid_counts": weight_counts,
        "households_excluded_from_complete_cases": int((~complete_cases).sum()),
        "complete_cases": _summary(
            total[complete_cases],
            components[complete_cases],
            weights[complete_cases],
            valid_weights[complete_cases],
            tolerance,
        ),
        "cleaned": _summary(total, components, weights, valid_weights, tolerance),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--household-tab", type=Path, required=True)
    parser.add_argument("--weight-column", default=DEFAULT_WEIGHT_COLUMN)
    parser.add_argument("--tolerance", type=float, default=0.01)
    args = parser.parse_args(argv)
    frame = pd.read_csv(args.household_tab, sep="\t", low_memory=False)
    result = reconcile_property(
        frame, weight_column=args.weight_column, tolerance=args.tolerance
    )
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
