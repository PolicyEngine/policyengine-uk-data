"""Asset checks for dataset quality validation."""

from typing import Iterable

import dagster as dg
import numpy as np
import pandas as pd


@dg.multi_asset_check(
    specs=[
        dg.AssetCheckSpec(name="person_count_reasonable", asset="base_frs"),
        dg.AssetCheckSpec(name="household_ratio_valid", asset="base_frs"),
        dg.AssetCheckSpec(name="no_null_columns", asset="base_frs"),
    ]
)
def base_frs_checks(base_frs: dict) -> Iterable[dg.AssetCheckResult]:
    """Structural checks for base FRS dataset."""
    person = base_frs["person"]
    household = base_frs["household"]

    n_people = len(person)
    yield dg.AssetCheckResult(
        check_name="person_count_reasonable",
        passed=bool(20_000 < n_people < 100_000),
        metadata={"person_count": n_people},
    )

    ratio = len(person) / len(household)
    yield dg.AssetCheckResult(
        check_name="household_ratio_valid",
        passed=bool(1.5 < ratio < 4.0),
        metadata={"ratio": round(ratio, 2)},
    )

    null_cols = [c for c in person.columns if person[c].isnull().all()]
    yield dg.AssetCheckResult(
        check_name="no_null_columns",
        passed=bool(len(null_cols) == 0),
        metadata={"null_columns": null_cols},
    )


@dg.multi_asset_check(
    specs=[
        dg.AssetCheckSpec(name="employment_income_rate", asset="enhanced_frs"),
        dg.AssetCheckSpec(name="employment_income_mean", asset="enhanced_frs"),
        dg.AssetCheckSpec(name="self_employment_rate", asset="enhanced_frs"),
        dg.AssetCheckSpec(name="pension_rate_elderly", asset="enhanced_frs"),
    ]
)
def income_checks(enhanced_frs: dict) -> Iterable[dg.AssetCheckResult]:
    """Income distribution sanity checks."""
    person = enhanced_frs["person"]

    if "age" in person.columns and "employment_income" in person.columns:
        working_age = person[(person["age"] >= 18) & (person["age"] < 65)]
        emp_rate = float((working_age["employment_income"] > 0).mean())
        yield dg.AssetCheckResult(
            check_name="employment_income_rate",
            passed=bool(0.35 < emp_rate < 0.85),
            metadata={"rate": f"{emp_rate:.1%}"},
        )

        earners = person[person["employment_income"] > 0]
        mean_income = float(earners["employment_income"].mean())
        yield dg.AssetCheckResult(
            check_name="employment_income_mean",
            passed=bool(15_000 < mean_income < 60_000),
            metadata={"mean": f"£{mean_income:,.0f}"},
        )
    else:
        yield dg.AssetCheckResult(
            check_name="employment_income_rate", passed=True
        )
        yield dg.AssetCheckResult(
            check_name="employment_income_mean", passed=True
        )

    if "self_employment_income" in person.columns and "age" in person.columns:
        working_age = person[(person["age"] >= 18) & (person["age"] < 65)]
        se_rate = float((working_age["self_employment_income"] > 0).mean())
        yield dg.AssetCheckResult(
            check_name="self_employment_rate",
            passed=bool(0.03 < se_rate < 0.25),
            metadata={"rate": f"{se_rate:.1%}"},
        )
    else:
        yield dg.AssetCheckResult(
            check_name="self_employment_rate", passed=True
        )

    if "private_pension_income" in person.columns and "age" in person.columns:
        elderly = person[person["age"] >= 66]
        if len(elderly) > 0:
            pension_rate = float((elderly["private_pension_income"] > 0).mean())
            yield dg.AssetCheckResult(
                check_name="pension_rate_elderly",
                passed=bool(0.15 < pension_rate < 0.80),
                metadata={"rate": f"{pension_rate:.1%}"},
            )
        else:
            yield dg.AssetCheckResult(
                check_name="pension_rate_elderly", passed=True
            )
    else:
        yield dg.AssetCheckResult(
            check_name="pension_rate_elderly", passed=True
        )


@dg.multi_asset_check(
    specs=[
        dg.AssetCheckSpec(name="homeownership_rate", asset="frs_with_wealth"),
        dg.AssetCheckSpec(
            name="vehicle_ownership_rate", asset="frs_with_wealth"
        ),
        dg.AssetCheckSpec(
            name="residence_value_mean", asset="frs_with_wealth"
        ),
    ]
)
def wealth_checks(frs_with_wealth: dict) -> Iterable[dg.AssetCheckResult]:
    """Wealth imputation sanity checks."""
    household = frs_with_wealth["household"]

    if "main_residence_value" in household.columns:
        ownership = float((household["main_residence_value"] > 0).mean())
        yield dg.AssetCheckResult(
            check_name="homeownership_rate",
            passed=bool(0.40 < ownership < 0.80),
            metadata={"rate": f"{ownership:.1%}"},
        )

        owners = household[household["main_residence_value"] > 0]
        if len(owners) > 0:
            mean_val = float(owners["main_residence_value"].mean())
            yield dg.AssetCheckResult(
                check_name="residence_value_mean",
                passed=bool(100_000 < mean_val < 600_000),
                metadata={"mean": f"£{mean_val:,.0f}"},
            )
        else:
            yield dg.AssetCheckResult(
                check_name="residence_value_mean", passed=True
            )
    else:
        yield dg.AssetCheckResult(check_name="homeownership_rate", passed=True)
        yield dg.AssetCheckResult(
            check_name="residence_value_mean", passed=True
        )

    if "num_vehicles" in household.columns:
        vehicle_rate = float((household["num_vehicles"] > 0).mean())
        yield dg.AssetCheckResult(
            check_name="vehicle_ownership_rate",
            passed=bool(0.50 < vehicle_rate < 0.90),
            metadata={"rate": f"{vehicle_rate:.1%}"},
        )
    else:
        yield dg.AssetCheckResult(
            check_name="vehicle_ownership_rate", passed=True
        )


@dg.multi_asset_check(
    specs=[
        dg.AssetCheckSpec(name="child_benefit_rate", asset="enhanced_frs"),
        dg.AssetCheckSpec(name="state_pension_rate", asset="enhanced_frs"),
    ]
)
def benefit_checks(enhanced_frs: dict) -> Iterable[dg.AssetCheckResult]:
    """Benefit receipt sanity checks."""
    person = enhanced_frs["person"]
    household = enhanced_frs["household"]

    if (
        "age" in person.columns
        and "household_id" in person.columns
        and "child_benefit_reported" in household.columns
    ):
        children = person[person["age"] < 16]
        hh_with_children = children["household_id"].unique()
        hh_subset = household[household["household_id"].isin(hh_with_children)]
        if len(hh_subset) > 0:
            cb_rate = float((hh_subset["child_benefit_reported"] > 0).mean())
            yield dg.AssetCheckResult(
                check_name="child_benefit_rate",
                passed=bool(cb_rate > 0.50),
                metadata={"rate": f"{cb_rate:.1%}"},
            )
        else:
            yield dg.AssetCheckResult(
                check_name="child_benefit_rate", passed=True
            )
    else:
        yield dg.AssetCheckResult(check_name="child_benefit_rate", passed=True)

    if "state_pension_reported" in person.columns and "age" in person.columns:
        elderly = person[person["age"] >= 67]
        if len(elderly) > 0:
            sp_rate = float((elderly["state_pension_reported"] > 0).mean())
            yield dg.AssetCheckResult(
                check_name="state_pension_rate",
                passed=bool(sp_rate > 0.70),
                metadata={"rate": f"{sp_rate:.1%}"},
            )
        else:
            yield dg.AssetCheckResult(
                check_name="state_pension_rate", passed=True
            )
    else:
        yield dg.AssetCheckResult(check_name="state_pension_rate", passed=True)


@dg.multi_asset_check(
    specs=[
        dg.AssetCheckSpec(
            name="weights_positive", asset="constituency_weights"
        ),
        dg.AssetCheckSpec(
            name="weights_sum_reasonable", asset="constituency_weights"
        ),
        dg.AssetCheckSpec(
            name="constituency_count", asset="constituency_weights"
        ),
    ]
)
def constituency_weights_checks(
    constituency_weights: np.ndarray,
) -> Iterable[dg.AssetCheckResult]:
    """Constituency weights sanity checks."""
    weights = constituency_weights

    yield dg.AssetCheckResult(
        check_name="weights_positive",
        passed=bool((weights >= 0).all()),
        metadata={"min_weight": float(weights.min())},
    )

    row_sums = weights.sum(axis=1)
    mean_sum = float(row_sums.mean())
    yield dg.AssetCheckResult(
        check_name="weights_sum_reasonable",
        passed=bool(mean_sum > 1000),
        metadata={"mean_sum": f"{mean_sum:,.0f}"},
    )

    n_const = weights.shape[0]
    yield dg.AssetCheckResult(
        check_name="constituency_count",
        passed=bool(n_const >= 600),
        metadata={"count": n_const},
    )


@dg.multi_asset_check(
    specs=[
        dg.AssetCheckSpec(name="weights_positive", asset="la_weights"),
        dg.AssetCheckSpec(name="la_count", asset="la_weights"),
    ]
)
def la_weights_checks(la_weights: np.ndarray) -> Iterable[dg.AssetCheckResult]:
    """Local authority weights sanity checks."""
    weights = la_weights

    yield dg.AssetCheckResult(
        check_name="weights_positive",
        passed=bool((weights >= 0).all()),
        metadata={"min_weight": float(weights.min())},
    )

    n_la = weights.shape[0]
    yield dg.AssetCheckResult(
        check_name="la_count",
        passed=bool(n_la >= 300),
        metadata={"count": n_la},
    )


all_checks = [
    base_frs_checks,
    income_checks,
    wealth_checks,
    benefit_checks,
    constituency_weights_checks,
    la_weights_checks,
]
