import pandas as pd
from policyengine_uk_data.utils.calibrate import calibrate_local_areas
from policyengine_uk_data.datasets.local_areas.local_authorities.loss import (
    create_local_authority_target_matrix,
    create_national_target_matrix,
)
from policyengine_uk_data.storage import STORAGE_FOLDER
from policyengine_uk.data import UKSingleYearDataset


def calibrate(
    dataset: UKSingleYearDataset,
    excluded_training_targets=[],
    log_csv="la_calibration_log.csv",
    verbose: bool = False,
):
    return calibrate_local_areas(
        dataset=dataset,
        matrix_fn=lambda ds: create_local_authority_target_matrix(
            ds, ds.time_period
        ),
        national_matrix_fn=lambda ds: create_national_target_matrix(
            ds, ds.time_period
        ),
        area_count=360,
        weight_file="local_authority_weights.h5",
        excluded_training_targets=excluded_training_targets,
        log_csv=log_csv,
        verbose=verbose,
        area_name="Local Authority",
        get_performance=get_performance,
    )


def get_performance(weights, m_c, y_c, m_n, y_n, excluded_targets):
    la_target_matrix, la_actuals = m_c, y_c
    national_target_matrix, national_actuals = m_n, y_n
    local_authorities = pd.read_csv(
        STORAGE_FOLDER / "local_authorities_2021.csv"
    )
    la_wide = weights @ la_target_matrix
    la_wide.index = local_authorities.code.values
    la_wide["name"] = local_authorities.name.values

    la_results = pd.melt(
        la_wide.reset_index(),
        id_vars=["index", "name"],
        var_name="variable",
        value_name="value",
    )

    la_actuals.index = local_authorities.code.values
    la_actuals["name"] = local_authorities.name.values
    la_actuals_long = pd.melt(
        la_actuals.reset_index(),
        id_vars=["index", "name"],
        var_name="variable",
        value_name="value",
    )

    la_target_validation = pd.merge(
        la_results,
        la_actuals_long,
        on=["index", "variable"],
        suffixes=("_target", "_actual"),
    )
    la_target_validation.drop("name_actual", axis=1, inplace=True)
    la_target_validation.columns = [
        "index",
        "name",
        "metric",
        "estimate",
        "target",
    ]

    la_target_validation["error"] = (
        la_target_validation["estimate"] - la_target_validation["target"]
    )
    la_target_validation["abs_error"] = la_target_validation["error"].abs()
    la_target_validation["rel_abs_error"] = (
        la_target_validation["abs_error"] / la_target_validation["target"]
    )

    national_performance = weights.sum(axis=0) @ national_target_matrix
    national_target_validation = pd.DataFrame(
        {
            "metric": national_performance.index,
            "estimate": national_performance.values,
        }
    )
    national_target_validation["target"] = national_actuals.values

    national_target_validation["error"] = (
        national_target_validation["estimate"]
        - national_target_validation["target"]
    )
    national_target_validation["abs_error"] = national_target_validation[
        "error"
    ].abs()
    national_target_validation["rel_abs_error"] = (
        national_target_validation["abs_error"]
        / national_target_validation["target"]
    )

    df = pd.concat(
        [
            la_target_validation,
            national_target_validation.assign(name="UK", index=0),
        ]
    ).reset_index(drop=True)

    df["validation"] = df.metric.isin(excluded_targets)

    return df


if __name__ == "__main__":
    calibrate()
