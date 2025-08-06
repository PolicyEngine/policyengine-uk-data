import pandas as pd
from policyengine_uk_data.utils.calibrate import calibrate_local_areas
from policyengine_uk_data.datasets.local_areas.constituencies.loss import (
    create_constituency_target_matrix,
    create_national_target_matrix,
)
from policyengine_uk_data.storage import STORAGE_FOLDER
from policyengine_uk.data import UKSingleYearDataset


def calibrate(
    dataset: UKSingleYearDataset,
    excluded_training_targets=[],
    log_csv="calibration_log.csv",
    verbose: bool = False,
):
    return calibrate_local_areas(
        dataset=dataset,
        matrix_fn=create_constituency_target_matrix,
        national_matrix_fn=create_national_target_matrix,
        area_count=650,
        weight_file="parliamentary_constituency_weights.h5",
        excluded_training_targets=excluded_training_targets,
        log_csv=log_csv,
        verbose=verbose,
        area_name="Constituency",
        get_performance=get_performance,
    )


def get_performance(weights, m_c, y_c, m_n, y_n, excluded_targets):
    constituency_target_matrix, constituency_actuals = m_c, y_c
    national_target_matrix, national_actuals = m_n, y_n
    constituencies = pd.read_csv(STORAGE_FOLDER / "constituencies_2024.csv")
    constituency_wide = weights @ constituency_target_matrix
    constituency_wide.index = constituencies.code.values
    constituency_wide["name"] = constituencies.name.values

    constituency_results = pd.melt(
        constituency_wide.reset_index(),
        id_vars=["index", "name"],
        var_name="variable",
        value_name="value",
    )

    constituency_actuals.index = constituencies.code.values
    constituency_actuals["name"] = constituencies.name.values
    constituency_actuals_long = pd.melt(
        constituency_actuals.reset_index(),
        id_vars=["index", "name"],
        var_name="variable",
        value_name="value",
    )

    constituency_target_validation = pd.merge(
        constituency_results,
        constituency_actuals_long,
        on=["index", "variable"],
        suffixes=("_target", "_actual"),
    )
    constituency_target_validation.drop("name_actual", axis=1, inplace=True)
    constituency_target_validation.columns = [
        "index",
        "name",
        "metric",
        "estimate",
        "target",
    ]

    constituency_target_validation["error"] = (
        constituency_target_validation["estimate"]
        - constituency_target_validation["target"]
    )
    constituency_target_validation["abs_error"] = (
        constituency_target_validation["error"].abs()
    )
    constituency_target_validation["rel_abs_error"] = (
        constituency_target_validation["abs_error"]
        / constituency_target_validation["target"]
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
            constituency_target_validation,
            national_target_validation.assign(name="UK", index=0),
        ]
    ).reset_index(drop=True)

    df["validation"] = df.metric.isin(excluded_targets)

    return df


if __name__ == "__main__":
    calibrate()
