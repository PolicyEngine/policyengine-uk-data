from policyengine_uk_data.datasets.frs import create_frs
from policyengine_uk_data.storage import STORAGE_FOLDER
import gc
import logging
import os
import io
import numpy as np
import h5py
from policyengine_uk_data.utils.uprating import uprate_dataset
from policyengine_uk_data.utils.progress import (
    ProcessingProgress,
    display_success_panel,
    display_error_panel,
)

logging.basicConfig(level=logging.INFO)

USE_MODAL = os.environ.get("MODAL_CALIBRATE", "0") == "1"


def _dump(arr) -> bytes:
    buf = io.BytesIO()
    np.save(buf, arr)
    return buf.getvalue()


def _build_weights_init(dataset, area_count, r):
    areas_per_household = np.maximum(r.sum(axis=0), 1)
    original_weights = np.log(
        dataset.household.household_weight.values / areas_per_household
        + np.random.random(len(dataset.household.household_weight.values))
        * 0.01
    )
    return np.ones((area_count, len(original_weights))) * original_weights


def _build_log(checkpoints, get_performance, m_c, y_c, m_n, y_n, log_csv):
    import pandas as pd

    performance = pd.DataFrame()
    for epoch, w_bytes in checkpoints:
        w = np.load(io.BytesIO(w_bytes))
        perf = get_performance(w, m_c, y_c, m_n, y_n, [])
        perf["epoch"] = epoch
        perf["loss"] = perf.rel_abs_error**2
        perf["target_name"] = [
            f"{a}/{m}" for a, m in zip(perf.name, perf.metric)
        ]
        performance = pd.concat([performance, perf], ignore_index=True)
    performance.to_csv(log_csv, index=False)
    final_epoch, final_bytes = checkpoints[-1]
    return np.load(io.BytesIO(final_bytes))


def _run_modal_calibrations(
    frs,
    epochs,
    create_constituency_target_matrix,
    create_local_authority_target_matrix,
    create_national_target_matrix,
    get_constituency_performance,
    get_la_performance,
):
    """
    Dispatch both calibrations concurrently to Modal GPU containers.
    Returns (constituency_weights, la_weights) as numpy arrays and
    writes constituency_calibration_log.csv / la_calibration_log.csv.
    """
    from policyengine_uk_data.utils.modal_calibrate import (
        app,
        run_calibration,
    )

    def _arr(x):
        return x.values if hasattr(x, "values") else x

    # Build national matrix once; keep in memory for log generation
    m_nat, y_nat = create_national_target_matrix(frs.copy())
    b_m_nat = _dump(_arr(m_nat))
    b_y_nat = _dump(_arr(y_nat))

    with app.run():
        # Constituency: build, spawn, keep matrices for log, free before LA
        frs_copy = frs.copy()
        matrix_c, y_c, r_c = create_constituency_target_matrix(frs_copy)
        wi_c = _build_weights_init(frs_copy, 650, r_c)
        fut_c = run_calibration.spawn(
            _dump(_arr(matrix_c)),
            _dump(_arr(y_c)),
            _dump(r_c),
            b_m_nat,
            b_y_nat,
            _dump(wi_c),
            epochs,
        )
        del wi_c, r_c, frs_copy
        gc.collect()

        # LA: build, spawn, keep matrices for log
        frs_copy = frs.copy()
        matrix_la, y_la, r_la = create_local_authority_target_matrix(frs_copy)
        wi_la = _build_weights_init(frs_copy, 360, r_la)
        fut_la = run_calibration.spawn(
            _dump(_arr(matrix_la)),
            _dump(_arr(y_la)),
            _dump(r_la),
            b_m_nat,
            b_y_nat,
            _dump(wi_la),
            epochs,
        )
        del wi_la, r_la, frs_copy
        gc.collect()

        checkpoints_c = fut_c.get()
        checkpoints_la = fut_la.get()

    weights_c = _build_log(
        checkpoints_c,
        get_constituency_performance,
        matrix_c,
        y_c,
        m_nat,
        y_nat,
        "constituency_calibration_log.csv",
    )
    weights_la = _build_log(
        checkpoints_la,
        get_la_performance,
        matrix_la,
        y_la,
        m_nat,
        y_nat,
        "la_calibration_log.csv",
    )

    return weights_c, weights_la


def main():
    """Create enhanced FRS dataset with rich progress tracking."""
    try:
        is_testing = os.environ.get("TESTING", "0") == "1"
        epochs = 32 if is_testing else 512

        progress_tracker = ProcessingProgress()

        steps = [
            "Create base FRS dataset",
            "Impute consumption",
            "Impute wealth",
            "Impute VAT",
            "Impute public service usage",
            "Impute income",
            "Impute capital gains",
            "Impute salary sacrifice",
            "Impute student loan plan",
            "Uprate to 2025",
            "Calibrate constituency weights",
            "Calibrate local authority weights",
            "Downrate to 2023",
            "Save final dataset",
        ]

        with progress_tracker.track_dataset_creation(steps) as (
            update_dataset,
            nested_progress,
        ):
            update_dataset("Create base FRS dataset", "processing")
            frs = create_frs(
                raw_frs_folder=STORAGE_FOLDER / "frs_2023_24",
                year=2023,
            )
            frs.save(STORAGE_FOLDER / "frs_2023_24.h5")
            update_dataset("Create base FRS dataset", "completed")

            if USE_MODAL:
                from policyengine_uk_data.utils.modal_calibrate import (
                    app,
                    run_imputation,
                )
                from policyengine_uk.data import UKSingleYearDataset
                import tempfile

                for step in [
                    "Impute consumption",
                    "Impute wealth",
                    "Impute VAT",
                    "Impute public service usage",
                    "Impute income",
                    "Impute capital gains",
                    "Impute salary sacrifice",
                    "Impute student loan plan",
                    "Uprate to 2025",
                ]:
                    update_dataset(step, "processing")

                with app.run():
                    frs_bytes = open(
                        STORAGE_FOLDER / "frs_2023_24.h5", "rb"
                    ).read()
                    frs_bytes_out = run_imputation.remote(frs_bytes, year=2023)

                with tempfile.NamedTemporaryFile(
                    suffix=".h5", delete=False
                ) as f:
                    f.write(frs_bytes_out)
                    frs_path = f.name
                frs = UKSingleYearDataset(file_path=frs_path)

                for step in [
                    "Impute consumption",
                    "Impute wealth",
                    "Impute VAT",
                    "Impute public service usage",
                    "Impute income",
                    "Impute capital gains",
                    "Impute salary sacrifice",
                    "Impute student loan plan",
                    "Uprate to 2025",
                ]:
                    update_dataset(step, "completed")
            else:
                from policyengine_uk_data.datasets.imputations import (
                    impute_consumption,
                    impute_wealth,
                    impute_vat,
                    impute_income,
                    impute_capital_gains,
                    impute_services,
                    impute_salary_sacrifice,
                    impute_student_loan_plan,
                )

                update_dataset("Impute wealth", "processing")
                frs = impute_wealth(frs)
                update_dataset("Impute wealth", "completed")

                update_dataset("Impute consumption", "processing")
                frs = impute_consumption(frs)
                update_dataset("Impute consumption", "completed")

                update_dataset("Impute VAT", "processing")
                frs = impute_vat(frs)
                update_dataset("Impute VAT", "completed")

                update_dataset("Impute public service usage", "processing")
                frs = impute_services(frs)
                update_dataset("Impute public service usage", "completed")

                update_dataset("Impute income", "processing")
                frs = impute_income(frs)
                update_dataset("Impute income", "completed")

                update_dataset("Impute capital gains", "processing")
                frs = impute_capital_gains(frs)
                update_dataset("Impute capital gains", "completed")

                update_dataset("Impute salary sacrifice", "processing")
                frs = impute_salary_sacrifice(frs)
                update_dataset("Impute salary sacrifice", "completed")

                update_dataset("Impute student loan plan", "processing")
                frs = impute_student_loan_plan(frs, year=2025)
                update_dataset("Impute student loan plan", "completed")

            if not USE_MODAL:
                update_dataset("Uprate to 2025", "processing")
                frs = uprate_dataset(frs, 2025)
                update_dataset("Uprate to 2025", "completed")

            from policyengine_uk_data.datasets.local_areas.constituencies.loss import (
                create_constituency_target_matrix,
            )
            from policyengine_uk_data.targets.build_loss_matrix import (
                create_target_matrix as create_national_target_matrix,
            )
            from policyengine_uk_data.datasets.local_areas.constituencies.calibrate import (
                get_performance,
            )
            from policyengine_uk_data.datasets.local_areas.local_authorities.calibrate import (
                get_performance as get_la_performance,
            )
            from policyengine_uk_data.datasets.local_areas.local_authorities.loss import (
                create_local_authority_target_matrix,
            )

            if USE_MODAL:
                update_dataset("Calibrate constituency weights", "processing")
                update_dataset(
                    "Calibrate local authority weights", "processing"
                )

                weights_c, weights_la = _run_modal_calibrations(
                    frs,
                    epochs,
                    create_constituency_target_matrix,
                    create_local_authority_target_matrix,
                    create_national_target_matrix,
                    get_performance,
                    get_la_performance,
                )

                with h5py.File(
                    STORAGE_FOLDER / "parliamentary_constituency_weights.h5",
                    "w",
                ) as f:
                    f.create_dataset("2025", data=weights_c)

                with h5py.File(
                    STORAGE_FOLDER / "local_authority_weights.h5", "w"
                ) as f:
                    f.create_dataset("2025", data=weights_la)

                frs_calibrated_constituencies = frs.copy()
                frs_calibrated_constituencies.household.household_weight = (
                    weights_c.sum(axis=0)
                )

                update_dataset("Calibrate constituency weights", "completed")
                update_dataset(
                    "Calibrate local authority weights", "completed"
                )
            else:
                from policyengine_uk_data.utils.calibrate import (
                    calibrate_local_areas,
                )

                update_dataset("Calibrate constituency weights", "processing")
                frs_calibrated_constituencies = calibrate_local_areas(
                    dataset=frs,
                    epochs=epochs,
                    matrix_fn=create_constituency_target_matrix,
                    national_matrix_fn=create_national_target_matrix,
                    area_count=650,
                    weight_file="parliamentary_constituency_weights.h5",
                    excluded_training_targets=[],
                    log_csv="constituency_calibration_log.csv",
                    verbose=True,
                    area_name="Constituency",
                    get_performance=get_performance,
                    nested_progress=nested_progress,
                )
                update_dataset("Calibrate constituency weights", "completed")

                update_dataset(
                    "Calibrate local authority weights", "processing"
                )
                calibrate_local_areas(
                    dataset=frs,
                    epochs=epochs,
                    matrix_fn=create_local_authority_target_matrix,
                    national_matrix_fn=create_national_target_matrix,
                    area_count=360,
                    weight_file="local_authority_weights.h5",
                    excluded_training_targets=[],
                    log_csv="la_calibration_log.csv",
                    verbose=True,
                    area_name="Local Authority",
                    get_performance=get_la_performance,
                    nested_progress=nested_progress,
                )
                update_dataset(
                    "Calibrate local authority weights", "completed"
                )

            update_dataset("Downrate to 2023", "processing")
            frs_calibrated = uprate_dataset(
                frs_calibrated_constituencies, 2023
            )
            update_dataset("Downrate to 2023", "completed")

            update_dataset("Save final dataset", "processing")
            frs_calibrated.save(STORAGE_FOLDER / "enhanced_frs_2023_24.h5")
            update_dataset("Save final dataset", "completed")

        display_success_panel(
            "Dataset creation completed successfully",
            details={
                "base_dataset": "frs_2023_24.h5",
                "enhanced_dataset": "enhanced_frs_2023_24.h5",
                "imputations_applied": "consumption, wealth, VAT, services, income, capital_gains, salary_sacrifice, student_loan_plan",
                "calibration": "national, LA and constituency targets",
                "calibration_backend": "Modal GPU" if USE_MODAL else "CPU",
            },
        )

    except Exception as e:
        display_error_panel(
            f"Dataset creation failed: {str(e)}",
            suggestions=[
                "Check that all required data files are present in storage folder",
                "Verify sufficient disk space for dataset creation",
                "Review log files for detailed error information",
            ],
        )
        raise


if __name__ == "__main__":
    main()
