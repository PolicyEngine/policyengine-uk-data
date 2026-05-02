import logging
import os

from policyengine_uk_data.utils.build_environment import (
    assert_local_build_environment,
)

logging.basicConfig(level=logging.INFO)


def _modal_calibration_requested() -> bool:
    return os.environ.get("MODAL_CALIBRATE", "0") == "1"


def _get_positive_int_env(name: str, default: int) -> int:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default

    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw_value!r}.") from exc

    if value < 1:
        raise ValueError(f"{name} must be >= 1, got {value}.")

    return value


def _array_values(value):
    return value.values if hasattr(value, "values") else value


def _dump_array(value) -> bytes:
    import io

    import numpy as np

    buffer = io.BytesIO()
    np.save(buffer, value)
    return buffer.getvalue()


def _build_weights_init(dataset, area_count: int, r):
    import numpy as np

    areas_per_household = np.maximum(r.sum(axis=0), 1)
    household_weights = dataset.household.household_weight.values
    original_weights = np.log(
        household_weights / areas_per_household
        + np.random.random(len(household_weights)) * 0.01
    )
    return np.ones((area_count, len(original_weights))) * original_weights


def _prepare_modal_calibration_payload(
    dataset,
    matrix_fn,
    area_count: int,
    m_national_bytes: bytes,
    y_national_bytes: bytes,
):
    import gc
    import numpy as np

    dataset_copy = dataset.copy()
    matrix, y, r = matrix_fn(dataset_copy)
    matrix_values = _array_values(matrix)
    y_values = _array_values(y)
    weights_init = _build_weights_init(dataset_copy, area_count, r)

    payload = {
        "matrix": _dump_array(matrix_values),
        "y": _dump_array(np.nan_to_num(y_values, nan=0.0)),
        "local_target_available": _dump_array(np.isfinite(y_values)),
        "r": _dump_array(r),
        "matrix_national": m_national_bytes,
        "y_national": y_national_bytes,
        "weights_init": _dump_array(weights_init),
    }
    log_inputs = (matrix.copy(), y.copy())

    del dataset_copy, matrix, y, r, matrix_values, y_values, weights_init
    gc.collect()

    return payload, log_inputs


def _load_weight_checkpoint(weight_bytes: bytes):
    import io

    import numpy as np

    return np.load(io.BytesIO(weight_bytes))


def _write_calibration_log(
    checkpoints,
    get_performance,
    matrix,
    y,
    m_national,
    y_national,
    log_csv: str,
):
    import pandas as pd

    performance = pd.DataFrame()
    for epoch, weight_bytes in checkpoints:
        weights = _load_weight_checkpoint(weight_bytes)
        performance_step = get_performance(
            weights,
            matrix,
            y,
            m_national,
            y_national,
            [],
        )
        performance_step["epoch"] = epoch
        performance_step["loss"] = performance_step.rel_abs_error**2
        performance_step["target_name"] = [
            f"{area}/{metric}"
            for area, metric in zip(performance_step.name, performance_step.metric)
        ]
        performance = pd.concat([performance, performance_step], ignore_index=True)

    performance.to_csv(log_csv, index=False)
    return _load_weight_checkpoint(checkpoints[-1][1])


def _run_modal_calibrations(
    frs,
    epochs: int,
    create_constituency_target_matrix,
    create_local_authority_target_matrix,
    create_national_target_matrix,
    get_constituency_performance,
    get_la_performance,
):
    import gc

    import h5py
    import modal

    from policyengine_uk_data.storage import STORAGE_FOLDER
    from policyengine_uk_data.utils.modal_calibrate import app, run_calibration

    m_national, y_national = create_national_target_matrix(frs.copy())
    m_national_bytes = _dump_array(_array_values(m_national))
    y_national_bytes = _dump_array(_array_values(y_national))

    constituency_payload, (matrix_c, y_c) = _prepare_modal_calibration_payload(
        dataset=frs,
        matrix_fn=create_constituency_target_matrix,
        area_count=650,
        m_national_bytes=m_national_bytes,
        y_national_bytes=y_national_bytes,
    )
    la_payload, (matrix_la, y_la) = _prepare_modal_calibration_payload(
        dataset=frs,
        matrix_fn=create_local_authority_target_matrix,
        area_count=360,
        m_national_bytes=m_national_bytes,
        y_national_bytes=y_national_bytes,
    )

    with modal.enable_output(), app.run():
        constituency_future = run_calibration.spawn(
            **constituency_payload,
            epochs=epochs,
        )
        la_future = run_calibration.spawn(
            **la_payload,
            epochs=epochs,
        )
        del constituency_payload, la_payload
        gc.collect()

        constituency_checkpoints = constituency_future.get()
        la_checkpoints = la_future.get()

    constituency_weights = _write_calibration_log(
        checkpoints=constituency_checkpoints,
        get_performance=get_constituency_performance,
        matrix=matrix_c,
        y=y_c,
        m_national=m_national,
        y_national=y_national,
        log_csv="constituency_calibration_log.csv",
    )
    la_weights = _write_calibration_log(
        checkpoints=la_checkpoints,
        get_performance=get_la_performance,
        matrix=matrix_la,
        y=y_la,
        m_national=m_national,
        y_national=y_national,
        log_csv="la_calibration_log.csv",
    )

    with h5py.File(STORAGE_FOLDER / "parliamentary_constituency_weights.h5", "w") as f:
        f.create_dataset("2025", data=constituency_weights)

    with h5py.File(STORAGE_FOLDER / "local_authority_weights.h5", "w") as f:
        f.create_dataset("2025", data=la_weights)

    return constituency_weights, la_weights


def main():
    """Create enhanced FRS dataset with rich progress tracking."""
    try:
        assert_local_build_environment()

        from policyengine_uk.data import UKSingleYearDataset
        from policyengine_uk_data.datasets.disability_benefits import (
            strip_internal_disability_reported_amounts,
        )
        from policyengine_uk_data.datasets.frs import create_frs
        from policyengine_uk_data.storage import STORAGE_FOLDER
        from policyengine_uk_data.utils.progress import (
            ProcessingProgress,
            display_success_panel,
            display_error_panel,
        )
        from policyengine_uk_data.utils.subsample import subsample_dataset
        from policyengine_uk_data.utils.uprating import uprate_dataset

        # Use reduced epochs and fidelity for testing
        is_testing = os.environ.get("TESTING", "0") == "1"
        epochs = 32 if is_testing else 512
        use_modal_calibration = _modal_calibration_requested()
        oa_clones = _get_positive_int_env(
            "PE_UK_DATA_OA_CLONES",
            2 if is_testing else 10,
        )

        progress_tracker = ProcessingProgress()

        # Define dataset creation steps
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
            "Clone and assign OA geography",
            "Uprate to 2025",
            "Calibrate constituency weights",
            "Calibrate local authority weights",
            "Downrate to 2023",
            "Save final dataset",
            "Create tiny datasets",
        ]

        with progress_tracker.track_dataset_creation(steps) as (
            update_dataset,
            nested_progress,
        ):
            # Create base FRS dataset
            update_dataset("Create base FRS dataset", "processing")
            frs = create_frs(
                raw_frs_folder=STORAGE_FOLDER / "frs_2023_24",
                year=2023,
                include_internal_disability_reported_amounts=True,
            )
            strip_internal_disability_reported_amounts(frs).save(
                STORAGE_FOLDER / "frs_2023_24.h5"
            )
            update_dataset("Create base FRS dataset", "completed")

            # Import imputation functions
            from policyengine_uk_data.datasets.imputations import (
                impute_consumption,
                impute_wealth,
                impute_vat,
                impute_income,
                impute_capital_gains,
                impute_services,
                impute_salary_sacrifice,
                impute_student_loan_plan,
                uprate_property_by_region,
            )

            # Apply imputations with progress tracking
            # Wealth must be imputed before consumption because consumption
            # uses num_vehicles as a predictor for fuel spending
            update_dataset("Impute wealth", "processing")
            frs = impute_wealth(frs)
            frs = uprate_property_by_region(frs)
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

            # Clone households and assign OA geography
            update_dataset("Clone and assign OA geography", "processing")
            from policyengine_uk_data.calibration.clone_and_assign import (
                clone_and_assign,
            )

            frs = clone_and_assign(frs, n_clones=oa_clones)
            update_dataset("Clone and assign OA geography", "completed")

            # Uprate dataset
            update_dataset("Uprate to 2025", "processing")
            frs = uprate_dataset(frs, 2025)
            update_dataset("Uprate to 2025", "completed")

            # Calibrate constituency weights with nested progress
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

            if use_modal_calibration:
                update_dataset("Calibrate constituency weights", "processing")
                update_dataset("Calibrate local authority weights", "processing")
                constituency_weights, _ = _run_modal_calibrations(
                    frs=frs,
                    epochs=epochs,
                    create_constituency_target_matrix=create_constituency_target_matrix,
                    create_local_authority_target_matrix=create_local_authority_target_matrix,
                    create_national_target_matrix=create_national_target_matrix,
                    get_constituency_performance=get_performance,
                    get_la_performance=get_la_performance,
                )
                frs_calibrated_constituencies = frs.copy()
                frs_calibrated_constituencies.household.household_weight = (
                    constituency_weights.sum(axis=0)
                )
                update_dataset("Calibrate constituency weights", "completed")
                update_dataset("Calibrate local authority weights", "completed")
            else:
                # Use a separate progress tracker for calibration with nested display
                from policyengine_uk_data.utils.calibrate import (
                    calibrate_local_areas,
                )

                # Run calibration with verbose progress
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
                    verbose=True,  # Enable nested progress display
                    area_name="Constituency",
                    get_performance=get_performance,
                    nested_progress=nested_progress,  # Pass the nested progress manager
                )
                update_dataset("Calibrate constituency weights", "completed")

                # Run calibration with verbose progress
                update_dataset("Calibrate local authority weights", "processing")
                calibrate_local_areas(
                    dataset=frs,
                    epochs=epochs,
                    matrix_fn=create_local_authority_target_matrix,
                    national_matrix_fn=create_national_target_matrix,
                    area_count=360,
                    weight_file="local_authority_weights.h5",
                    excluded_training_targets=[],
                    log_csv="la_calibration_log.csv",
                    verbose=True,  # Enable nested progress display
                    area_name="Local Authority",
                    get_performance=get_la_performance,
                    nested_progress=nested_progress,  # Pass the nested progress manager
                )
                update_dataset("Calibrate local authority weights", "completed")

            # Downrate and save
            update_dataset("Downrate to 2023", "processing")
            frs_calibrated = uprate_dataset(frs_calibrated_constituencies, 2023)
            update_dataset("Downrate to 2023", "completed")

            update_dataset("Save final dataset", "processing")
            strip_internal_disability_reported_amounts(frs_calibrated).save(
                STORAGE_FOLDER / "enhanced_frs_2023_24.h5"
            )
            update_dataset("Save final dataset", "completed")

            # Create tiny (n=1000 households) versions for testing
            update_dataset("Create tiny datasets", "processing")
            TINY_SIZE = 1_000

            frs_base = UKSingleYearDataset(
                file_path=str(STORAGE_FOLDER / "frs_2023_24.h5")
            )
            tiny_frs = subsample_dataset(frs_base, TINY_SIZE)
            tiny_frs.save(STORAGE_FOLDER / "frs_2023_24_tiny.h5")

            tiny_enhanced = subsample_dataset(
                strip_internal_disability_reported_amounts(frs_calibrated),
                TINY_SIZE,
            )
            tiny_enhanced.save(STORAGE_FOLDER / "enhanced_frs_2023_24_tiny.h5")
            update_dataset("Create tiny datasets", "completed")

        # Display success message
        display_success_panel(
            "Dataset creation completed successfully",
            details={
                "base_dataset": "frs_2023_24.h5",
                "enhanced_dataset": "enhanced_frs_2023_24.h5",
                "tiny_base_dataset": "frs_2023_24_tiny.h5",
                "tiny_enhanced_dataset": "enhanced_frs_2023_24_tiny.h5",
                "imputations_applied": "consumption, wealth, VAT, services, income, capital_gains, salary_sacrifice, student_loan_plan",
                "calibration": "national, LA and  constituency targets",
                "calibration_backend": (
                    "Modal GPU" if use_modal_calibration else "CPU"
                ),
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
