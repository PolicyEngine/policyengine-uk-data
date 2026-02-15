from policyengine_uk_data.datasets.frs import create_frs
from policyengine_uk_data.storage import STORAGE_FOLDER
import logging
import os
from policyengine_uk.data import UKSingleYearDataset
from policyengine_uk_data.utils.uprating import uprate_dataset
from policyengine_uk_data.utils.progress import (
    ProcessingProgress,
    display_success_panel,
    display_error_panel,
)

logging.basicConfig(level=logging.INFO)


def main():
    """Create enhanced FRS dataset with rich progress tracking."""
    try:
        # Use reduced epochs and fidelity for testing
        is_testing = os.environ.get("TESTING", "0") == "1"
        epochs = 32 if is_testing else 512

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

            # Create base FRS dataset
            update_dataset("Create base FRS dataset", "processing")
            frs = create_frs(
                raw_frs_folder=STORAGE_FOLDER / "frs_2023_24",
                year=2023,
            )
            frs.save(STORAGE_FOLDER / "frs_2023_24.h5")
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
            )

            # Apply imputations with progress tracking
            # Wealth must be imputed before consumption because consumption
            # uses num_vehicles as a predictor for fuel spending
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
            frs = impute_student_loan_plan(frs, year=2023)
            update_dataset("Impute student loan plan", "completed")

            # Uprate dataset
            update_dataset("Uprate to 2025", "processing")
            frs = uprate_dataset(frs, 2025)
            update_dataset("Uprate to 2025", "completed")

            # Calibrate constituency weights with nested progress
            from policyengine_uk_data.datasets.local_areas.constituencies.calibrate import (
                calibrate,
            )

            update_dataset("Calibrate constituency weights", "processing")

            # Use a separate progress tracker for calibration with nested display
            from policyengine_uk_data.utils.calibrate import (
                calibrate_local_areas,
            )
            from policyengine_uk_data.datasets.local_areas.constituencies.loss import (
                create_constituency_target_matrix,
            )
            from policyengine_uk_data.targets.build_loss_matrix import (
                create_target_matrix as create_national_target_matrix,
            )
            from policyengine_uk_data.datasets.local_areas.constituencies.calibrate import (
                get_performance,
            )

            # Run calibration with verbose progress
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

            from policyengine_uk_data.datasets.local_areas.local_authorities.calibrate import (
                get_performance as get_la_performance,
            )
            from policyengine_uk_data.datasets.local_areas.local_authorities.loss import (
                create_local_authority_target_matrix,
            )

            # Run calibration with verbose progress
            frs_calibrated_las = calibrate_local_areas(
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

            update_dataset("Calibrate dataset", "completed")

            # Downrate and save
            update_dataset("Downrate to 2023", "processing")
            frs_calibrated = uprate_dataset(
                frs_calibrated_constituencies, 2023
            )
            update_dataset("Downrate to 2023", "completed")

            update_dataset("Save final dataset", "processing")
            frs_calibrated.save(STORAGE_FOLDER / "enhanced_frs_2023_24.h5")
            update_dataset("Save final dataset", "completed")

        # Display success message
        display_success_panel(
            "Dataset creation completed successfully",
            details={
                "base_dataset": "frs_2023_24.h5",
                "enhanced_dataset": "enhanced_frs_2023_24.h5",
                "imputations_applied": "consumption, wealth, VAT, services, income, capital_gains, salary_sacrifice, student_loan_plan",
                "calibration": "national, LA and  constituency targets",
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
