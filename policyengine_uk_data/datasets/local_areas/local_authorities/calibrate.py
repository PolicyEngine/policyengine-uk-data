from policyengine_uk_data.utils.calibrate import calibrate_local_areas
from policyengine_uk_data.datasets.local_areas.local_authorities.loss import (
    create_local_authority_target_matrix,
    create_national_target_matrix,
)
from policyengine_uk.data import UKSingleYearDataset


def calibrate(
    dataset: UKSingleYearDataset,
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
        excluded_training_targets=[],
        log_csv=None,
        verbose=verbose,
        area_name="Local Authority",
    )


if __name__ == "__main__":
    calibrate()
