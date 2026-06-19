from policyengine_uk_data.datasets.frs_release import CURRENT_FRS_RELEASE
from policyengine_uk_data.calibration.long_geography import (
    LONG_GEOGRAPHY_WEIGHTS_FILE,
)
from policyengine_uk_data.storage import STORAGE_FOLDER
from policyengine_uk_data.utils.data_upload import upload_data_files
from policyengine_uk_data.utils.hf_destinations import PRIVATE_REPO


def upload_datasets():
    frs_release = CURRENT_FRS_RELEASE
    dataset_files = [
        STORAGE_FOLDER / frs_release.base_dataset_file,
        STORAGE_FOLDER / frs_release.enhanced_dataset_file,
        STORAGE_FOLDER / frs_release.tiny_base_dataset_file,
        STORAGE_FOLDER / frs_release.tiny_enhanced_dataset_file,
        STORAGE_FOLDER / "parliamentary_constituency_weights.h5",
        STORAGE_FOLDER / "local_authority_weights.h5",
        STORAGE_FOLDER / LONG_GEOGRAPHY_WEIGHTS_FILE,
    ]

    for file_path in dataset_files:
        if not file_path.exists():
            raise ValueError(f"File {file_path} does not exist.")

    upload_data_files(
        files=dataset_files,
        hf_repo_name=PRIVATE_REPO,
        hf_repo_type="model",
        gcs_bucket_name="policyengine-uk-data-private",
    )


if __name__ == "__main__":
    upload_datasets()
