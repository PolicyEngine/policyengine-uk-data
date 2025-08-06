from policyengine_uk_data.storage import STORAGE_FOLDER
from policyengine_uk_data.utils.data_upload import upload_data_files


def upload_datasets():
    dataset_files = [
        STORAGE_FOLDER / "frs_2023_24.h5",
        STORAGE_FOLDER / "enhanced_frs_2023_24.h5",
        STORAGE_FOLDER / "parliamentary_constituency_weights.h5",
    ]

    for file_path in dataset_files:
        if not file_path.exists():
            raise ValueError(f"File {file_path} does not exist.")

    upload_data_files(
        files=dataset_files,
        hf_repo_name="policyengine/policyengine-uk-data-private",
        hf_repo_type="model",
        gcs_bucket_name="policyengine-uk-data-private",
    )


if __name__ == "__main__":
    upload_datasets()
