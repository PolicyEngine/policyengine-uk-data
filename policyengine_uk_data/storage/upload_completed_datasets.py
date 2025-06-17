from policyengine_uk_data.datasets import EnhancedFRS_2022_23, FRS_2022_23
from policyengine_uk_data.storage import STORAGE_FOLDER
from policyengine_uk_data.utils.data_upload import upload_data_files


def upload_datasets():
    dataset_files = [
        FRS_2022_23.file_path,
        EnhancedFRS_2022_23.file_path,
        STORAGE_FOLDER / "parliamentary_constituency_weights.h5",
        STORAGE_FOLDER / "local_authority_weights.h5",
    ]

    public_dataset_files = [
        STORAGE_FOLDER / "synthetic_frs_2022_23.h5",
    ]

    for file_path in dataset_files + public_dataset_files:
        if not file_path.exists():
            raise ValueError(f"File {file_path} does not exist.")

    upload_data_files(
        files=dataset_files,
        hf_repo_name="policyengine/policyengine-uk-data-private",
        hf_repo_type="model",
        gcs_bucket_name="policyengine-uk-data-private",
    )

    upload_data_files(
        files=public_dataset_files,
        hf_repo_name="policyengine/policyengine-uk-data-public",
        hf_repo_type="model",
        gcs_bucket_name="policyengine-uk-data-public",
    )


if __name__ == "__main__":
    upload_datasets()
