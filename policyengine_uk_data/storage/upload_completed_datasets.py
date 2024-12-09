from policyengine_uk_data.datasets import EnhancedFRS_2022_23, FRS_2022_23
from policyengine_uk_data.storage import STORAGE_FOLDER
from policyengine_uk_data.utils.huggingface import upload


def upload_datasets():
    for dataset in [FRS_2022_23, EnhancedFRS_2022_23]:
        dataset = dataset()
        if not dataset.exists:
            raise ValueError(
                f"Dataset {dataset.name} does not exist at {dataset.file_path}."
            )

        upload(
            dataset.file_path,
            "policyengine/policyengine-uk-data",
            dataset.file_path.name,
        )

    # Constituency weights:

    upload(
        STORAGE_FOLDER / "parliamentary_constituency_weights.h5",
        "policyengine/policyengine-uk-data",
        "parliamentary_constituency_weights.h5",
    )

    # Local authority weights:

    upload(
        STORAGE_FOLDER / "local_authority_weights.h5",
        "policyengine/policyengine-uk-data",
        "local_authority_weights.h5",
    )


if __name__ == "__main__":
    upload_datasets()
