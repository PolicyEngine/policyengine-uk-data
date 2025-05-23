from policyengine_uk_data.datasets import EnhancedFRS_2022_23, FRS_2022_23
from policyengine_uk_data.storage import STORAGE_FOLDER
from policyengine_uk_data.utils.huggingface import upload
from google.cloud import storage
import google.auth
import os
from importlib import metadata
import json


def upload_datasets():
    credentials, project_id = google.auth.default()
    storage_client = storage.Client(
        credentials=credentials, project=project_id
    )
    bucket = storage_client.bucket("policyengine-uk-data-private")

    # Upload versions

    with open(STORAGE_FOLDER / "version.json", "w") as f:
        f.write(
            json.dumps({"version": metadata.version("policyengine-uk-data")})
        )

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
        blob = dataset.file_path.name
        blob = bucket.blob(blob)
        blob.upload_from_filename(dataset.file_path)
        print(
            f"Uploaded {dataset.file_path.name} to GCS bucket policyengine-uk-data-private."
        )

    # Constituency weights:

    upload(
        STORAGE_FOLDER / "parliamentary_constituency_weights.h5",
        "policyengine/policyengine-uk-data",
        "parliamentary_constituency_weights.h5",
    )

    blob = "parliamentary_constituency_weights.h5"
    blob = bucket.blob(blob)
    blob.upload_from_filename(
        STORAGE_FOLDER / "parliamentary_constituency_weights.h5"
    )
    print(
        f"Uploaded parliamentary_constituency_weights.h5 to GCS bucket policyengine-uk-data-private."
    )

    # Local authority weights:

    upload(
        STORAGE_FOLDER / "local_authority_weights.h5",
        "policyengine/policyengine-uk-data",
        "local_authority_weights.h5",
    )

    blob = "local_authority_weights.h5"
    blob = bucket.blob(blob)
    blob.upload_from_filename(STORAGE_FOLDER / "local_authority_weights.h5")
    print(
        f"Uploaded local_authority_weights.h5 to GCS bucket policyengine-uk-data-private."
    )

    upload(
        STORAGE_FOLDER / "version.json",
        "policyengine/policyengine-uk-data",
        "version.json",
    )

    blob = "version.json"
    blob = bucket.blob(blob)
    blob.upload_from_filename(STORAGE_FOLDER / "version.json")
    print(f"Uploaded version.json to GCS bucket policyengine-uk-data-private.")


if __name__ == "__main__":
    upload_datasets()
