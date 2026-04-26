"""Upload the explicitly public UK calibrated transfer dataset artifacts."""

from policyengine_uk_data.datasets import (
    ENHANCED_CPS_FILE,
    ENHANCED_CPS_MANIFEST_FILE,
    ENHANCED_CPS_SOURCE_FILE,
)
from policyengine_uk_data.utils.data_upload import upload_files_to_hf
from policyengine_uk_data.utils.hf_destinations import PUBLIC_REPO


def upload_public_transfer_dataset() -> None:
    files = [
        ENHANCED_CPS_FILE,
        ENHANCED_CPS_SOURCE_FILE,
        ENHANCED_CPS_MANIFEST_FILE,
    ]
    for file_path in files:
        if not file_path.exists():
            raise ValueError(f"File {file_path} does not exist.")

    upload_files_to_hf(
        files=files,
        hf_repo_name=PUBLIC_REPO,
    )


if __name__ == "__main__":
    upload_public_transfer_dataset()
