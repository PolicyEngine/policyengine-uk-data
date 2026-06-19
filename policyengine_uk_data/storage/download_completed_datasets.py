from policyengine_uk_data.datasets.frs_release import CURRENT_FRS_RELEASE
from policyengine_uk_data.calibration.long_geography import (
    LONG_GEOGRAPHY_WEIGHTS_FILE,
)
from policyengine_uk_data.utils.hf_destinations import PRIVATE_REPO
from policyengine_uk_data.utils.huggingface import download
from pathlib import Path

FOLDER = Path(__file__).parent

FILES = [
    CURRENT_FRS_RELEASE.enhanced_dataset_file,
    CURRENT_FRS_RELEASE.base_dataset_file,
    "parliamentary_constituency_weights.h5",
    "local_authority_weights.h5",
    LONG_GEOGRAPHY_WEIGHTS_FILE,
]

FILES = [FOLDER / file for file in FILES]

for file in FILES:
    download(
        repo=PRIVATE_REPO,
        repo_filename=file.name,
        local_folder=file.parent,
    )
