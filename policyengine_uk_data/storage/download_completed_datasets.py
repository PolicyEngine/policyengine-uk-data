from policyengine_uk_data.utils.huggingface import download, upload
from pathlib import Path


FOLDER = Path(__file__).parent

FILES = [
    "enhanced_frs_2022_23.h5",
    "frs_2022_23.h5",
    "parliamentary_constituency_weights.h5",
    "local_authority_weights.h5",
]

FILES = [FOLDER / file for file in FILES]

for file in FILES:
    download(
        repo="policyengine/policyengine-uk-data",
        repo_filename=file.name,
        local_folder=file.parent,
    )
