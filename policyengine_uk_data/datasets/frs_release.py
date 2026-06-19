import os
from dataclasses import dataclass


@dataclass(frozen=True)
class FRSRelease:
    name: str
    survey_year: int
    base_year: int
    calibration_year: int
    ukds_study_number: int
    doi: str
    ukds_tab_zip_filename: str
    ukds_tab_zip_sha256: str
    ukds_tab_subdir: str

    @property
    def raw_zip_name(self) -> str:
        return f"{self.name}.zip"

    @property
    def base_dataset_name(self) -> str:
        return self.name

    @property
    def enhanced_dataset_name(self) -> str:
        return f"enhanced_{self.name}"

    @property
    def tiny_base_dataset_name(self) -> str:
        return f"{self.name}_tiny"

    @property
    def tiny_enhanced_dataset_name(self) -> str:
        return f"enhanced_{self.name}_tiny"

    @property
    def base_dataset_file(self) -> str:
        return f"{self.base_dataset_name}.h5"

    @property
    def enhanced_dataset_file(self) -> str:
        return f"{self.enhanced_dataset_name}.h5"

    @property
    def tiny_base_dataset_file(self) -> str:
        return f"{self.tiny_base_dataset_name}.h5"

    @property
    def tiny_enhanced_dataset_file(self) -> str:
        return f"{self.tiny_enhanced_dataset_name}.h5"


FRS_2024_25 = FRSRelease(
    name="frs_2024_25",
    survey_year=2024,
    base_year=2024,
    calibration_year=2025,
    ukds_study_number=9563,
    doi="http://doi.org/10.5255/UKDA-SN-9563-1",
    ukds_tab_zip_filename=(
        "9563tab_05DD0069587DBD25E5719D355CE05FC0827D5EDD58C24ECE9"
        "AB85ACD954A9AEB_V1.zip"
    ),
    ukds_tab_zip_sha256=(
        "05dd0069587dbd25e5719d355ce05fc0827d5edd58c24ece9ab85acd954a9aeb"
    ),
    ukds_tab_subdir="UKDA-9563-tab/tab",
)

# FRS 2023-24 (UKDS SN 9252). The raw tabs live flat at the root of the HF
# `frs_2023_24.zip`, so the extractor's flat-file fallback handles them (the
# UKDA-9252 subdir below is provenance only). The zip filename/sha256 here
# describe the HuggingFace prerequisite the build actually downloads; neither
# is verified at build time. Provided so the 2023-24 enhanced FRS can be
# rebuilt with the current loader (which now populates employment_sector and
# sic_industry_division), then published by the data controller.
FRS_2023_24 = FRSRelease(
    name="frs_2023_24",
    survey_year=2023,
    base_year=2023,
    calibration_year=2024,
    ukds_study_number=9252,
    doi="http://doi.org/10.5255/UKDA-SN-9252-1",
    ukds_tab_zip_filename="frs_2023_24.zip",
    ukds_tab_zip_sha256=(
        "86843cef448510d3b54aa1218a3bf17f5804c1af91a7a71f31176c231b2f1058"
    ),
    ukds_tab_subdir="UKDA-9252-tab/tab",
)

RELEASES = {release.name: release for release in (FRS_2024_25, FRS_2023_24)}

DEFAULT_FRS_RELEASE = "frs_2024_25"


def get_frs_release() -> FRSRelease:
    """Resolve the FRS release to build/load.

    Defaults to the current release (2024-25). Set the
    ``PE_UK_DATA_FRS_RELEASE`` environment variable (e.g. ``frs_2023_24``) to
    target another release without editing code — used to rebuild and publish
    an earlier enhanced FRS with the current loader.
    """
    name = os.environ.get("PE_UK_DATA_FRS_RELEASE", DEFAULT_FRS_RELEASE)
    if name not in RELEASES:
        raise ValueError(
            f"Unknown FRS release {name!r}; choose from {sorted(RELEASES)}."
        )
    return RELEASES[name]


CURRENT_FRS_RELEASE = get_frs_release()
