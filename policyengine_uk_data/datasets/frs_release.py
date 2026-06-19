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


CURRENT_FRS_RELEASE = FRSRelease(
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
