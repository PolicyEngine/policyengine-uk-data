from dataclasses import dataclass


@dataclass(frozen=True)
class LCFSRelease:
    name: str
    survey_year: int
    fuel_price_year: int
    ukds_study_number: int
    doi: str
    household_tab_filename: str
    person_tab_filename: str

    @property
    def raw_zip_name(self) -> str:
        return f"{self.name}.zip"


@dataclass(frozen=True)
class WASRelease:
    name: str
    latest_round: int
    end_year: int
    ukds_study_number: int
    doi: str
    household_tab_filename: str
    person_tab_filename: str

    @property
    def raw_zip_name(self) -> str:
        return f"{self.name}.zip"


@dataclass(frozen=True)
class ETBRelease:
    name: str
    latest_year: int
    default_training_year: int
    ukds_study_number: int
    doi: str
    household_tab_filename: str
    person_tab_filename: str

    @property
    def raw_zip_name(self) -> str:
        return f"{self.name}.zip"


CURRENT_LCFS_RELEASE = LCFSRelease(
    name="lcfs_2023_24",
    survey_year=2023,
    fuel_price_year=2023,
    ukds_study_number=9468,
    doi="10.5255/UKDA-SN-9468-3",
    household_tab_filename="9468_dvhh_ukanon_v2_2023.tab",
    person_tab_filename="9468_dvper_ukanon_202324_2023.tab",
)


CURRENT_WAS_RELEASE = WASRelease(
    name="was_2006_22",
    latest_round=8,
    end_year=2022,
    ukds_study_number=7215,
    doi="10.5255/UKDA-SN-7215-20",
    household_tab_filename="7215_was_round_8_hhold_eul_may_2025_230525.tab",
    person_tab_filename="7215_was_round_8_person_eul_may_2025_230525.tab",
)


CURRENT_ETB_RELEASE = ETBRelease(
    name="etb_1977_24",
    latest_year=2024,
    default_training_year=2023,
    ukds_study_number=8856,
    doi="10.5255/UKDA-SN-8856-4",
    household_tab_filename="8856_householdv2_1977-2024.tab",
    person_tab_filename="8856_personv2_2018-2024.tab",
)
