from policyengine_core.data import Dataset
from pathlib import Path
import pandas as pd
import warnings
from typing import Type
from policyengine_uk_data.storage import STORAGE_FOLDER


class DWP_FRS(Dataset):
    data_format = Dataset.TABLES
    folder = None

    def generate(self):
        """Generate the survey data from the original TAB files.

        Args:
            tab_folder (Path): The folder containing the original TAB files.
        """

        tab_folder = self.folder

        if isinstance(tab_folder, str):
            tab_folder = Path(tab_folder)

        tab_folder = Path(tab_folder.parent / tab_folder.stem)
        # Load the data
        tables = {}
        for tab_file in tab_folder.glob("*.tab"):
            table_name = tab_file.stem
            if "frs" in table_name:
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                tables[table_name] = pd.read_csv(
                    tab_file, delimiter="\t"
                ).apply(pd.to_numeric, errors="coerce")
                tables[table_name].columns = tables[
                    table_name
                ].columns.str.upper()

            sernum = (
                "sernum"
                if "sernum" in tables[table_name].columns
                else "SERNUM"
            )  # FRS inconsistently users sernum/SERNUM in different years

            if "PERSON" in tables[table_name].columns:
                tables[table_name]["person_id"] = (
                    tables[table_name][sernum] * 1e2
                    + tables[table_name].BENUNIT * 1e1
                    + tables[table_name].PERSON
                ).astype(int)

            if "BENUNIT" in tables[table_name].columns:
                tables[table_name]["benunit_id"] = (
                    tables[table_name][sernum] * 1e2
                    + tables[table_name].BENUNIT * 1e1
                ).astype(int)

            if sernum in tables[table_name].columns:
                tables[table_name]["household_id"] = (
                    tables[table_name][sernum] * 1e2
                ).astype(int)
            if table_name in ("adult", "child"):
                tables[table_name].set_index(
                    "person_id", inplace=True, drop=False
                )
            elif table_name == "benunit":
                tables[table_name].set_index(
                    "benunit_id", inplace=True, drop=False
                )
            elif table_name == "househol":
                tables[table_name].set_index(
                    "household_id", inplace=True, drop=False
                )
        tables["benunit"] = tables["benunit"][
            tables["benunit"].benunit_id.isin(tables["adult"].benunit_id)
        ]
        tables["househol"] = tables["househol"][
            tables["househol"].household_id.isin(tables["adult"].household_id)
        ]

        # Save the data
        self.save_dataset(tables)


class DWP_FRS_2020_21(DWP_FRS):
    folder = STORAGE_FOLDER / "frs_2020_21"
    name = "dwp_frs_2020_21"
    label = "DWP FRS (2020-21)"
    file_path = STORAGE_FOLDER / "dwp_frs_2020_21.h5"
    time_period = 2020


class DWP_FRS_2021_22(DWP_FRS):
    folder = STORAGE_FOLDER / "frs_2021_22"
    name = "dwp_frs_2021_22"
    label = "DWP FRS (2021-22)"
    file_path = STORAGE_FOLDER / "dwp_frs_2021_22.h5"
    time_period = 2021


class DWP_FRS_2022_23(DWP_FRS):
    folder = STORAGE_FOLDER / "frs_2022_23"
    name = "dwp_frs_2022_23"
    label = "DWP FRS (2022-23)"
    file_path = STORAGE_FOLDER / "dwp_frs_2022_23.h5"
    time_period = 2022


if __name__ == "__main__":
    DWP_FRS_2020_21().generate()
    DWP_FRS_2022_23().generate()
