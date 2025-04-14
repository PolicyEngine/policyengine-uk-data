from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import BaseModel

from policyengine_uk_data_v2.impute import QRF
from policyengine_uk_data_v2.utils import save_dataframes_to_h5, load_dataframes_from_h5, load_parameters

import yaml
import logging
from policyengine_uk.system import parameters as policy_parameters

from .ids import add_ids
from .housing import add_housing
from .income import add_incomes
from .demographics import add_demographics
from .benefits import add_benefits
from .ukda import FRS, load_frs_tables, FRS_TABLE_NAMES

class PolicyEngineFRSDataset:
    frs: FRS
    person: pd.DataFrame
    benunit: pd.DataFrame
    household: pd.DataFrame
    state: pd.DataFrame

    def save_to_h5(
        self,
        file_path: str | Path,
    ):
        save_dataframes_to_h5(
            person=self.person,
            benunit=self.benunit,
            household=self.household,
            state=self.state,
            output_path=file_path,
            year=self.year,
        )

    def load_from_h5(
        self,
        file_path: str | Path,
        year: int,
    ):
        """
        Load the dataframes from the given file path.
        """

        # Load the dataframes from the given file path
        dfs = load_dataframes_from_h5(
            file_path,
        )
        self.person = dfs.get("person")
        self.benunit = dfs.get("benunit")
        self.household = dfs.get("household")
        self.state = dfs.get("state")

        # Set the year attribute
        self.year = year

    def save_to_dataframes(
        self,
        folder: str | Path,
    ):
        """
        Save the dataframes to the given folder.
        """
        folder = Path(folder)
        self.person.to_csv(folder / "person.csv", index=False)
        self.benunit.to_csv(folder / "benunit.csv", index=False)
        self.household.to_csv(folder / "household.csv", index=False)
        self.state.to_csv(folder / "state.csv", index=False)

    def load_from_dataframes(
        self,
        folder: str | Path,
        year: int,
    ):
        """
        Load the dataframes from the given folder.
        """
        folder = Path(folder)
        self.person = pd.read_csv(folder / "person.csv")
        self.benunit = pd.read_csv(folder / "benunit.csv")
        self.household = pd.read_csv(folder / "household.csv")
        self.state = pd.read_csv(folder / "state.csv")
        self.year = year

    def build(self, year: int, tab_folder: str | Path = None):
        self.year = year
        frs = load_frs_tables(
            tab_folder,
        )

        person, benunit, household, state = (
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
        )


        self.count_adults = len(frs.adult)
        self.count_children = len(frs.child)
        self.count_people = self.count_adults + self.count_children

        self.zero_for_children = np.zeros(self.count_children)
        self.false_for_children = np.zeros(self.count_children, dtype=bool)

        data_parameters = load_parameters()
        # Add ID variables to original frs tables for convenience
        for table_name in FRS_TABLE_NAMES:
            table = getattr(frs, table_name)
            if "PERSON" in table.columns:
                table["person_id"] = (
                    table["SERNUM"] * 1e2 + table.BENUNIT * 1e1 + table.PERSON
                ).astype(int)

            if "BENUNIT" in table.columns:
                table["benunit_id"] = (
                    table["SERNUM"] * 1e2 + table.BENUNIT * 1e1
                ).astype(int)

            if "SERNUM" in table.columns:
                table["household_id"] = (table["SERNUM"] * 1e2).astype(int)
        _frs_person = (
            pd.concat([frs.adult, frs.child], axis=0).fillna(0).reset_index()
        )
        # Add primary and foreign keys
        person, benunit, household, state = add_ids(
            person,
            benunit,
            household,
            state,
            frs,
            _frs_person,
        )

        person, household = add_demographics(
            person,
            household,
            frs,
            _frs_person,
            year,
        )

        person, benunit, household, state = add_incomes(
            person,
            benunit,
            household,
            state,
            frs,
            _frs_person,
        )

        person, benunit, household, state = add_housing(
            person,
            benunit,
            household,
            state,
            frs,
            year,
        )

        person, benunit, household = add_benefits(person, benunit, _frs_person, frs, policy_parameters(year))

        self.person = person
        self.benunit = benunit
        self.household = household
        self.state = state


