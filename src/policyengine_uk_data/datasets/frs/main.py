from pathlib import Path
from typing import Tuple, Union, Optional, Dict, Any

import numpy as np
import pandas as pd
from policyengine_core.parameters import ParameterNode
from policyengine_uk.system import parameters as policy_parameters

from policyengine_uk_data.utils.datasets import (
    load_dataframes_from_h5,
    save_dataframes_to_h5,
)

from .benefits import add_benefits
from .demographics import add_demographics
from .housing import add_housing
from .ids import add_ids
from .income import add_incomes
from .ukda import FRS, FRS_TABLE_NAMES, load_frs_tables


class PolicyEngineFRSDataset:
    frs: FRS
    person: pd.DataFrame
    benunit: pd.DataFrame
    household: pd.DataFrame
    state: pd.DataFrame
    year: int
    count_adults: int
    count_children: int
    count_people: int
    zero_for_children: np.ndarray
    false_for_children: np.ndarray

    def save_to_h5(
        self,
        file_path: Union[str, Path],
    ) -> None:
        """Save the dataset to an HDF5 file.
        
        Args:
            file_path (Union[str, Path]): Path to save the HDF5 file.
        """
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
        file_path: Union[str, Path],
        year: int,
    ) -> None:
        """Load the dataset from an HDF5 file.
        
        Args:
            file_path (Union[str, Path]): Path to the HDF5 file.
            year (int): The year associated with the dataset.
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
        folder: Union[str, Path],
    ) -> None:
        """Save the dataframes to CSV files in the specified folder.
        
        Args:
            folder (Union[str, Path]): The folder to save the CSV files to.
        """
        folder = Path(folder)
        self.person.to_csv(folder / "person.csv", index=False)
        self.benunit.to_csv(folder / "benunit.csv", index=False)
        self.household.to_csv(folder / "household.csv", index=False)
        self.state.to_csv(folder / "state.csv", index=False)

    def load_from_dataframes(
        self,
        folder: Union[str, Path],
        year: int,
    ) -> None:
        """Load the dataset from CSV files in the specified folder.
        
        Args:
            folder (Union[str, Path]): The folder containing the CSV files.
            year (int): The year associated with the dataset.
        """
        folder = Path(folder)
        self.person = pd.read_csv(folder / "person.csv")
        self.benunit = pd.read_csv(folder / "benunit.csv")
        self.household = pd.read_csv(folder / "household.csv")
        self.state = pd.read_csv(folder / "state.csv")
        self.year = year

    def build(
        self, 
        year: int, 
        tab_folder: Optional[Union[str, Path]] = None
    ) -> None:
        """Build the FRS dataset for the specified year.
        
        Args:
            year (int): The year to build the dataset for.
            tab_folder (Optional[Union[str, Path]], optional): Folder containing the tab-delimited FRS files. Defaults to None.
        """
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
        _frs_person = pd.concat([frs.adult, frs.child], axis=0).fillna(0).reset_index()
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

        person, benunit, household = add_benefits(
            person, benunit, household, _frs_person, frs, policy_parameters(year)
        )

        self.person = person
        self.benunit = benunit
        self.household = household
        self.state = state
