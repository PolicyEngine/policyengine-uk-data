from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import BaseModel

from policyengine_uk_data_v2.impute import QRF

FRS_TABLE_NAMES = (
    "adult",
    "child",
    "accounts",
    "benefits",
    "job",
    "oddjob",
    "self.benunit",
    "househol",
    "chldcare",
    "pension",
    "maint",
    "mortgage",
    "penprov",
)


class FRS(BaseModel):
    adult: pd.DataFrame
    child: pd.DataFrame
    accounts: pd.DataFrame
    benefits: pd.DataFrame
    job: pd.DataFrame
    oddjob: pd.DataFrame
    benunit: pd.DataFrame
    househol: pd.DataFrame
    chldcare: pd.DataFrame
    pension: pd.DataFrame
    maint: pd.DataFrame
    mortgage: pd.DataFrame
    penprov: pd.DataFrame

    class Config:
        arbitrary_types_allowed = True


def load_frs_tables(
    ukda_tab_folder: str | Path,
):
    tables = {}

    for table_name in FRS_TABLE_NAMES:
        tables[table_name] = pd.read_csv(
            Path(ukda_tab_folder) / f"{table_name}.tab",
            low_memory=False,
            delimiter="\t",
        ).apply(pd.to_numeric, errors="coerce")
        tables[table_name].columns = tables[table_name].columns.str.upper()

    return FRS(
        adult=tables["adult"],
        child=tables["child"],
        accounts=tables["accounts"],
        benefits=tables["benefits"],
        job=tables["job"],
        oddjob=tables["oddjob"],
        benunit=tables["self.benunit"],
        househol=tables["househol"],
        chldcare=tables["chldcare"],
        pension=tables["pension"],
        maint=tables["maint"],
        mortgage=tables["mortgage"],
        penprov=tables["penprov"],
    )

def concat(*args):
    """
    Concatenate the given arrays along the first axis.
    """
    return np.concatenate(args, axis=0)


class PolicyEngineselfFRSDataset:
    frs: FRS
    person: pd.DataFrame
    benunit: pd.DataFrame
    household: pd.DataFrame
    state: pd.DataFrame

    def __init__(self, year: int, tab_folder: str | Path):
        self.frs = load_frs_tables(
            "/Users/nikhilwoodruff/Downloads/UKDA-9252-tab/tab",
        )

        self.person, self.benunit, self.household, self.state = (
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
        )

        self.count_adults = len(self.frs.adult)
        self.count_children = len(self.frs.child)
        self.count_people = self.count_adults + self.count_children

        self.zero_for_children = np.zeros(self.count_children)
        self.false_for_children = np.zeros(self.count_children, dtype=bool)

    def build(self):
        # Add ID variables to original self.frs tables for convenience
        for table_name in FRS_TABLE_NAMES:
            table = getattr(self.frs, table_name)
            if "person" in table.columns:
                table["person_id"] = (
                    table["SERNUM"] * 1e2 + table.benunit * 1e1 + table.person
                ).astype(int)

            if "benunit" in table.columns:
                table["benunit_id"] = (
                    table["SERNUM"] * 1e2 + table.benunit * 1e1
                ).astype(int)

            if "SERNUM" in table.columns:
                table["household_id"] = (table["SERNUM"] * 1e2).astype(int)
        # Add primary and foreign keys
        self.person["person_id"] = self._value_from_adult_child_tables(
            column="person_id"
        )
        self.person["person_benunit_id"] = self._value_from_adult_child_tables(
            column="benunit_id"
        )
        self.person["person_household_id"] = self._value_from_adult_child_tables(
            column="household_id"
        )
        self.person["person_state_id"] = np.array([1] * self.count_people)
        self.benunit["benunit_id"] = self.frs.benunit.benunit_id
        self.household["household_id"] = self.frs.househol.household_id
        self.state["state_id"] = np.array([1])

        # Add grossing weights
        self.household["self.household_weight"] = self.frs.househol.GROSS4

        # Add basic self.personal variables
        self.person["age"] = self._value_from_adult_child_tables(
            adult_column="AGE80",
            child_column="AGE",
        )
        self.person["birth_year"] = np.ones_like(self.person.age) * (
            self.year - self.person.age
        )
        # Age fields are AGE80 (top-coded) and AGE in the adult and 
        # child tables, respectively.
        self.person["gender"] = np.where(
            self._value_from_adult_child_tables("SEX") == 1, "MALE", "FEMALE"
        ).astype("S")
        self.person["hours_worked"] = (
            self._value_from_adult_child_tables("TOTHOURS").fillna(0).clip(lower=0) * 52
        )
        self.person["is_self.household_head"] = concat(
            self.frs.adult.HRPID == 1, self.false_for_children
        )
        self.person["is_self.benunit_head"] = concat(
            self.frs.adult.UPERSON == 1,
            self.false_for_children,
        )
        self.person["marital_status"] = self._value_from_adult_child_tables(
            column="MARITAL",
            default_value=2,
        ).map(
            {
                1: "MARRIED",
                2: "SINGLE",
                3: "SINGLE",
                4: "WIDOWED",
                5: "SEPARATED",
                6: "DIVORCED",
            }
        )

        if "FTED" in self.frs.adult.columns:
            fted = self._value_from_adult_child_tables(
                column="FTED",
            )
        else:
            fted = self._value_from_adult_child_tables(
                column="EDUCFT",
            )
        typeed2 = self._value_from_adult_child_tables(
            column="TYPEED2",
        )
        age = self.person.age
        self.person["current_education"] = np.select(
            [
                fted.isin((2, -1, 0)),  # By default, not in education
                typeed2 == 1,  # In pre-primary
                typeed2.isin((2, 4))  # In primary, or...
                | (
                    typeed2.isin((3, 8)) & (age < 11)
                )  # special or private education (and under 11), or...
                | (
                    (typeed2 == 0) & (fted == 1) & (age > 5) & (age < 11)
                ),  # not given, full-time and between 5 and 11
                typeed2.isin((5, 6))  # In secondary, or...
                | (
                    typeed2.isin((3, 8)) & (age >= 11) & (age <= 16)
                )  # special/private and meets age criteria, or...
                | (
                    (typeed2 == 0) & (fted == 1) & (age <= 16)
                ),  # not given, full-time and under 17
                typeed2  # Non-advanced further education, or...
                == 7
                | (
                    typeed2.isin((3, 8)) & (age > 16)
                )  # special/private and meets age criteria, or...
                | (
                    (typeed2 == 0) & (fted == 1) & (age > 16)
                ),  # not given, full-time and over 16
                typeed2.isin((7, 8)) & (age >= 19),  # In post-secondary
                (typeed2 == 9)
                | (
                    (typeed2 == 0) & (fted == 1) & (age >= 19)
                ),  # In tertiary, or meets age condition
            ],
            [
                "NOT_IN_EDUCATION",
                "PRE_PRIMARY",
                "PRIMARY",
                "LOWER_SECONDARY",
                "UPPER_SECONDARY",
                "POST_SECONDARY",
                "TERTIARY",
            ],
            default="NOT_IN_EDUCATION",
        )

        # Add employment status
        self.person["employment_status"] = self._value_from_adult_child_tables(
            column="EMPSTATI",
        ).map(
            {
                0: "CHILD",
                1: "FT_EMPLOYED",
                2: "PT_EMPLOYED",
                3: "FT_SELF_EMPLOYED",
                4: "PT_SELF_EMPLOYED",
                5: "UNEMPLOYED",
                6: "RETIRED",
                7: "STUDENT",
                8: "CARER",
                9: "LONG_TERM_DISABLED",
                10: "SHORT_TERM_DISABLED",
            }
        )

        self.household["region"] = self.frs.househol.GVTREGNO.map(
            {
                1: "NORTH_EAST",
                2: "NORTH_WEST",
                4: "YORKSHIRE",
                5: "EAST_MIDLANDS",
                6: "WEST_MIDLANDS",
                7: "EAST_OF_ENGLAND",
                8: "LONDON",
                9: "SOUTH_EAST",
                10: "SOUTH_WEST",
                11: "WALES",
                12: "SCOTLAND",
                13: "NORTHERN_IRELAND",
            }
        )

        self.household["tenure_type"] = self.frs.househol.PTENTYP2.map(
            {
                1: "RENT_FROM_COUNCIL",
                2: "RENT_FROM_HA",
                3: "RENT_PRIVATELY",
                4: "RENT_PRIVATELY",
                5: "OWNED_OUTRIGHT",
                6: "OWNED_WITH_MORTGAGE",
            }
        )

        self.household["num_bedrooms"] = self.frs.househol.BEDROOM6

        self.household["council_tax"] = self.frs.househol.CTANNUAL
        self.household["council_tax_band"] = self.frs.househol.CTBAND

        # Fill in missing Council Tax bands and values using QRF

        council_tax_model = QRF()

        imputation_source = self.household.council_tax.notna() * (
            self.household.region != "NORTHERN_IRELAND"
        )
        needs_imputation = self.household.council_tax.isna() * (
            self.household.region != "NORTHERN_IRELAND"
        )

        council_tax_model.fit(
            X=self.household[imputation_source][["num_bedrooms", "region"]],
            y=self.household[imputation_source][["council_tax_band", "council_tax"]],
        )

        self.household.loc[needs_imputation, "council_tax"] = council_tax_model.predict(
            X=self.household[needs_imputation][["num_bedrooms", "region"]],
        )
        self.household["council_tax_band"] = self.household.council_tax_band.map(
            {
                1: "A",
                2: "B",
                3: "C",
                4: "D",
                5: "E",
                6: "F",
                7: "G",
                8: "H",
                9: "I",
            }
        )

        # Domestic rates variables are all weeklyised, unlike Council Tax variables
        # (despite the variable name suggesting otherwise)
        domestic_rates_variable = "RTANNUAL" if self.year < 2021 else "NIRATLIA"
        self.household["domestic_rates"] = (
            np.select(
                [
                    self.frs.househol[domestic_rates_variable] >= 0,
                    self.frs.househol.RT2REBAM >= 0,
                    True,
                ],
                [
                    self.frs.househol[domestic_rates_variable],
                    self.frs.househol.RT2REBAM,
                    0,
                ],
            )
            * 52
        )

    def _value_from_adult_child_tables(
        self,
        column: str = None,
        adult_column: str = None,
        child_column: str = None,
        default_value=0,
    ) -> pd.Series:
        """
        This function takes a column name and returns the values from the adult
        and child tables. If the column is not present in either table, it will
        return NaN for that table.
        """
        if adult_column is None:
            adult_column = column
        if child_column is None:
            child_column = column

        adult_values = self.frs.adult.get(
            adult_column, np.ones(self.count_adults) * default_value
        )
        child_values = self.frs.child.get(
            child_column, np.ones(self.count_children) * default_value
        )

        return pd.Series(
            np.concatenate(
                [
                    adult_values,
                    child_values,
                ]
            )
        )
