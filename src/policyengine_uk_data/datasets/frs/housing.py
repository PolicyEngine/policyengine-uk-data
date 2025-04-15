import numpy as np
import pandas as pd
from typing import Tuple

from policyengine_uk_data.impute import QRF
from .ukda import UKDA_FRS


def add_housing(
    person: pd.DataFrame,
    benunit: pd.DataFrame,
    household: pd.DataFrame,
    state: pd.DataFrame,
    frs: UKDA_FRS,
    year: int
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Add housing-related variables to the household dataframe.
    
    Args:
        person (pd.DataFrame): Person-level dataframe.
        benunit (pd.DataFrame): Benefit unit-level dataframe.
        household (pd.DataFrame): Household-level dataframe.
        state (pd.DataFrame): State-level dataframe.
        frs (FRS): The FRS data object.
        year (int): The year of the data.
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]: Updated dataframes.
    """
    household["region"] = frs.househol.GVTREGNO.map(
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

    household["tenure_type"] = frs.househol.PTENTYP2.map(
        {
            1: "RENT_FROM_COUNCIL",
            2: "RENT_FROM_HA",
            3: "RENT_PRIVATELY",
            4: "RENT_PRIVATELY",
            5: "OWNED_OUTRIGHT",
            6: "OWNED_WITH_MORTGAGE",
        }
    )

    household["num_bedrooms"] = frs.househol.BEDROOM6

    household["council_tax"] = frs.househol.CTANNUAL
    household["council_tax_band"] = frs.househol.CTBAND

    # Fill in missing Council Tax bands and values using QRF

    council_tax_model = QRF()

    imputation_source = household.council_tax.notna() * (
        household.region != "NORTHERN_IRELAND"
    )
    needs_imputation = household.council_tax.isna() * (
        household.region != "NORTHERN_IRELAND"
    )

    council_tax_model.fit(
        X=household[imputation_source][["num_bedrooms", "region"]],
        y=household[imputation_source][["council_tax_band", "council_tax"]],
    )

    household.loc[needs_imputation, "council_tax"] = council_tax_model.predict(
        X=household[needs_imputation][["num_bedrooms", "region"]],
    )
    household.council_tax = household.council_tax.fillna(0)
    household["council_tax_band"] = household.council_tax_band.map(
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
    domestic_rates_variable = "RTANNUAL" if year < 2021 else "NIRATLIA"
    household["domestic_rates"] = (
        np.select(
            [
                frs.househol[domestic_rates_variable] >= 0,
                frs.househol.RT2REBAM >= 0,
                True,
            ],
            [
                frs.househol[domestic_rates_variable],
                frs.househol.RT2REBAM,
                0,
            ],
        )
        * 52
    )

    household["main_residential_property_purchased_is_first_home"] = (
        np.random.random() < 0.2
    )
    household["household_owns_tv"] = np.random.random() < 0.96

    return person, benunit, household, state
