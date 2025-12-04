from policyengine_uk import Microsimulation
import pandas as pd
import numpy as np
from pathlib import Path

from policyengine_uk_data.utils.loss import (
    create_target_matrix as create_national_target_matrix,
)
from policyengine_uk_data.storage import STORAGE_FOLDER
from policyengine_uk.data import UKSingleYearDataset
from policyengine_uk_data.utils.uc_data import uc_la_households

FOLDER = Path(__file__).parent

# Uprating factors from FYE 2020 to 2025 (OBR Nov 2025 EFO)
# RHDI index: 1985.1 (2025-26) / 1467.6 (2020-21) = 1.352
UPRATING_NET_INCOME_BHC_2020_TO_2025 = 1985.1 / 1467.6
# House price index: 103.5 (2025-26) / 84.9 (2020-21) = 1.219
UPRATING_HOUSING_COSTS_2020_TO_2025 = 103.5 / 84.9


def load_ons_la_income_targets() -> pd.DataFrame:
    """Load ONS income estimates by local authority.

    Returns a DataFrame with columns: la_code, total_income, net_income_bhc, net_income_ahc
    (mean income per household, FYE 2020)
    """
    xlsx = pd.ExcelFile(STORAGE_FOLDER / "local_authority_ons_income.xlsx")

    def load_sheet(sheet_name: str, value_col: str) -> pd.DataFrame:
        df = pd.read_excel(xlsx, sheet_name=sheet_name, header=3)
        df.columns = [
            "msoa_code", "msoa_name", "la_code", "la_name",
            "region_code", "region_name", value_col,
            "upper_ci", "lower_ci", "ci_width"
        ]
        df = df.iloc[1:].dropna(subset=["msoa_code"])
        df[value_col] = pd.to_numeric(df[value_col])
        return df[["la_code", value_col]]

    total = load_sheet("Total annual income", "total_income")
    bhc = load_sheet("Net income before housing costs", "net_income_bhc")
    ahc = load_sheet("Net income after housing costs", "net_income_ahc")

    # Group by LA to get mean income per household
    la_total = total.groupby("la_code")["total_income"].mean().reset_index()
    la_bhc = bhc.groupby("la_code")["net_income_bhc"].mean().reset_index()
    la_ahc = ahc.groupby("la_code")["net_income_ahc"].mean().reset_index()

    return la_total.merge(la_bhc, on="la_code").merge(la_ahc, on="la_code")


def create_local_authority_target_matrix(
    dataset: UKSingleYearDataset,
    time_period: int = None,
    reform=None,
):
    if time_period is None:
        time_period = dataset.time_period
    ages = pd.read_csv(FOLDER / "targets" / "age.csv")
    incomes = pd.read_csv(FOLDER / "targets" / "spi_by_la.csv")
    la_codes = pd.read_csv(STORAGE_FOLDER / "local_authorities_2021.csv")

    sim = Microsimulation(dataset=dataset, reform=reform)
    original_weights = sim.calculate("household_weight", 2025).values
    sim.default_calculation_period = time_period

    matrix = pd.DataFrame()
    y = pd.DataFrame()

    INCOME_VARIABLES = [
        "self_employment_income",
        "employment_income",
    ]

    national_incomes = pd.read_csv(STORAGE_FOLDER / "incomes_projection.csv")
    national_incomes = national_incomes[
        national_incomes.year
        == max(national_incomes.year.min(), int(dataset.time_period))
    ]

    for income_variable in INCOME_VARIABLES:
        income_values = sim.calculate(income_variable).values
        in_spi_frame = sim.calculate("income_tax").values > 0
        matrix[f"hmrc/{income_variable}/amount"] = sim.map_result(
            income_values * in_spi_frame, "person", "household"
        )
        local_targets = incomes[f"{income_variable}_amount"].values
        local_target_sum = local_targets.sum()
        national_target = national_incomes[
            (national_incomes.total_income_lower_bound == 12_570)
            & (national_incomes.total_income_upper_bound == np.inf)
        ][income_variable + "_amount"].iloc[0]
        national_consistency_adjustment_factor = (
            national_target / local_target_sum
        )
        y[f"hmrc/{income_variable}/amount"] = (
            local_targets * national_consistency_adjustment_factor
        )
        matrix[f"hmrc/{income_variable}/count"] = sim.map_result(
            (income_values != 0) * in_spi_frame, "person", "household"
        )
        local_targets = incomes[f"{income_variable}_count"].values
        local_target_sum = local_targets.sum()
        national_target = national_incomes[
            (national_incomes.total_income_lower_bound == 12_570)
            & (national_incomes.total_income_upper_bound == np.inf)
        ][income_variable + "_count"].iloc[0]
        y[f"hmrc/{income_variable}/count"] = (
            incomes[f"{income_variable}_count"].values
            * national_consistency_adjustment_factor
        )

    age = sim.calculate("age").values
    national_demographics = pd.read_csv(STORAGE_FOLDER / "demographics.csv")
    uk_total_population = (
        national_demographics[national_demographics.name == "uk_population"][
            str(time_period)
        ].values[0]
        * 1e6
    )

    age = sim.calculate("age").values
    targets_total_pop = 0
    for lower_age in range(0, 80, 10):
        upper_age = lower_age + 10

        in_age_band = (age >= lower_age) & (age < upper_age)

        age_str = f"{lower_age}_{upper_age}"
        matrix[f"age/{age_str}"] = sim.map_result(
            in_age_band, "person", "household"
        )

        age_count = ages[
            [str(age) for age in range(lower_age, upper_age)]
        ].sum(axis=1)

        age_str = f"{lower_age}_{upper_age}"
        y[f"age/{age_str}"] = age_count.values
        targets_total_pop += age_count.values.sum()

    # Adjust for consistency
    for lower_age in range(0, 80, 10):
        upper_age = lower_age + 10

        in_age_band = (age >= lower_age) & (age < upper_age)

        age_str = f"{lower_age}_{upper_age}"
        y[f"age/{age_str}"] *= uk_total_population / targets_total_pop * 0.9

    # UC household count by local authority
    y["uc_households"] = uc_la_households.household_count.values
    matrix["uc_households"] = sim.map_result(
        (sim.calculate("universal_credit").values > 0).astype(int),
        "benunit",
        "household",
    )

    # ONS income targets by local authority
    # ONS definitions:
    #   total_income (ONS) = household_market_income + household_benefits (PE)
    #   net_income_bhc (ONS) = hbai_household_net_income (PE)
    #   net_income_ahc (ONS) = hbai_household_net_income_ahc (PE)
    ons_income = load_ons_la_income_targets()
    households_by_la = pd.read_csv(STORAGE_FOLDER / "households_by_la_2025.csv")

    # Merge ONS income with our LA codes to get targets aligned
    ons_merged = la_codes.merge(
        ons_income, left_on="code", right_on="la_code", how="left"
    ).merge(
        households_by_la[["code", "households_2025"]],
        on="code",
        how="left"
    )

    # Calculate PE household income variables
    hbai_net_income = sim.calculate("hbai_household_net_income").values
    hbai_net_income_ahc = sim.calculate("hbai_household_net_income_ahc").values
    housing_costs = hbai_net_income_ahc - hbai_net_income

    # Add to matrix (household-level values, will be summed with weights)
    matrix["ons/net_income_bhc"] = hbai_net_income
    matrix["ons/net_income_ahc"] = hbai_net_income_ahc
    matrix["ons/housing_costs"] = housing_costs

    # Calculate LA-level targets: mean income * households, uprated to 2025
    ons_merged["net_income_bhc_target"] = (
        ons_merged["net_income_bhc"]
        * ons_merged["households_2025"]
        * UPRATING_NET_INCOME_BHC_2020_TO_2025
    )
    ons_merged["housing_costs_target"] = (
        ons_merged["net_income_bhc_target"]
        - ons_merged["net_income_ahc_target"]
    ) * UPRATING_HOUSING_COSTS_2020_TO_2025
    ons_merged["net_income_ahc_target"] = (
        ons_merged["net_income_bhc_target"]
        - ons_merged["housing_costs_target"]
    )

    country_mask = create_country_mask(
        household_countries=sim.calculate("country").values,
        codes=la_codes.code,
    )

    # For LAs without ONS data (Scotland, NI, newer merged LAs), use model
    # values at initial weights as targets (so calibration doesn't change them)
    has_ons_data = ons_merged["net_income_bhc"].notna().values
    initial_la_weights = (original_weights / 360) * country_mask
    model_net_income_bhc = initial_la_weights @ hbai_net_income
    model_net_income_ahc = initial_la_weights @ hbai_net_income_ahc
    model_housing_costs = initial_la_weights @ housing_costs

    y["ons/net_income_bhc"] = np.where(
        has_ons_data,
        ons_merged["net_income_bhc_target"].values,
        model_net_income_bhc,
    )
    y["ons/net_income_ahc"] = np.where(
        has_ons_data,
        ons_merged["net_income_ahc_target"].values,
        model_net_income_ahc,
    )
    y["ons/housing_costs"] = np.where(
        has_ons_data,
        ons_merged["housing_costs_target"].values,
        model_housing_costs,
    )

    return matrix, y, country_mask


def create_country_mask(
    household_countries: np.ndarray, codes: pd.Series
) -> np.ndarray:
    # Create a matrix R to accompany the loss matrix M s.t. (W x M) x R = Y_
    # where Y_ is the target matrix for the country where no target is constructed from weights from a different country.

    constituency_countries = codes.apply(lambda code: code[0]).map(
        {
            "E": "ENGLAND",
            "W": "WALES",
            "S": "SCOTLAND",
            "N": "NORTHERN_IRELAND",
        }
    )

    r = np.zeros((len(codes), len(household_countries)))

    for i in range(len(codes)):
        r[i] = household_countries == constituency_countries[i]

    return r
