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

# Placeholder uprating factors from FYE 2020 to 2025 (to be updated)
UPRATING_TOTAL_INCOME_2020_TO_2025 = 1.3  # TODO: use OBR RHDI
UPRATING_NET_INCOME_BHC_2020_TO_2025 = 1.3  # TODO: use OBR RHDI
UPRATING_NET_INCOME_AHC_2020_TO_2025 = 1.3  # TODO: use OBR RHDI / house prices


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
    household_market_income = sim.calculate("household_market_income").values
    household_benefits = sim.calculate("household_benefits").values
    hbai_net_income = sim.calculate("hbai_household_net_income").values
    hbai_net_income_ahc = sim.calculate("hbai_household_net_income_ahc").values

    # PE total income = market income + benefits (to match ONS total income)
    pe_total_income = household_market_income + household_benefits

    # Add to matrix (household-level values, will be summed with weights)
    matrix["ons/total_income"] = pe_total_income
    matrix["ons/net_income_bhc"] = hbai_net_income
    matrix["ons/net_income_ahc"] = hbai_net_income_ahc

    # Calculate LA-level targets: mean income * households, uprated to 2025
    # For LAs without ONS data (Scotland, NI, newer merged LAs), set target to 0
    ons_merged["total_income_target"] = (
        ons_merged["total_income"].fillna(0)
        * ons_merged["households_2025"].fillna(0)
        * UPRATING_TOTAL_INCOME_2020_TO_2025
    )
    ons_merged["net_income_bhc_target"] = (
        ons_merged["net_income_bhc"].fillna(0)
        * ons_merged["households_2025"].fillna(0)
        * UPRATING_NET_INCOME_BHC_2020_TO_2025
    )
    ons_merged["net_income_ahc_target"] = (
        ons_merged["net_income_ahc"].fillna(0)
        * ons_merged["households_2025"].fillna(0)
        * UPRATING_NET_INCOME_AHC_2020_TO_2025
    )

    y["ons/total_income"] = ons_merged["total_income_target"].values
    y["ons/net_income_bhc"] = ons_merged["net_income_bhc_target"].values
    y["ons/net_income_ahc"] = ons_merged["net_income_ahc_target"].values

    country_mask = create_country_mask(
        household_countries=sim.calculate("country").values,
        codes=la_codes.code,
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
