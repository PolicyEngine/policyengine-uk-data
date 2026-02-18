"""Local authority extra targets: ONS income, tenure, private rent.

These targets are only available at LA level (not constituency).

Sources:
- ONS small area income estimates:
  https://www.ons.gov.uk/employmentandlabourmarket/peopleinwork/earningsandworkinghours/datasets/smallareaincomeestimatesformiddlelayersuperoutputareasenglandandwales
- English Housing Survey tenure:
  https://www.gov.uk/government/statistics/english-housing-survey-2023
- VOA private rental market statistics:
  https://www.ons.gov.uk/peoplepopulationandcommunity/housing/datasets/privaterentalmarketsummarystatisticsinengland
"""

import logging

import pandas as pd

from policyengine_uk_data.targets.sources._common import STORAGE

logger = logging.getLogger(__name__)

# Uprating factors from FYE 2020 to 2025 (OBR Nov 2025 EFO)
UPRATING_NET_INCOME_BHC_2020_TO_2025 = 1985.1 / 1467.6
UPRATING_HOUSING_COSTS_2020_TO_2025 = 103.5 / 84.9

_REF_ONS_INCOME = (
    "https://www.ons.gov.uk/employmentandlabourmarket/peopleinwork/"
    "earningsandworkinghours/datasets/"
    "smallareaincomeestimatesformiddlelayersuperoutputareasenglandandwales"
)
_REF_TENURE = (
    "https://www.gov.uk/government/statistics/english-housing-survey-2023"
)
_REF_RENT = (
    "https://www.ons.gov.uk/peoplepopulationandcommunity/housing/datasets/"
    "privaterentalmarketsummarystatisticsinengland"
)


def load_ons_la_income() -> pd.DataFrame:
    """Load ONS income estimates by local authority.

    Returns DataFrame with columns: la_code, total_income, net_income_bhc,
    net_income_ahc (mean income per household, FYE 2020).
    """
    xlsx_path = STORAGE / "local_authority_ons_income.xlsx"
    if not xlsx_path.exists():
        logger.warning("ONS LA income file not found: %s", xlsx_path)
        return pd.DataFrame()

    xlsx = pd.ExcelFile(xlsx_path)

    def load_sheet(sheet_name: str, value_col: str) -> pd.DataFrame:
        df = pd.read_excel(xlsx, sheet_name=sheet_name, header=3)
        df.columns = [
            "msoa_code",
            "msoa_name",
            "la_code",
            "la_name",
            "region_code",
            "region_name",
            value_col,
            "upper_ci",
            "lower_ci",
            "ci_width",
        ]
        df = df.iloc[1:].dropna(subset=["msoa_code"])
        df[value_col] = pd.to_numeric(df[value_col])
        return df[["la_code", value_col]]

    total = load_sheet("Total annual income", "total_income")
    bhc = load_sheet("Net income before housing costs", "net_income_bhc")
    ahc = load_sheet("Net income after housing costs", "net_income_ahc")

    la_total = total.groupby("la_code")["total_income"].mean().reset_index()
    la_bhc = bhc.groupby("la_code")["net_income_bhc"].mean().reset_index()
    la_ahc = ahc.groupby("la_code")["net_income_ahc"].mean().reset_index()

    return la_total.merge(la_bhc, on="la_code").merge(la_ahc, on="la_code")


def load_household_counts() -> pd.DataFrame:
    """Load household counts by LA (Census 2021).

    Returns DataFrame with columns: la_code, households.
    """
    path = STORAGE / "la_count_households.xlsx"
    if not path.exists():
        logger.warning("LA household count file not found: %s", path)
        return pd.DataFrame()
    df = pd.read_excel(path, sheet_name="Dataset")
    df.columns = ["la_code", "la_name", "households"]
    return df[["la_code", "households"]]


def load_tenure_data() -> pd.DataFrame:
    """Load tenure percentages by LA.

    Returns DataFrame with columns: la_code, owned_outright_pct,
    owned_mortgage_pct, private_rent_pct, social_rent_pct.
    """
    path = STORAGE / "la_tenure.xlsx"
    if not path.exists():
        logger.warning("LA tenure file not found: %s", path)
        return pd.DataFrame()
    df = pd.read_excel(path, sheet_name="data download")
    df.columns = [
        "region_code",
        "region_name",
        "la_code",
        "la_name",
        "owned_outright_pct",
        "owned_mortgage_pct",
        "private_rent_pct",
        "social_rent_pct",
    ]
    return df[
        [
            "la_code",
            "owned_outright_pct",
            "owned_mortgage_pct",
            "private_rent_pct",
            "social_rent_pct",
        ]
    ]


def load_private_rents() -> pd.DataFrame:
    """Load median monthly private rents by LA.

    Returns DataFrame with columns: area_code, median_annual_rent.
    """
    path = STORAGE / "la_private_rents_median.xlsx"
    if not path.exists():
        logger.warning("LA private rent file not found: %s", path)
        return pd.DataFrame()
    df = pd.read_excel(path, sheet_name="Figure 3", header=5)
    df.columns = [
        "col0",
        "la_code_old",
        "area_code",
        "area_name",
        "room",
        "studio",
        "one_bed",
        "two_bed",
        "three_bed",
        "four_plus",
        "median_monthly_rent",
    ]
    df = df[df["area_code"].astype(str).str.match(r"^E0[6789]")]
    df["median_monthly_rent"] = pd.to_numeric(
        df["median_monthly_rent"], errors="coerce"
    )
    df["median_annual_rent"] = df["median_monthly_rent"] * 12
    return df[["area_code", "median_annual_rent"]]
