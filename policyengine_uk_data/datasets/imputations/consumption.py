"""
Consumption imputation using Living Costs and Food Survey data.

This module imputes household consumption patterns (including energy spending)
using QRF models trained on LCFS data, with vehicle ownership from the Wealth
and Assets Survey for fuel spending and housing characteristics for energy.

Key features:
- Gas and electricity are imputed separately using LCFS interview variables
  (B226=electricity DD, B489=electricity PPM, B490=gas PPM)
  rather than just the aggregate P537 domestic-energy total.
- Housing predictors (tenure_type, accommodation_type) added alongside income
  and demographics, matching the strong drivers in NEED admin data.
- Imputed totals are calibrated to NEED 2023 mean kWh targets by income band,
  converted to spend using Ofgem Q2 2026 unit rates (Apr 2026 price cap) so the
  stored £/yr values represent FY26/27 price levels. Energy variables are not
  CPI-uprated, so this avoids the need for price-level adjustment at simulation time.
  NEED income bands use Experian modelled gross household income, so calibration
  matches against gross income (LCFS P344p / FRS household_gross_income) rather
  than HBAI net income.
"""

import pandas as pd
import numpy as np
from policyengine_uk_data.datasets.frs_release import CURRENT_FRS_RELEASE
from policyengine_uk_data.datasets.private_releases import (
    CURRENT_LCFS_RELEASE,
    CURRENT_WAS_RELEASE,
)
from policyengine_uk_data.storage import STORAGE_FOLDER
from policyengine_uk.data import UKSingleYearDataset
from policyengine_uk import Microsimulation
from policyengine_uk_data.datasets.frs import WEEKS_IN_YEAR

LCFS_TAB_FOLDER = STORAGE_FOLDER / CURRENT_LCFS_RELEASE.name

# Default seed for the stochastic ICE-vehicle flag drawn from
# `NTS_2024_ICE_VEHICLE_SHARE`. Kept at 42 for backward compatibility with
# existing artefact fingerprints; callers can override via the fixture's
# local RNG rather than the process-wide np.random global.
_HAS_FUEL_SEED = 42

# EV/ICE vehicle mix from NTS 2024
NTS_2024_ICE_VEHICLE_SHARE = 0.90

# DESNZ weekly road-fuel price statistics, fiscal-year average UK pump prices.
# 2023 prices cover 2023-04-01 to 2024-03-31 for the current LCFS release.
# Data source:
# https://www.data.gov.uk/dataset/21db6396-3daf-4d90-8b3f-054995256018/petrol-and-diesel-prices
# LCFS records nominal fuel spending, while PolicyEngine derives litres via
# ``spending / model pump price``.
LCFS_FUEL_PRICE_GBP_PER_LITRE = {
    "petrol_spending": {
        2021: 1.3890790089424998,
        2023: 1.4615903846153844,
    },
    "diesel_spending": {
        2021: 1.4291180616502566,
        2023: 1.5348538461538461,
    },
}
FUEL_PRICE_PARAMETER_NAME = {
    "petrol_spending": "petrol",
    "diesel_spending": "diesel",
}
CONSUMPTION_MODEL_FILENAME = (
    f"consumption_{CURRENT_LCFS_RELEASE.name}_{CURRENT_WAS_RELEASE.name}"
    "_fuel_litre_proxy_2026_05.pkl"
)
HAS_FUEL_MODEL_FILENAME = f"has_fuel_{CURRENT_WAS_RELEASE.name}.pkl"

REGIONS = {
    1: "NORTH_EAST",
    2: "NORTH_WEST",
    3: "YORKSHIRE",
    4: "EAST_MIDLANDS",
    5: "WEST_MIDLANDS",
    6: "EAST_OF_ENGLAND",
    7: "LONDON",
    8: "SOUTH_EAST",
    9: "SOUTH_WEST",
    10: "WALES",
    11: "SCOTLAND",
    12: "NORTHERN_IRELAND",
}

# LCFS A122 → FRS tenure_type mapping
# LCFS: 1=council, 2=HA, 3=private rent, 4=rent-free, 5=owned w mortgage,
#        6=shared ownership, 7=owned outright, 8=other
LCFS_TENURE_MAP = {
    1: "RENT_FROM_COUNCIL",
    2: "RENT_FROM_HA",
    3: "RENT_PRIVATELY",
    4: "RENT_PRIVATELY",  # rent-free → private for simplicity
    5: "OWNED_WITH_MORTGAGE",
    6: "OWNED_WITH_MORTGAGE",  # shared ownership
    7: "OWNED_OUTRIGHT",
    8: "RENT_PRIVATELY",
}

# LCFS A121 → FRS accommodation_type mapping
# LCFS coding inferred from the LCFS user guide:
# 1=detached house, 2=semi-detached, 3=terraced, 4=flat (purpose-built),
# 5=flat/other (converted), 6=caravan/mobile, 7=bungalow/other house, 8=other
LCFS_ACCOMM_MAP = {
    1: "HOUSE_DETACHED",
    2: "HOUSE_SEMI_DETACHED",
    3: "HOUSE_TERRACED",
    4: "FLAT",
    5: "FLAT",
    6: "MOBILE",
    7: "HOUSE_DETACHED",
    8: "OTHER",
}

HOUSEHOLD_LCF_RENAMES = {
    "g018": "is_adult",
    "g019": "is_child",
    "gorx": "region",
    "p389p": "hbai_household_net_income",
    "p344p": "household_gross_income",
    "weighta": "household_weight",
}
PERSON_LCF_RENAMES = {
    "b303p": "employment_income",
    "b3262p": "self_employment_income",
    "b3381": "state_pension",
    "p049p": "private_pension_income",
}

CONSUMPTION_VARIABLE_RENAMES = {
    "p601": "food_and_non_alcoholic_beverages_consumption",
    "p602": "alcohol_and_tobacco_consumption",
    "p603": "clothing_and_footwear_consumption",
    "p604": "housing_water_and_electricity_consumption",
    "p605": "household_furnishings_consumption",
    "p606": "health_consumption",
    "p607": "transport_consumption",
    "p608": "communication_consumption",
    "p609": "recreation_consumption",
    "p610": "education_consumption",
    "p611": "restaurants_and_hotels_consumption",
    "p612": "miscellaneous_consumption",
    "c72211": "petrol_spending",
    "c72212": "diesel_spending",
    "p537": "domestic_energy_consumption",  # aggregate kept for backward compat
}

# LCFS detailed COICOP codes for passenger transport by road (7.3.2): bus and
# coach fares. There is no single P-code for bus fares alone — P607
# (transport_consumption) bundles vehicle purchase, running costs, fuel, air and
# rail — so bus_fare_spending is summed from the detailed 7.3.2 codes. Excludes
# rail (7.3.1, c731xx), air (7.3.3), combined tickets (7.3.5) and taxis (7.3.6).
# Codes verified present in the LCFS 2021/22 dvhh file; see the LCFS data
# dictionary for sub-code definitions.
BUS_FARE_LCFS_CODES = ["c73212", "c73213", "c73214"]

PREDICTOR_VARIABLES = [
    "is_adult",
    "is_child",
    "region",
    "employment_income",
    "self_employment_income",
    "private_pension_income",
    "hbai_household_net_income",
    "tenure_type",
    "accommodation_type",
    "has_fuel_consumption",
]

IMPUTATIONS = [
    "food_and_non_alcoholic_beverages_consumption",
    "alcohol_and_tobacco_consumption",
    "clothing_and_footwear_consumption",
    "housing_water_and_electricity_consumption",
    "household_furnishings_consumption",
    "health_consumption",
    "transport_consumption",
    "communication_consumption",
    "recreation_consumption",
    "education_consumption",
    "restaurants_and_hotels_consumption",
    "miscellaneous_consumption",
    "petrol_spending",
    "diesel_spending",
    "bus_fare_spending",  # COICOP 7.3.2 bus & coach fares (see BUS_FARE_LCFS_CODES)
    "domestic_energy_consumption",  # aggregate; backward compat with price cap subsidy
    "electricity_consumption",
    "gas_consumption",
]

HAS_FUEL_PREDICTOR_VARIABLES = [
    "household_net_income",
    "num_adults",
    "num_children",
    "private_pension_income",
    "employment_income",
    "self_employment_income",
    "region",
]


def get_has_fuel_model_path():
    return STORAGE_FOLDER / HAS_FUEL_MODEL_FILENAME


def get_has_fuel_model_metadata() -> dict:
    return {
        "was_release_name": CURRENT_WAS_RELEASE.name,
        "was_household_tab_filename": CURRENT_WAS_RELEASE.household_tab_filename,
        "predictor_variables": tuple(HAS_FUEL_PREDICTOR_VARIABLES),
        "impute_variables": ("has_fuel_consumption",),
        "ice_vehicle_share": NTS_2024_ICE_VEHICLE_SHARE,
        "seed": _HAS_FUEL_SEED,
    }


def get_consumption_model_path():
    return STORAGE_FOLDER / CONSUMPTION_MODEL_FILENAME


def get_consumption_model_metadata() -> dict:
    return {
        "lcfs_release_name": CURRENT_LCFS_RELEASE.name,
        "lcfs_household_tab_filename": CURRENT_LCFS_RELEASE.household_tab_filename,
        "lcfs_person_tab_filename": CURRENT_LCFS_RELEASE.person_tab_filename,
        "lcfs_fuel_price_year": CURRENT_LCFS_RELEASE.fuel_price_year,
        "was_release_name": CURRENT_WAS_RELEASE.name,
        "was_household_tab_filename": CURRENT_WAS_RELEASE.household_tab_filename,
        "frs_base_year": CURRENT_FRS_RELEASE.base_year,
        "predictor_variables": tuple(PREDICTOR_VARIABLES),
        "impute_variables": tuple(IMPUTATIONS),
        "domestic_energy_consumption_source": "calibrated_electricity_plus_gas",
    }


def _qrf_model_matches_current_metadata(
    model, metadata: dict, outputs: list[str]
) -> bool:
    if getattr(model, "metadata", {}) != metadata:
        return False

    trained_outputs = getattr(model.model, "imputed_variables", None)
    return list(trained_outputs) == outputs


# ── NEED 2023 calibration targets ─────────────────────────────────────────────
# Source: NEED 2023 headline tables (published 2025), England & Wales, ~18M dwellings.
# Tables 11b/12b: mean gas/electricity kWh by income; 9b/10b by tenure;
# 5b/6b by property type; 15b/16b by region; 13b/14b by number of adults.
# NEED kWh targets are physical consumption (stable across years). We convert to
# annual spend using Ofgem Q2 2026 unit rates so the raked values represent
# FY26/27 price levels. Energy variables are not CPI-uprated, so this ensures
# the stored £/yr values are correct for the target simulation year.
# Standing charges excluded to keep to unit-rate spend consistent with LCFS recording.
OFGEM_Q2_2026_ELEC_RATE = 24.67 / 100  # £/kWh (Apr 2026 price cap)
OFGEM_Q2_2026_GAS_RATE = 5.74 / 100  # £/kWh (Apr 2026 price cap)

# NEED 2023 mean kWh by income band (Table 11b gas, Table 12b electricity)
# Income bands are gross household income (Experian modelled data)
NEED_INCOME_BANDS = [
    (0, 15_000, "under_15k", 7_755, 2_412),  # gas kWh, elec kWh
    (15_000, 20_000, "15k_20k", 9_196, 2_700),
    (20_000, 30_000, "20k_30k", 9_886, 2_915),
    (30_000, 40_000, "30k_40k", 10_697, 3_114),
    (40_000, 50_000, "40k_50k", 11_230, 3_276),
    (50_000, 60_000, "50k_60k", 11_721, 3_410),
    (60_000, 70_000, "60k_70k", 12_200, 3_548),
    (70_000, 100_000, "70k_100k", 13_244, 3_872),
    (100_000, 150_000, "100k_150k", 15_727, 4_598),
    (150_000, np.inf, "over_150k", 20_359, 5_944),
]

# NEED 2023 mean kWh by tenure (Table 9b gas, Table 10b electricity)
# NEED tenures: Owner-occupied, Privately rented, Council/HA
# FRS tenure_type values: OWNED_OUTRIGHT, OWNED_WITH_MORTGAGE → owner-occupied
#                         RENT_PRIVATELY → private rented
#                         RENT_FROM_COUNCIL, RENT_FROM_HA → social
NEED_TENURE_GAS = {"owner": 12_339, "private_rent": 10_183, "social": 8_357}
NEED_TENURE_ELEC = {"owner": 3_465, "private_rent": 3_261, "social": 2_896}

TENURE_TO_NEED = {
    "OWNED_OUTRIGHT": "owner",
    "OWNED_WITH_MORTGAGE": "owner",
    "RENT_PRIVATELY": "private_rent",
    "RENT_FROM_COUNCIL": "social",
    "RENT_FROM_HA": "social",
}

# NEED 2023 mean kWh by property type (Table 5b gas, Table 6b electricity)
# NEED types: Detached, Semi-detached, End/Mid terrace, Bungalow, Converted flat, Purpose-built flat
# FRS accommodation_type: HOUSE_DETACHED, HOUSE_SEMI_DETACHED, HOUSE_TERRACED, FLAT, MOBILE, OTHER
NEED_ACCOMM_GAS = {
    "detached": 15_518,
    "semi": 11_715,
    "terraced": 10_365,
    "flat": 7_058,
    "other": 11_303,
}
NEED_ACCOMM_ELEC = {
    "detached": 4_346,
    "semi": 3_338,
    "terraced": 3_096,
    "flat": 2_896,
    "other": 3_327,
}

ACCOMM_TO_NEED = {
    "HOUSE_DETACHED": "detached",
    "HOUSE_SEMI_DETACHED": "semi",
    "HOUSE_TERRACED": "terraced",
    "FLAT": "flat",
    "MOBILE": "other",
    # OTHER is a small FRS catch-all with no clear NEED equivalent; excluded from raking
}

# NEED 2023 mean kWh by region (Table 15b gas, Table 16b electricity)
# Region strings match FRS/LCFS REGIONS dict values
NEED_REGION_GAS = {
    "NORTH_EAST": 11_278,
    "NORTH_WEST": 11_111,
    "YORKSHIRE": 11_552,
    "EAST_MIDLANDS": 11_234,
    "WEST_MIDLANDS": 11_485,
    "EAST_OF_ENGLAND": 11_334,
    "LONDON": 12_335,
    "SOUTH_EAST": 11_555,
    "SOUTH_WEST": 9_811,
    "WALES": 10_558,
}
NEED_REGION_ELEC = {
    "NORTH_EAST": 2_822,
    "NORTH_WEST": 3_211,
    "YORKSHIRE": 3_114,
    "EAST_MIDLANDS": 3_266,
    "WEST_MIDLANDS": 3_332,
    "EAST_OF_ENGLAND": 3_543,
    "LONDON": 3_275,
    "SOUTH_EAST": 3_568,
    "SOUTH_WEST": 3_537,
    "WALES": 3_151,
}

# Pre-computed NEED spend targets (£/yr at FY26/27 prices) — kWh × Ofgem Q2 2026
# unit rate, keyed by energy type. Used in the raking loop of impute_consumption().
_NEED_SPEND = {
    "electricity": {
        "income": [
            (lo, hi, elec_kwh * OFGEM_Q2_2026_ELEC_RATE)
            for lo, hi, _, _, elec_kwh in NEED_INCOME_BANDS
        ],
        "tenure": {k: v * OFGEM_Q2_2026_ELEC_RATE for k, v in NEED_TENURE_ELEC.items()},
        "accomm": {k: v * OFGEM_Q2_2026_ELEC_RATE for k, v in NEED_ACCOMM_ELEC.items()},
        "region": {k: v * OFGEM_Q2_2026_ELEC_RATE for k, v in NEED_REGION_ELEC.items()},
    },
    "gas": {
        "income": [
            (lo, hi, gas_kwh * OFGEM_Q2_2026_GAS_RATE)
            for lo, hi, _, gas_kwh, _ in NEED_INCOME_BANDS
        ],
        "tenure": {k: v * OFGEM_Q2_2026_GAS_RATE for k, v in NEED_TENURE_GAS.items()},
        "accomm": {k: v * OFGEM_Q2_2026_GAS_RATE for k, v in NEED_ACCOMM_GAS.items()},
        "region": {k: v * OFGEM_Q2_2026_GAS_RATE for k, v in NEED_REGION_GAS.items()},
    },
}


def _need_targets(income: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return NEED-implied annual gas and electricity spend (£) for each household."""
    gas_spend = np.full(len(income), np.nan)
    elec_spend = np.full(len(income), np.nan)
    for lo, hi, _, gas_kwh, elec_kwh in NEED_INCOME_BANDS:
        mask = (income >= lo) & (income < hi)
        gas_spend[mask] = gas_kwh * OFGEM_Q2_2026_GAS_RATE
        elec_spend[mask] = elec_kwh * OFGEM_Q2_2026_ELEC_RATE
    # Fill any missing with overall means
    overall_gas = (
        sum(g for *_, g, _ in NEED_INCOME_BANDS)
        / len(NEED_INCOME_BANDS)
        * OFGEM_Q2_2026_GAS_RATE
    )
    overall_elec = (
        sum(e for *_, e in NEED_INCOME_BANDS)
        / len(NEED_INCOME_BANDS)
        * OFGEM_Q2_2026_ELEC_RATE
    )
    gas_spend = np.where(np.isnan(gas_spend), overall_gas, gas_spend)
    elec_spend = np.where(np.isnan(elec_spend), overall_elec, elec_spend)
    return gas_spend, elec_spend


def _derive_energy_from_lcfs(household: pd.DataFrame) -> pd.DataFrame:
    """
    Derive separate electricity and gas annual spend from LCFS interview variables.

    Variable identification (from LCFS 2021/22 questionnaire structure):
    - B226: electricity direct-debit / quarterly-bill payment (weekly equivalent £)
    - B489: total energy PPM payment — when B490 also present, contains electricity+gas
    - B490: gas PPM payment (weekly equivalent £, always ≤ P537)
    - P537: total domestic energy (interview-based aggregate, weekly £)

    Logic:
    - Households with B226 > 0: electricity = B226, gas = max(P537 - B226, 0)
    - Households with B489 > 0 and B490 > 0: electricity = B489 - B490, gas = B490
    - Households with B489 > 0 only: split P537 by mean electricity share (~50%)
    - Remaining: split P537 by mean electricity share

    All values are annualised (multiply weekly × 52) downstream with other variables.
    """
    p537 = household["p537"]
    b226 = household["b226"]
    b489 = household["b489"]
    b490 = household["b490"]

    # Mean electricity share from DD-billed households (B226/P537 median ≈ 0.55)
    dd_mask = (b226 > 0) & (p537 > 0)
    mean_elec_share = (b226[dd_mask] / p537[dd_mask]).clip(0, 1).mean()
    if np.isnan(mean_elec_share):
        mean_elec_share = 0.52  # fallback: typical UK elec/(elec+gas) spend share

    electricity = np.zeros(len(household))
    gas = np.zeros(len(household))

    # Case 1: electricity billed separately via DD/quarterly (B226 > 0)
    mask1 = b226 > 0
    electricity[mask1] = b226[mask1]
    gas[mask1] = np.maximum(p537[mask1] - b226[mask1], 0)

    # Case 2: both fuels on PPM meters (B489 total, B490 gas portion)
    mask2 = (~mask1) & (b489 > 0) & (b490 > 0)
    electricity[mask2] = np.maximum(b489[mask2] - b490[mask2], 0)
    gas[mask2] = b490[mask2]

    # Case 3: electricity PPM only (B489 > 0, B490 = 0)
    mask3 = (~mask1) & (b489 > 0) & (b490 == 0)
    electricity[mask3] = b489[mask3] * mean_elec_share
    gas[mask3] = b489[mask3] * (1 - mean_elec_share)

    # Case 4: no bill variables available — split P537 by mean share
    mask4 = (~mask1) & (b489 == 0)
    electricity[mask4] = p537[mask4] * mean_elec_share
    gas[mask4] = p537[mask4] * (1 - mean_elec_share)

    # Clamp to non-negative; raw LCFS bill variables occasionally produce
    # small negatives (e.g. B490 > B489 inconsistency, or implausible
    # negative P537 entries). Consumption totals can't be negative by
    # definition and downstream NEED calibration preserves zero.
    electricity = np.maximum(electricity, 0.0)
    gas = np.maximum(gas, 0.0)

    household = household.copy()
    household["electricity_consumption"] = electricity
    household["gas_consumption"] = gas
    return household


def _normalise_lcfs_columns(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data.columns = [column.lower() for column in data.columns]
    return data


def _calibrate_energy_to_need(
    household: pd.DataFrame, income_col: str = "household_gross_income"
) -> pd.DataFrame:
    """
    Rescale imputed electricity and gas spend to match NEED 2023 income-band means
    at FY26/27 price levels (Ofgem Q2 2026 rates).

    NEED 2023 income bands use Experian modelled gross household income, so we
    match against gross income rather than HBAI net income.

    For each NEED income band, computes the ratio of the NEED-implied mean spend
    to the LCFS-derived mean spend and applies it multiplicatively. This preserves
    within-band distributional shape while anchoring the level to admin data.
    """
    income = household[income_col].values  # already annual at this point
    gas_target, elec_target = _need_targets(income)

    household = household.copy()
    for lo, hi, _, _, _ in NEED_INCOME_BANDS:
        mask = (income >= lo) & (income < hi)
        if mask.sum() == 0:
            continue
        lcfs_gas_mean = household["gas_consumption"][mask].mean()
        lcfs_elec_mean = household["electricity_consumption"][mask].mean()
        need_gas_mean = gas_target[mask].mean()
        need_elec_mean = elec_target[mask].mean()

        if lcfs_gas_mean > 0:
            household.loc[mask, "gas_consumption"] *= need_gas_mean / lcfs_gas_mean
        if lcfs_elec_mean > 0:
            household.loc[mask, "electricity_consumption"] *= (
                need_elec_mean / lcfs_elec_mean
            )

    return household


def create_has_fuel_model():
    """
    Train a model to predict has_fuel_consumption from demographics.

    Uses WAS vehicle ownership: 90% of vehicle owners use petrol/diesel (ICE + hybrid).
    """
    from policyengine_uk_data.utils.qrf import QRF
    from policyengine_uk_data.datasets.imputations.wealth import (
        WAS_TAB_FOLDER,
        generate_was_table,
    )

    model_path = get_has_fuel_model_path()
    if model_path.exists():
        cached = QRF(file_path=model_path)
        if _qrf_model_matches_current_metadata(
            cached,
            get_has_fuel_model_metadata(),
            ["has_fuel_consumption"],
        ):
            return cached

    was = pd.read_csv(
        WAS_TAB_FOLDER / CURRENT_WAS_RELEASE.household_tab_filename,
        sep="\t",
        low_memory=False,
    )
    was = generate_was_table(was)

    num_vehicles = was["num_vehicles"].fillna(0).clip(lower=0)
    has_vehicle = num_vehicles > 0
    # Use a local RNG so we don't mutate the global np.random state (which
    # would silently change any unrelated consumer of np.random that runs
    # after this function).
    rng = np.random.default_rng(_HAS_FUEL_SEED)
    has_fuel = (
        has_vehicle & (rng.random(len(was)) < NTS_2024_ICE_VEHICLE_SHARE)
    ).astype(float)

    was_df = was[HAS_FUEL_PREDICTOR_VARIABLES].copy()
    was_df["has_fuel_consumption"] = has_fuel
    was_df = was_df.dropna()

    model = QRF()
    model.metadata = get_has_fuel_model_metadata()
    model.fit(
        was_df[HAS_FUEL_PREDICTOR_VARIABLES],
        was_df[["has_fuel_consumption"]],
    )
    model.save(model_path)
    return model


def impute_has_fuel_to_lcfs(household: pd.DataFrame) -> pd.DataFrame:
    model = create_has_fuel_model()
    input_df = pd.DataFrame(
        {
            "household_net_income": household["hbai_household_net_income"],
            "num_adults": household["is_adult"],
            "num_children": household["is_child"],
            "private_pension_income": household["private_pension_income"],
            "employment_income": household["employment_income"],
            "self_employment_income": household["self_employment_income"],
            "region": household["region"],
        }
    )
    output_df = model.predict(input_df)
    household = household.copy()
    household["has_fuel_consumption"] = output_df["has_fuel_consumption"].values.clip(
        0, 1
    )
    return household


def generate_lcfs_table(lcfs_person: pd.DataFrame, lcfs_household: pd.DataFrame):
    """
    Build the LCFS training table for consumption imputation.

    Adds electricity and gas derived from interview variables (B226/B489/B490),
    calibrates to NEED 2023 income-band targets, and includes housing predictors
    (tenure_type, accommodation_type) alongside the existing income/demographic ones.
    """
    lcfs_person = _normalise_lcfs_columns(lcfs_person)
    lcfs_household = _normalise_lcfs_columns(lcfs_household)

    person = lcfs_person.rename(columns=PERSON_LCF_RENAMES)
    household = lcfs_household.rename(columns=HOUSEHOLD_LCF_RENAMES)
    household["region"] = household["region"].map(REGIONS)

    # Housing predictors — map LCFS codes to FRS enum strings
    household["tenure_type"] = lcfs_household["a122"].map(LCFS_TENURE_MAP)
    household["accommodation_type"] = lcfs_household["a121"].map(LCFS_ACCOMM_MAP)

    # Derive gas and electricity before renaming/annualising P537
    household = _derive_energy_from_lcfs(household)

    household = household.rename(columns=CONSUMPTION_VARIABLE_RENAMES)

    # Bus & coach fares (COICOP 7.3.2), summed from the detailed LCFS codes.
    # Recorded household-level only — LCFS has no person-level fare field — so
    # this is the household total; allocating to individuals (e.g. for an
    # age-targeted fare reform) requires an external age-usage profile (NTS).
    household["bus_fare_spending"] = sum(
        pd.to_numeric(household[code], errors="coerce").fillna(0)
        for code in BUS_FARE_LCFS_CODES
    )

    # Annualise weekly LCFS values. Use the same WEEKS_IN_YEAR constant
    # (365.25 / 7 ≈ 52.1786) as `datasets/frs.py` rather than a bare `* 52`,
    # which underestimates annual totals by ~0.34% and skews VAT / energy
    # imputation targets against FRS income.
    annualise = list(CONSUMPTION_VARIABLE_RENAMES.values()) + [
        "bus_fare_spending",
        "hbai_household_net_income",
        "household_gross_income",
        "electricity_consumption",
        "gas_consumption",
    ]
    for variable in annualise:
        household[variable] = household[variable] * WEEKS_IN_YEAR
    for variable in PERSON_LCF_RENAMES.values():
        totals_by_case = person.groupby("case")[variable].sum()
        household[variable] = (
            household["case"].map(totals_by_case).fillna(0) * WEEKS_IN_YEAR
        )
    household.household_weight *= 1_000

    # Calibrate energy to NEED 2023 targets by income band
    household = _calibrate_energy_to_need(household)
    household["domestic_energy_consumption"] = (
        household["electricity_consumption"] + household["gas_consumption"]
    )

    # Impute has_fuel_consumption from WAS vehicle ownership
    household = impute_has_fuel_to_lcfs(household)

    return household[PREDICTOR_VARIABLES + IMPUTATIONS + ["household_weight"]].dropna()


def uprate_lcfs_table(household: pd.DataFrame, time_period: str) -> pd.DataFrame:
    from policyengine_uk.system import system

    start_period = CURRENT_LCFS_RELEASE.fuel_price_year
    target_year = int(str(time_period)[:4])
    for variable in FUEL_PRICE_PARAMETER_NAME:
        household[variable] *= fuel_spending_litre_proxy_uprating(
            variable=variable,
            start_year=start_period,
            end_year=target_year,
            parameters=system.parameters,
        )

    cpi = system.parameters.gov.economic_assumptions.indices.obr.consumer_price_index
    cpi_uprating = cpi(time_period) / cpi(start_period)

    energy_vars = {
        "electricity_consumption",
        "gas_consumption",
        "domestic_energy_consumption",
    }
    for variable in IMPUTATIONS:
        if (
            variable not in ["petrol_spending", "diesel_spending"]
            and variable not in energy_vars
        ):
            household[variable] *= cpi_uprating
    # Uprate income predictor so training distribution matches FRS target year
    for col in [
        "hbai_household_net_income",
        "household_gross_income",
        "employment_income",
        "self_employment_income",
        "private_pension_income",
    ]:
        if col in household.columns:
            household[col] *= cpi_uprating
    return household


def fuel_spending_litre_proxy_uprating(
    *,
    variable: str,
    start_year: int,
    end_year: int,
    parameters=None,
) -> float:
    """Uprate LCFS fuel spending to target-year litres at model pump prices."""
    from policyengine_uk.system import system
    from policyengine_uk_data.sources.road_fuel_volume import (
        road_fuel_volume_uprating,
    )

    if variable not in FUEL_PRICE_PARAMETER_NAME:
        raise ValueError(f"Unsupported fuel variable: {variable}")

    if parameters is None:
        parameters = system.parameters

    try:
        lcfs_price = LCFS_FUEL_PRICE_GBP_PER_LITRE[variable][start_year]
    except KeyError as exc:
        raise ValueError(
            f"Missing LCFS fuel price for {variable} in fiscal year {start_year}"
        ) from exc

    model_price = getattr(
        parameters.household.consumption.fuel.prices,
        FUEL_PRICE_PARAMETER_NAME[variable],
    )
    population = parameters.gov.economic_assumptions.indices.ons.population
    return (
        road_fuel_volume_uprating(start_year=start_year, end_year=end_year)
        / (population(end_year) / population(start_year))
        * model_price(end_year)
        / lcfs_price
    )


def _fuel_litre_proxy_mlitres(
    household: pd.DataFrame,
    fiscal_year: int,
    parameters=None,
) -> float:
    """Return weighted petrol + diesel litres represented by spending proxies."""
    from policyengine_uk.system import system

    if parameters is None:
        parameters = system.parameters

    total_litres = 0.0
    for variable, price_parameter_name in FUEL_PRICE_PARAMETER_NAME.items():
        price = getattr(
            parameters.household.consumption.fuel.prices,
            price_parameter_name,
        )(fiscal_year)
        total_litres += household[variable] / price
    return (total_litres * household["household_weight"]).sum() / 1_000_000


def calibrate_fuel_litre_proxies_to_road_fuel(
    household: pd.DataFrame,
    fiscal_year: int,
    parameters=None,
) -> float:
    """Scale imputed fuel proxies to HMRC/OBR road-fuel litre controls.

    PolicyEngine UK derives petrol and diesel litres from spending divided by
    pump prices. Applying one multiplicative factor to petrol and diesel
    spending preserves the household distribution while making the resulting
    litres reconcile to the external fuel-duty volume benchmark.
    """
    from policyengine_uk_data.sources.road_fuel_volume import (
        road_fuel_clearances_mlitres,
    )

    actual_mlitres = _fuel_litre_proxy_mlitres(
        household,
        fiscal_year,
        parameters=parameters,
    )
    if actual_mlitres <= 0:
        return 1.0

    target_mlitres = road_fuel_clearances_mlitres(end_year=fiscal_year)[fiscal_year]
    scale = target_mlitres / actual_mlitres
    for variable in FUEL_PRICE_PARAMETER_NAME:
        household[variable] *= scale
    return scale


def calibrate_dataset_fuel_litre_proxies_to_road_fuel(
    dataset: UKSingleYearDataset,
    parameters=None,
) -> float:
    """Scale a dataset's fuel proxies after final household weights are set."""
    return calibrate_fuel_litre_proxies_to_road_fuel(
        dataset.household,
        int(dataset.time_period),
        parameters=parameters,
    )


def save_imputation_models():
    from policyengine_uk_data.utils.qrf import QRF

    consumption = QRF()
    consumption.metadata = get_consumption_model_metadata()
    lcfs_household = pd.read_csv(
        LCFS_TAB_FOLDER / CURRENT_LCFS_RELEASE.household_tab_filename,
        delimiter="\t",
        low_memory=False,
    )
    lcfs_person = pd.read_csv(
        LCFS_TAB_FOLDER / CURRENT_LCFS_RELEASE.person_tab_filename,
        delimiter="\t",
    )
    household = generate_lcfs_table(lcfs_person, lcfs_household)
    household = uprate_lcfs_table(household, str(CURRENT_FRS_RELEASE.base_year))
    consumption.fit(household[PREDICTOR_VARIABLES], household[IMPUTATIONS])
    consumption.save(get_consumption_model_path())
    return consumption


def create_consumption_model(overwrite_existing: bool = False):
    from policyengine_uk_data.utils.qrf import QRF

    model_path = get_consumption_model_path()
    if model_path.exists() and not overwrite_existing:
        cached = QRF(file_path=model_path)
        if _qrf_model_matches_current_metadata(
            cached,
            get_consumption_model_metadata(),
            IMPUTATIONS,
        ):
            return cached
    return save_imputation_models()


def impute_consumption(dataset: UKSingleYearDataset) -> UKSingleYearDataset:
    """
    Impute consumption variables (including separate electricity and gas) using
    LCFS-trained QRF model with housing and demographic predictors.
    """
    dataset = dataset.copy()

    sim = Microsimulation(dataset=dataset)
    num_vehicles = sim.calculate("num_vehicles", map_to="household").values

    # Local RNG — see note at module level (_HAS_FUEL_SEED).
    rng = np.random.default_rng(_HAS_FUEL_SEED)
    has_vehicle = num_vehicles > 0
    is_ice = rng.random(len(num_vehicles)) < NTS_2024_ICE_VEHICLE_SHARE
    has_fuel_consumption = (has_vehicle & is_ice).astype(float)
    dataset.household["has_fuel_consumption"] = has_fuel_consumption

    model = create_consumption_model()
    predictors = model.input_columns

    non_fuel_predictors = [p for p in predictors if p != "has_fuel_consumption"]
    input_df = sim.calculate_dataframe(non_fuel_predictors, map_to="household")
    input_df["has_fuel_consumption"] = has_fuel_consumption

    output_df = model.predict(input_df)
    for column in output_df.columns:
        dataset.household[column] = output_df[column].values

    # Re-calibrate electricity and gas to NEED 2023 kWh targets (converted to
    # FY26/27 £ at Ofgem Q2 2026 rates) using iterative raking over income band,
    # tenure, accommodation type, and region. Energy variables are not CPI-uprated,
    # so raking at the target-year price level ensures stored £/yr values are
    # correct for the simulation year without further adjustment.
    # NEED income bands use Experian modelled gross income, so we use
    # household_gross_income rather than hbai_household_net_income.
    income = sim.calculate("household_gross_income", map_to="household").values
    tenure = sim.calculate("tenure_type", map_to="household").values
    accomm = sim.calculate("accommodation_type", map_to="household").values
    region = sim.calculate("region", map_to="household").values
    weights = sim.calculate("household_weight", map_to="household").values

    # Pre-compute boolean masks (reused across raking iterations)
    income_masks = []
    for lo, hi, _ in _NEED_SPEND["electricity"]["income"]:
        income_masks.append((income >= lo) & (income < hi))
    tenure_masks = {frs_val: tenure == frs_val for frs_val in TENURE_TO_NEED}
    accomm_masks = {frs_val: accomm == frs_val for frs_val in ACCOMM_TO_NEED}
    region_masks = {reg: region == reg for reg in _NEED_SPEND["electricity"]["region"]}

    def _wmean(arr, mask):
        w = weights[mask]
        total_w = w.sum()
        if total_w == 0:
            return 0.0
        return (arr[mask] * w).sum() / total_w

    for energy, col in [
        ("electricity", "electricity_consumption"),
        ("gas", "gas_consumption"),
    ]:
        targets = _NEED_SPEND[energy]
        arr = dataset.household[col].values.copy()
        for _ in range(50):  # iterative raking
            for i, (_, _, target) in enumerate(targets["income"]):
                mask = income_masks[i]
                wm = _wmean(arr, mask)
                if mask.sum() > 0 and wm > 0:
                    arr[mask] *= target / wm
            for frs_val, need_key in TENURE_TO_NEED.items():
                if need_key not in targets["tenure"]:
                    continue
                mask = tenure_masks[frs_val]
                wm = _wmean(arr, mask)
                if mask.sum() > 0 and wm > 0:
                    arr[mask] *= targets["tenure"][need_key] / wm
            for frs_val, need_key in ACCOMM_TO_NEED.items():
                if need_key not in targets["accomm"]:
                    continue
                mask = accomm_masks[frs_val]
                wm = _wmean(arr, mask)
                if mask.sum() > 0 and wm > 0:
                    arr[mask] *= targets["accomm"][need_key] / wm
            for reg_val, target in targets["region"].items():
                mask = region_masks[reg_val]
                wm = _wmean(arr, mask)
                if mask.sum() > 0 and wm > 0:
                    arr[mask] *= target / wm
        dataset.household[col] = arr

    dataset.household["domestic_energy_consumption"] = (
        dataset.household["electricity_consumption"]
        + dataset.household["gas_consumption"]
    )

    # Zero out car-fuel spending for non-ICE households
    no_fuel = has_fuel_consumption == 0
    dataset.household["petrol_spending"][no_fuel] = 0
    dataset.household["diesel_spending"][no_fuel] = 0

    dataset.validate()
    return dataset
