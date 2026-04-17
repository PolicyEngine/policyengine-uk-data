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
from policyengine_uk_data.storage import STORAGE_FOLDER
from policyengine_uk.data import UKSingleYearDataset
from policyengine_uk import Microsimulation
from policyengine_uk_data.datasets.frs import WEEKS_IN_YEAR

LCFS_TAB_FOLDER = STORAGE_FOLDER / "lcfs_2021_22"

# Default seed for the stochastic ICE-vehicle flag drawn from
# `NTS_2024_ICE_VEHICLE_SHARE`. Kept at 42 for backward compatibility with
# existing artefact fingerprints; callers can override via the fixture's
# local RNG rather than the process-wide np.random global.
_HAS_FUEL_SEED = 42

# EV/ICE vehicle mix from NTS 2024
NTS_2024_ICE_VEHICLE_SHARE = 0.90

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
# LCFS coding inferred from LCFS 2021/22 user guide:
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
    "G018": "is_adult",
    "G019": "is_child",
    "Gorx": "region",
    "P389p": "hbai_household_net_income",
    "p344p": "household_gross_income",
    "weighta": "household_weight",
}
PERSON_LCF_RENAMES = {
    "B303p": "employment_income",
    "B3262p": "self_employment_income",
    "B3381": "state_pension",
    "P049p": "private_pension_income",
}

CONSUMPTION_VARIABLE_RENAMES = {
    "P601": "food_and_non_alcoholic_beverages_consumption",
    "P602": "alcohol_and_tobacco_consumption",
    "P603": "clothing_and_footwear_consumption",
    "P604": "housing_water_and_electricity_consumption",
    "P605": "household_furnishings_consumption",
    "P606": "health_consumption",
    "P607": "transport_consumption",
    "P608": "communication_consumption",
    "P609": "recreation_consumption",
    "P610": "education_consumption",
    "P611": "restaurants_and_hotels_consumption",
    "P612": "miscellaneous_consumption",
    "C72211": "petrol_spending",
    "C72212": "diesel_spending",
    "P537": "domestic_energy_consumption",  # aggregate kept for backward compat
}

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
    "domestic_energy_consumption",  # aggregate; backward compat with price cap subsidy
    "electricity_consumption",
    "gas_consumption",
]

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
    p537 = household["P537"]
    b226 = household["B226"]
    b489 = household["B489"]
    b490 = household["B490"]

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

    household = household.copy()
    household["electricity_consumption"] = electricity
    household["gas_consumption"] = gas
    return household


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
        REGIONS,
    )

    model_path = STORAGE_FOLDER / "has_fuel_model.pkl"
    if model_path.exists():
        return QRF(file_path=model_path)

    was = pd.read_csv(
        WAS_TAB_FOLDER / "was_round_7_hhold_eul_march_2022.tab",
        sep="\t",
        low_memory=False,
    )
    was.columns = [c.lower() for c in was.columns]

    num_vehicles = was["vcarnr7"].fillna(0).clip(lower=0)
    has_vehicle = num_vehicles > 0
    # Use a local RNG so we don't mutate the global np.random state (which
    # would silently change any unrelated consumer of np.random that runs
    # after this function).
    rng = np.random.default_rng(_HAS_FUEL_SEED)
    has_fuel = (
        has_vehicle & (rng.random(len(was)) < NTS_2024_ICE_VEHICLE_SHARE)
    ).astype(float)

    was_df = pd.DataFrame(
        {
            "household_net_income": was["dvtotinc_bhcr7"],
            "num_adults": was["numadultr7"],
            "num_children": was["numch18r7"],
            "private_pension_income": was["dvgippenr7_aggr"],
            "employment_income": was["dvgiempr7_aggr"],
            "self_employment_income": was["dvgiser7_aggr"],
            "region": was["gorr7"].map(REGIONS),
            "has_fuel_consumption": has_fuel,
        }
    ).dropna()

    predictors = [
        "household_net_income",
        "num_adults",
        "num_children",
        "private_pension_income",
        "employment_income",
        "self_employment_income",
        "region",
    ]
    model = QRF()
    model.fit(was_df[predictors], was_df[["has_fuel_consumption"]])
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
    person = lcfs_person.rename(columns=PERSON_LCF_RENAMES)
    household = lcfs_household.rename(columns=HOUSEHOLD_LCF_RENAMES)
    household["region"] = household["region"].map(REGIONS)

    # Housing predictors — map LCFS codes to FRS enum strings
    household["tenure_type"] = lcfs_household["A122"].map(LCFS_TENURE_MAP)
    household["accommodation_type"] = lcfs_household["A121"].map(LCFS_ACCOMM_MAP)

    # Derive gas and electricity before renaming/annualising P537
    household = _derive_energy_from_lcfs(household)

    household = household.rename(columns=CONSUMPTION_VARIABLE_RENAMES)

    # Annualise weekly LCFS values. Use the same WEEKS_IN_YEAR constant
    # (365.25 / 7 ≈ 52.1786) as `datasets/frs.py` rather than a bare `* 52`,
    # which underestimates annual totals by ~0.34% and skews VAT / energy
    # imputation targets against FRS income.
    annualise = list(CONSUMPTION_VARIABLE_RENAMES.values()) + [
        "hbai_household_net_income",
        "household_gross_income",
        "electricity_consumption",
        "gas_consumption",
    ]
    for variable in annualise:
        household[variable] = household[variable] * WEEKS_IN_YEAR
    for variable in PERSON_LCF_RENAMES.values():
        household[variable] = (
            person[variable].groupby(person.case).sum()[household.case]
            * WEEKS_IN_YEAR
        )
    household.household_weight *= 1_000

    # Calibrate energy to NEED 2023 targets by income band
    household = _calibrate_energy_to_need(household)

    # Impute has_fuel_consumption from WAS vehicle ownership
    household = impute_has_fuel_to_lcfs(household)

    return household[PREDICTOR_VARIABLES + IMPUTATIONS + ["household_weight"]].dropna()


def uprate_lcfs_table(household: pd.DataFrame, time_period: str) -> pd.DataFrame:
    from policyengine_uk.system import system

    start_period = 2021
    fuel_uprating = 1.3
    household["petrol_spending"] *= fuel_uprating
    household["diesel_spending"] *= fuel_uprating

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


def save_imputation_models():
    from policyengine_uk_data.utils.qrf import QRF

    consumption = QRF()
    lcfs_household = pd.read_csv(
        LCFS_TAB_FOLDER / "lcfs_2021_dvhh_ukanon.tab",
        delimiter="\t",
        low_memory=False,
    )
    lcfs_person = pd.read_csv(
        LCFS_TAB_FOLDER / "lcfs_2021_dvper_ukanon202122.tab", delimiter="\t"
    )
    household = generate_lcfs_table(lcfs_person, lcfs_household)
    household = uprate_lcfs_table(household, "2024")
    consumption.fit(household[PREDICTOR_VARIABLES], household[IMPUTATIONS])
    consumption.save(STORAGE_FOLDER / "consumption.pkl")
    return consumption


def create_consumption_model(overwrite_existing: bool = False):
    from policyengine_uk_data.utils.qrf import QRF

    if (STORAGE_FOLDER / "consumption.pkl").exists() and not overwrite_existing:
        return QRF(file_path=STORAGE_FOLDER / "consumption.pkl")
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

    # Zero out car-fuel spending for non-ICE households
    no_fuel = has_fuel_consumption == 0
    dataset.household["petrol_spending"][no_fuel] = 0
    dataset.household["diesel_spending"][no_fuel] = 0

    dataset.validate()
    return dataset
