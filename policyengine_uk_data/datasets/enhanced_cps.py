"""Public UK enhanced CPS built from open U.S. microdata.

This dataset starts from PolicyEngine's public 1,000-household CPS-derived
sample, maps those records into a UKSingleYearDataset, and then calibrates the
household weights to UK national/region/country targets.

It is a public calibrated dataset. It does not replace the FRS or enhanced FRS,
but it follows the same general strategy: public microdata plus imputations and
reweighting against official aggregates.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from policyengine_uk.data import UKSingleYearDataset

from policyengine_uk_data.storage import STORAGE_FOLDER
from policyengine_uk_data.utils.reweight import calibrate_household_weights

ENHANCED_CPS_SOURCE_FILE = STORAGE_FOLDER / "enhanced_cps_source_2025.csv"
ENHANCED_CPS_FILE = STORAGE_FOLDER / "enhanced_cps_2025.h5"
COUNCIL_TAX_BANDS_FILE = STORAGE_FOLDER / "council_tax_bands_2024.csv"

USD_TO_GBP = 0.79
NEW_STATE_PENSION_2025 = 224.96 * 52

REGION_SHARES = (
    ("NORTH_EAST", 0.04),
    ("NORTH_WEST", 0.11),
    ("YORKSHIRE", 0.08),
    ("EAST_MIDLANDS", 0.07),
    ("WEST_MIDLANDS", 0.09),
    ("EAST_OF_ENGLAND", 0.10),
    ("LONDON", 0.13),
    ("SOUTH_EAST", 0.15),
    ("SOUTH_WEST", 0.09),
    ("WALES", 0.05),
    ("SCOTLAND", 0.08),
    ("NORTHERN_IRELAND", 0.03),
)

BASE_CONSUMPTION_SHARES = {
    "food_and_non_alcoholic_beverages_consumption": 0.13,
    "alcohol_and_tobacco_consumption": 0.03,
    "clothing_and_footwear_consumption": 0.05,
    "household_furnishings_consumption": 0.05,
    "communication_consumption": 0.03,
    "recreation_consumption": 0.10,
    "restaurants_and_hotels_consumption": 0.08,
    "miscellaneous_consumption": 0.17,
}


def _gbp(value: float | int, exchange_rate: float) -> float:
    return round(float(value) * exchange_rate, 2)


def _pick_region(household_id: int) -> str:
    cumulative = 0.0
    threshold = ((household_id * 2654435761) % 10_000) / 10_000
    for region, share in REGION_SHARES:
        cumulative += share
        if threshold <= cumulative:
            return region
    return REGION_SHARES[-1][0]


def _sum_inputs(people: list[dict], *keys: str) -> float:
    total = 0.0
    for person in people:
        inputs = person.get("inputs", {})
        total += sum(float(inputs.get(key, 0.0)) for key in keys)
    return total


def _vehicle_count(
    household_inputs: dict,
    adults: list[dict],
    region: str,
    owns_home: bool,
) -> int:
    employed_adults = sum(
        1
        for adult in adults
        if float(adult.get("employment_income", 0.0))
        + float(adult.get("inputs", {}).get("self_employment_income", 0.0))
        > 0
    )
    auto_loan_balance = float(household_inputs.get("auto_loan_balance", 0.0))
    auto_loan_interest = float(household_inputs.get("auto_loan_interest", 0.0))
    wealth = float(household_inputs.get("net_worth", 0.0))

    vehicles = 0
    if auto_loan_balance > 0 or auto_loan_interest > 0:
        vehicles = 1
    elif employed_adults > 0 and region != "LONDON":
        vehicles = 1
    elif employed_adults > 0 and owns_home:
        vehicles = 1
    elif wealth > 150_000 and owns_home and region != "LONDON":
        vehicles = 1

    if employed_adults >= 2 and region != "LONDON":
        vehicles = max(vehicles, 2)
    if auto_loan_balance > 25_000 and employed_adults >= 2:
        vehicles = 2

    return min(vehicles, 2)


def _state_pension_amount(person_inputs: dict, exchange_rate: float) -> float:
    social_security = _gbp(
        float(person_inputs.get("social_security_retirement", 0.0)),
        exchange_rate,
    )
    return min(social_security, NEW_STATE_PENSION_2025)


def _private_pension_amount(person_inputs: dict, exchange_rate: float) -> float:
    return _gbp(
        float(person_inputs.get("taxable_private_pension_income", 0.0))
        + float(person_inputs.get("taxable_ira_distributions", 0.0)),
        exchange_rate,
    )


def _capital_gains_amount(person_inputs: dict, exchange_rate: float) -> float:
    return _gbp(
        float(person_inputs.get("short_term_capital_gains", 0.0))
        + float(person_inputs.get("long_term_capital_gains", 0.0)),
        exchange_rate,
    )


def _pip_category(person: dict) -> str:
    inputs = person.get("inputs", {})
    disabled = bool(inputs.get("is_disabled", False))
    age = int(person.get("age", 0))
    if not disabled or age < 16:
        return "NONE"

    severe_signal = (
        float(inputs.get("disability_benefits", 0.0)) > 0
        or float(inputs.get("social_security_disability", 0.0)) > 0
        or float(inputs.get("veterans_benefits", 0.0)) > 0
    )
    low_earnings = (
        float(person.get("employment_income", 0.0))
        + float(inputs.get("self_employment_income", 0.0))
        < 12_000
    )
    return "ENHANCED" if severe_signal or low_earnings else "STANDARD"


def _household_cash_income(people: list[dict], exchange_rate: float) -> float:
    total = 0.0
    for person in people:
        inputs = person.get("inputs", {})
        total += _gbp(float(person.get("employment_income", 0.0)), exchange_rate)
        total += _gbp(float(inputs.get("self_employment_income", 0.0)), exchange_rate)
        total += _state_pension_amount(inputs, exchange_rate)
        total += _private_pension_amount(inputs, exchange_rate)
        total += _gbp(float(inputs.get("rental_income", 0.0)), exchange_rate)
        total += _gbp(
            float(inputs.get("taxable_interest_income", 0.0))
            + float(inputs.get("tax_exempt_interest_income", 0.0)),
            exchange_rate,
        )
        total += _gbp(
            float(inputs.get("qualified_dividend_income", 0.0))
            + float(inputs.get("non_qualified_dividend_income", 0.0)),
            exchange_rate,
        )
        total += _gbp(
            float(inputs.get("miscellaneous_income", 0.0))
            + float(inputs.get("social_security_disability", 0.0))
            + float(inputs.get("social_security_survivors", 0.0))
            + float(inputs.get("social_security_dependents", 0.0))
            + float(inputs.get("disability_benefits", 0.0))
            + float(inputs.get("veterans_benefits", 0.0)),
            exchange_rate,
        )
    return total


def _consumption_profile(
    *,
    cash_income: float,
    household_wealth: float,
    council_tax: float,
    household_rent: float,
    mortgage_interest_repayment: float,
    adults: int,
    children: int,
    vehicle_count: int,
    owns_home: bool,
    has_disabled_adult: bool,
) -> dict[str, float]:
    base_floor = 8_500 + 2_000 * adults + 1_800 * children
    dissaving = min(household_wealth * 0.012, 18_000)
    housing_costs = household_rent + mortgage_interest_repayment + council_tax
    total = max(
        base_floor,
        0.78 * cash_income + 0.45 * housing_costs + dissaving,
    )

    shares = dict(BASE_CONSUMPTION_SHARES)
    shares["transport_consumption"] = 0.07 + 0.04 * vehicle_count
    shares["housing_water_and_electricity_consumption"] = (
        0.20 if household_rent > 0 or mortgage_interest_repayment > 0 else 0.15
    )
    shares["health_consumption"] = 0.05 if has_disabled_adult else 0.03
    shares["education_consumption"] = 0.03 if children > 0 else 0.01

    if owns_home:
        shares["household_furnishings_consumption"] += 0.01
        shares["miscellaneous_consumption"] -= 0.01

    share_total = sum(shares.values())
    shares = {key: value / share_total for key, value in shares.items()}
    components = {key: round(total * share, 2) for key, share in shares.items()}

    transport = components["transport_consumption"]
    if vehicle_count == 0:
        petrol_spending = 0.0
        diesel_spending = 0.0
    elif vehicle_count == 1:
        petrol_spending = round(transport * 0.32, 2)
        diesel_spending = round(transport * 0.04, 2)
    else:
        petrol_spending = round(transport * 0.30, 2)
        diesel_spending = round(transport * 0.10, 2)

    full_rate_vat_expenditure_rate = 0.44
    full_rate_vat_expenditure_rate += 0.02 * int(adults == 2)
    full_rate_vat_expenditure_rate -= 0.03 * int(children > 0)
    full_rate_vat_expenditure_rate += 0.02 * int(cash_income > 60_000)
    full_rate_vat_expenditure_rate = float(
        np.clip(full_rate_vat_expenditure_rate, 0.35, 0.55)
    )

    return {
        **components,
        "petrol_spending": petrol_spending,
        "diesel_spending": diesel_spending,
        "full_rate_vat_expenditure_rate": full_rate_vat_expenditure_rate,
    }


def _load_council_tax_band_shares() -> pd.DataFrame:
    band_counts = pd.read_csv(COUNCIL_TAX_BANDS_FILE)
    bands = ["A", "B", "C", "D", "E", "F", "G", "H", "I"]
    band_counts["Region"] = band_counts["Region"].replace(
        {
            "East of England": "EAST_OF_ENGLAND",
            "East_of_England": "EAST_OF_ENGLAND",
        }
    )
    band_counts["Region"] = band_counts["Region"].str.upper().str.replace(" ", "_")
    totals = band_counts[bands].sum(axis=1)
    shares = band_counts[["Region", *bands]].copy()
    shares[bands] = shares[bands].div(totals, axis=0)
    return shares.set_index("Region")


def _assign_council_tax_bands(households: pd.DataFrame) -> pd.DataFrame:
    bands = ["A", "B", "C", "D", "E", "F", "G", "H", "I"]
    band_shares = _load_council_tax_band_shares()
    households = households.copy()
    households["council_tax_band"] = "D"

    for region, region_rows in households.groupby("region", sort=False):
        if region not in band_shares.index:
            continue
        shares = band_shares.loc[region, bands].astype(float).values
        cumulative = np.cumsum(shares)
        ordered = region_rows.sort_values(
            ["housing_score", "household_id"],
            ascending=[True, True],
        )
        total_weight = ordered["household_weight"].sum()
        if total_weight <= 0:
            continue
        percentiles = (
            ordered["household_weight"].cumsum() - 0.5 * ordered["household_weight"]
        ) / total_weight
        band_codes = [
            bands[np.searchsorted(cumulative, pct, side="right")] for pct in percentiles
        ]
        households.loc[ordered.index, "council_tax_band"] = band_codes

    return households


def _build_base_dataset(
    source_file_path: str | Path,
    fiscal_year: int,
    exchange_rate: float,
    max_rows: int | None,
) -> UKSingleYearDataset:
    source = pd.read_csv(source_file_path)
    if max_rows is not None:
        source = source.head(max_rows).copy()

    person_rows: list[dict] = []
    benunit_rows: list[dict] = []
    household_rows: list[dict] = []

    for _, row in source.iterrows():
        scenario = json.loads(row["scenario_json"])
        household_id = int(row["household_id"])
        benunit_id = household_id
        region = _pick_region(household_id)
        adults = scenario["adults"]
        children = scenario["children"]
        people = adults + children

        has_mortgage = any(
            "deductible_mortgage_interest" in person.get("inputs", {})
            for person in adults
        )
        owns_home = has_mortgage or any(
            "real_estate_taxes" in person.get("inputs", {}) for person in adults
        )
        if has_mortgage:
            tenure_type = "OWNED_WITH_MORTGAGE"
        elif owns_home:
            tenure_type = "OWNED_OUTRIGHT"
        else:
            tenure_type = "RENT_PRIVATELY"

        rent = 0.0
        if tenure_type == "RENT_PRIVATELY":
            rent = max(
                float(person.get("inputs", {}).get("pre_subsidy_rent", 0.0))
                for person in people
            )

        household_inputs = scenario.get("household_inputs", {})
        council_tax = _gbp(
            _sum_inputs(adults, "real_estate_taxes") * 0.18,
            exchange_rate,
        )
        mortgage_interest_repayment = _gbp(
            _sum_inputs(adults, "deductible_mortgage_interest"),
            exchange_rate,
        )
        household_wealth = _gbp(
            float(household_inputs.get("net_worth", 0.0)),
            exchange_rate,
        )
        household_rent = _gbp(rent, exchange_rate)
        cash_income = _household_cash_income(people, exchange_rate)
        vehicle_count = _vehicle_count(
            household_inputs=household_inputs,
            adults=adults,
            region=region,
            owns_home=owns_home,
        )
        consumption_inputs = _consumption_profile(
            cash_income=cash_income,
            household_wealth=household_wealth,
            council_tax=council_tax,
            household_rent=household_rent,
            mortgage_interest_repayment=mortgage_interest_repayment,
            adults=len(adults),
            children=len(children),
            vehicle_count=vehicle_count,
            owns_home=owns_home,
            has_disabled_adult=any(
                bool(person.get("inputs", {}).get("is_disabled", False))
                for person in adults
            ),
        )

        household_rows.append(
            {
                "household_id": household_id,
                "household_weight": float(row["household_weight"]),
                "region": region,
                "tenure_type": tenure_type,
                "council_tax": council_tax,
                "rent": household_rent,
                "mortgage_interest_repayment": mortgage_interest_repayment,
                "mortgage_capital_repayment": 0.0,
                "savings": _gbp(
                    _sum_inputs(adults, "bank_account_assets", "bond_assets"),
                    exchange_rate,
                ),
                "household_wealth": household_wealth,
                "num_vehicles": vehicle_count,
                **consumption_inputs,
                "housing_score": (
                    household_wealth
                    + 12 * household_rent
                    + 18 * council_tax
                    + 10 * mortgage_interest_repayment
                    + 15_000 * int(owns_home)
                    + 5_000 * vehicle_count
                ),
            }
        )
        benunit_rows.append({"benunit_id": benunit_id})

        for person_index, person in enumerate(people, start=1):
            inputs = person.get("inputs", {})
            person_id = household_id * 10 + person_index
            is_joint_couple = scenario["filing_status"] == "joint" and person_index <= 2

            person_rows.append(
                {
                    "person_id": person_id,
                    "person_household_id": household_id,
                    "person_benunit_id": benunit_id,
                    "age": int(person["age"]),
                    "gender": "MALE" if (household_id + person_index) % 2 else "FEMALE",
                    "marital_status": "MARRIED" if is_joint_couple else "SINGLE",
                    "employment_income": _gbp(
                        float(person.get("employment_income", 0.0)),
                        exchange_rate,
                    ),
                    "self_employment_income": _gbp(
                        float(inputs.get("self_employment_income", 0.0)),
                        exchange_rate,
                    ),
                    "savings_interest_income": _gbp(
                        float(inputs.get("taxable_interest_income", 0.0))
                        + float(inputs.get("tax_exempt_interest_income", 0.0)),
                        exchange_rate,
                    ),
                    "dividend_income": _gbp(
                        float(inputs.get("qualified_dividend_income", 0.0))
                        + float(inputs.get("non_qualified_dividend_income", 0.0)),
                        exchange_rate,
                    ),
                    "private_pension_income": _private_pension_amount(
                        inputs, exchange_rate
                    ),
                    "state_pension_reported": _state_pension_amount(
                        inputs, exchange_rate
                    ),
                    "capital_gains_before_response": _capital_gains_amount(
                        inputs, exchange_rate
                    ),
                    "property_income": _gbp(
                        float(inputs.get("rental_income", 0.0)),
                        exchange_rate,
                    ),
                    "miscellaneous_income": _gbp(
                        float(inputs.get("miscellaneous_income", 0.0))
                        + float(inputs.get("social_security_disability", 0.0))
                        + float(inputs.get("social_security_survivors", 0.0))
                        + float(inputs.get("social_security_dependents", 0.0))
                        + float(inputs.get("disability_benefits", 0.0))
                        + float(inputs.get("veterans_benefits", 0.0)),
                        exchange_rate,
                    ),
                    "employment_expenses": _gbp(
                        float(
                            inputs.get("unreimbursed_business_employee_expenses", 0.0)
                        ),
                        exchange_rate,
                    ),
                    "private_pension_contributions": _gbp(
                        float(inputs.get("traditional_401k_contributions", 0.0))
                        + float(inputs.get("traditional_ira_contributions", 0.0))
                        + max(
                            float(
                                inputs.get("self_employed_pension_contributions", 0.0)
                            ),
                            0.0,
                        ),
                        exchange_rate,
                    ),
                    "gift_aid": _gbp(
                        float(inputs.get("charitable_cash_donations", 0.0)),
                        exchange_rate,
                    ),
                    "blind_persons_allowance": 1250.0
                    if bool(inputs.get("is_blind", False))
                    else 0.0,
                    "is_disabled_for_benefits": bool(inputs.get("is_disabled", False)),
                    "pip_dl_category": _pip_category(person),
                    "pip_m_category": _pip_category(person),
                    "hours_worked": float(
                        inputs.get(
                            "weekly_hours_worked",
                            inputs.get("hours_worked_last_week", 0.0),
                        )
                    )
                    * 52,
                    "is_disabled": bool(inputs.get("is_disabled", False)),
                    "is_student": bool(
                        inputs.get("is_full_time_college_student", False)
                    ),
                }
            )

    household = _assign_council_tax_bands(pd.DataFrame(household_rows)).drop(
        columns=["housing_score"]
    )

    return UKSingleYearDataset(
        person=pd.DataFrame(person_rows),
        benunit=pd.DataFrame(benunit_rows),
        household=household,
        fiscal_year=fiscal_year,
    )


def create_enhanced_cps(
    source_file_path: str | Path = ENHANCED_CPS_SOURCE_FILE,
    fiscal_year: int = 2025,
    exchange_rate: float = USD_TO_GBP,
    max_rows: int | None = None,
    calibrate: bool = True,
):
    """Create the public UK enhanced CPS dataset."""
    dataset = _build_base_dataset(
        source_file_path=source_file_path,
        fiscal_year=fiscal_year,
        exchange_rate=exchange_rate,
        max_rows=max_rows,
    )
    if calibrate:
        weights, _ = calibrate_household_weights(dataset, str(fiscal_year))
        dataset.household.household_weight = weights
    return dataset


def save_enhanced_cps(
    output_file_path: str | Path = ENHANCED_CPS_FILE,
    source_file_path: str | Path = ENHANCED_CPS_SOURCE_FILE,
    fiscal_year: int = 2025,
    exchange_rate: float = USD_TO_GBP,
    max_rows: int | None = None,
    calibrate: bool = True,
) -> UKSingleYearDataset:
    """Create and save the public UK enhanced CPS dataset."""
    dataset = create_enhanced_cps(
        source_file_path=source_file_path,
        fiscal_year=fiscal_year,
        exchange_rate=exchange_rate,
        max_rows=max_rows,
        calibrate=calibrate,
    )
    dataset.save(output_file_path)
    return dataset
