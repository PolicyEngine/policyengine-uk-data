"""Public synthetic UK transfer dataset built from PolicyBench households.

This dataset maps PolicyBench's public 1,000-household US sample into a
synthetic UKSingleYearDataset so we can run UK policy logic on the same
household records without depending on restricted FRS microdata.

It is a transfer dataset, not a representative UK baseline:
- households originate from the US Enhanced CPS
- UK-specific geography and tenure are assigned synthetically
- monetary values are converted with a fixed USD->GBP factor
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from policyengine_uk.data import UKSingleYearDataset

from policyengine_uk_data.storage import STORAGE_FOLDER

POLICYBENCH_TRANSFER_SOURCE_FILE = (
    STORAGE_FOLDER / "policybench_transfer_source_2025.csv"
)

USD_TO_GBP = 0.79

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


def create_policybench_transfer(
    source_file_path: str | Path = POLICYBENCH_TRANSFER_SOURCE_FILE,
    fiscal_year: int = 2025,
    exchange_rate: float = USD_TO_GBP,
    max_rows: int | None = None,
) -> UKSingleYearDataset:
    """Create a synthetic UK dataset from PolicyBench's public source sample."""
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

        household_rows.append(
            {
                "household_id": household_id,
                "household_weight": float(row["household_weight"]),
                "region": region,
                "tenure_type": tenure_type,
                # US property taxes are only a rough proxy for council tax.
                "council_tax": _gbp(
                    _sum_inputs(adults, "real_estate_taxes") * 0.18,
                    exchange_rate,
                ),
                "rent": _gbp(rent, exchange_rate),
                "mortgage_interest_repayment": _gbp(
                    _sum_inputs(adults, "deductible_mortgage_interest"),
                    exchange_rate,
                ),
                "mortgage_capital_repayment": 0.0,
                "savings": _gbp(
                    _sum_inputs(adults, "bank_account_assets", "bond_assets"),
                    exchange_rate,
                ),
                "household_wealth": _gbp(
                    float(scenario.get("household_inputs", {}).get("net_worth", 0.0)),
                    exchange_rate,
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
                    # Source scenarios are public but do not include sex, so
                    # assign a deterministic synthetic gender split.
                    "gender": "MALE"
                    if (household_id + person_index) % 2
                    else "FEMALE",
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
                    "private_pension_income": _gbp(
                        float(inputs.get("taxable_private_pension_income", 0.0))
                        + float(inputs.get("tax_exempt_private_pension_income", 0.0)),
                        exchange_rate,
                    ),
                    "state_pension_reported": _gbp(
                        float(inputs.get("social_security_retirement", 0.0)),
                        exchange_rate,
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
                            inputs.get(
                                "unreimbursed_business_employee_expenses", 0.0
                            )
                        ),
                        exchange_rate,
                    ),
                    "private_pension_contributions": _gbp(
                        float(inputs.get("traditional_401k_contributions", 0.0))
                        + float(inputs.get("traditional_ira_contributions", 0.0))
                        + max(
                            float(inputs.get("self_employed_pension_contributions", 0.0)),
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

    dataset = UKSingleYearDataset(
        person=pd.DataFrame(person_rows),
        benunit=pd.DataFrame(benunit_rows),
        household=pd.DataFrame(household_rows),
        fiscal_year=fiscal_year,
    )
    return dataset


def save_policybench_transfer(
    output_file_path: str | Path = STORAGE_FOLDER / "policybench_transfer_2025.h5",
    source_file_path: str | Path = POLICYBENCH_TRANSFER_SOURCE_FILE,
    fiscal_year: int = 2025,
    exchange_rate: float = USD_TO_GBP,
    max_rows: int | None = None,
) -> UKSingleYearDataset:
    """Create and save the public PolicyBench UK transfer dataset."""
    dataset = create_policybench_transfer(
        source_file_path=source_file_path,
        fiscal_year=fiscal_year,
        exchange_rate=exchange_rate,
        max_rows=max_rows,
    )
    dataset.save(output_file_path)
    return dataset
