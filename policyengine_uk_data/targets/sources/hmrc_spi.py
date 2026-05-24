"""HMRC Survey of Personal Incomes targets.

Downloads and parses the SPI ODS (Tables 3.6 and 3.7) to get income
distributions by total income band and income type for 2023-24.

For future year projections, utils/incomes_projection.py starts from
the same parsed SPI band table and uprates it with PolicyEngine's
uprating factors.

Property income amounts are scaled up by 1.9x because the SPI only
covers taxpayers with "some liability to tax", missing ~half of all
landlord income. The HMRC Property Rental Income Statistics (2024)
show £46.68bn for 2020-21 vs SPI's ~£24.5bn.
See: https://www.gov.uk/government/statistics/property-rental-income-statistics
See also: https://github.com/PolicyEngine/policyengine-uk-data/issues/230

Source: https://www.gov.uk/government/statistics/income-tax-summarised-accounts-statistics
"""

import io
import logging
from functools import lru_cache
from pathlib import Path

import pandas as pd
import requests

from policyengine_uk_data.targets.schema import Target, Unit
from policyengine_uk_data.targets.sources._common import (
    HEADERS,
    STORAGE,
    load_config,
    to_float,
)

logger = logging.getLogger(__name__)

# Income bands in the SPI tables (lower bounds)
_BAND_LOWER = [
    12_570,
    15_000,
    20_000,
    30_000,
    40_000,
    50_000,
    70_000,
    100_000,
    150_000,
    200_000,
    300_000,
    500_000,
    1_000_000,
]
_BAND_UPPER = _BAND_LOWER[1:] + [float("inf")]

# SPI year: the ODS is for tax year 2023-24, mapped to calendar 2024
_SPI_YEAR = 2024

# HMRC Property Rental Income Statistics show ~1.9x more property income
# than the SPI (£46.68bn vs £24.5bn for 2020-21), because SPI only covers
# taxpayers with liability while the rental stats cover all ITSA landlords.
_PROPERTY_INCOME_SCALE = 1.9


@lru_cache(maxsize=1)
def _download_ods(url: str) -> bytes:
    """Download an ODS file."""
    r = requests.get(url, headers=HEADERS, allow_redirects=True, timeout=60)
    r.raise_for_status()
    return r.content


def _parse_table_36(ods_bytes: bytes) -> pd.DataFrame:
    """Parse Table 3.6: employment, self-employment, pensions by band.

    Columns: lower_bound, self_employment_income_count/amount,
    employment_income_count/amount, state_pension_count/amount,
    private_pension_income_count/amount.
    """
    df = pd.read_excel(
        io.BytesIO(ods_bytes),
        sheet_name="Table_3_6",
        engine="odf",
        header=None,
    )
    # Data rows start at row 5, end before "All ranges"
    data_rows = []
    for i in range(5, len(df)):
        lower = df.iloc[i, 0]
        if not isinstance(lower, (int, float)):
            break
        data_rows.append(
            {
                "lower_bound": int(lower),
                "self_employment_income_count": to_float(df.iloc[i, 1]),
                "self_employment_income_amount": to_float(df.iloc[i, 2]),
                "employment_income_count": to_float(df.iloc[i, 4]),
                "employment_income_amount": to_float(df.iloc[i, 5]),
                "state_pension_count": to_float(df.iloc[i, 7]),
                "state_pension_amount": to_float(df.iloc[i, 8]),
                "private_pension_income_count": to_float(df.iloc[i, 10]),
                "private_pension_income_amount": to_float(df.iloc[i, 11]),
            }
        )
    return pd.DataFrame(data_rows)


def _parse_table_37(ods_bytes: bytes) -> pd.DataFrame:
    """Parse Table 3.7: property, interest, dividends by band.

    Columns: lower_bound, property_income_count/amount,
    savings_interest_income_count/amount, dividend_income_count/amount.
    """
    df = pd.read_excel(
        io.BytesIO(ods_bytes),
        sheet_name="Table_3_7",
        engine="odf",
        header=None,
    )
    data_rows = []
    for i in range(5, len(df)):
        lower = df.iloc[i, 0]
        if not isinstance(lower, (int, float)):
            break
        data_rows.append(
            {
                "lower_bound": int(lower),
                "property_income_count": to_float(df.iloc[i, 1]),
                "property_income_amount": to_float(df.iloc[i, 2]),
                "savings_interest_income_count": to_float(df.iloc[i, 4]),
                "savings_interest_income_amount": to_float(df.iloc[i, 5]),
                "dividend_income_count": to_float(df.iloc[i, 7]),
                "dividend_income_amount": to_float(df.iloc[i, 8]),
            }
        )
    return pd.DataFrame(data_rows)


INCOME_VARIABLES = [
    "employment_income",
    "self_employment_income",
    "state_pension",
    "private_pension_income",
    "property_income",
    "dividend_income",
]

SPI_INCOME_TABLE_VARIABLES = [
    "employment_income",
    "self_employment_income",
    "state_pension",
    "private_pension_income",
    "property_income",
    "savings_interest_income",
    "dividend_income",
]


def _format_bound(value: float) -> str:
    if value == float("inf"):
        return "inf"
    return f"{float(value):_.0f}"


def _format_band_label(lower: float, upper: float) -> str:
    return f"{_format_bound(lower)}_to_{_format_bound(upper)}"


def _income_band_table_from_ods(ods_bytes: bytes) -> pd.DataFrame:
    """Parse the official SPI ODS into canonical income-band rows.

    Amounts are returned in GBP and counts in people, matching the CSV
    artifacts consumed by the projection and local-area calibration paths.
    Property income is intentionally unscaled here; target creation applies
    the rental-statistics scale factor at the point of use.
    """
    t36 = _parse_table_36(ods_bytes)
    t37 = _parse_table_37(ods_bytes)
    merged = t36.merge(t37, on="lower_bound", how="outer").sort_values("lower_bound")

    rows = []
    for idx, row in merged.reset_index(drop=True).iterrows():
        lower = int(row["lower_bound"])
        upper = _BAND_UPPER[idx] if idx < len(_BAND_UPPER) else float("inf")
        output = {
            "total_income_lower_bound": lower,
            "total_income_upper_bound": upper,
        }
        for variable in SPI_INCOME_TABLE_VARIABLES:
            count_col = f"{variable}_count"
            amount_col = f"{variable}_amount"
            if count_col in row.index:
                output[count_col] = float(row[count_col]) * 1e3
            if amount_col in row.index:
                output[amount_col] = float(row[amount_col]) * 1e6
        rows.append(output)

    table = pd.DataFrame(rows)
    aggregate = {
        "total_income_lower_bound": _BAND_LOWER[0],
        "total_income_upper_bound": float("inf"),
    }
    for variable in SPI_INCOME_TABLE_VARIABLES:
        for suffix in ("count", "amount"):
            column = f"{variable}_{suffix}"
            if column in table:
                aggregate[column] = table[column].sum()
    return pd.concat([table, pd.DataFrame([aggregate])], ignore_index=True)


def get_income_band_table(include_aggregate: bool = True) -> pd.DataFrame:
    """Return the current official SPI income-band table.

    The final aggregate row has lower bound 12,570 and upper bound infinity.
    It is useful for local-area scaling, but target generation drops it to
    avoid double-counting against the detailed bands.
    """
    config = load_config()
    ods_bytes = _download_ods(config["hmrc"]["spi_collated"])
    table = _income_band_table_from_ods(ods_bytes)
    if include_aggregate:
        return table
    is_aggregate = (table["total_income_lower_bound"] == _BAND_LOWER[0]) & (
        table["total_income_upper_bound"] == float("inf")
    )
    return table[~is_aggregate].reset_index(drop=True)


def get_targets() -> list[Target]:
    """Build income-band targets from the live HMRC SPI ODS.

    Also reads incomes_projection.csv if available, which contains
    projected future year data generated by the microsimulation.
    """
    config = load_config()
    ref = config["hmrc"]["spi_collated"]
    targets = []

    # Parse base year from official ODS
    try:
        merged = get_income_band_table(include_aggregate=False)

        for _, row in merged.iterrows():
            lower = float(row["total_income_lower_bound"])
            upper = float(row["total_income_upper_bound"])
            band_label = _format_band_label(lower, upper)

            for variable in INCOME_VARIABLES:
                amount_col = f"{variable}_amount"
                count_col = f"{variable}_count"

                if amount_col in row.index and row[amount_col] > 0:
                    amount = float(row[amount_col])
                    if variable == "property_income":
                        amount *= _PROPERTY_INCOME_SCALE
                    targets.append(
                        Target(
                            name=f"hmrc/{variable}_income_band_{band_label}",
                            variable=variable,
                            source="hmrc_spi",
                            unit=Unit.GBP,
                            values={_SPI_YEAR: amount},
                            breakdown_variable="total_income",
                            lower_bound=float(lower),
                            upper_bound=float(upper),
                            reference_url=ref,
                        )
                    )

                if count_col in row.index and row[count_col] > 0:
                    targets.append(
                        Target(
                            name=(f"hmrc/{variable}_count_income_band_{band_label}"),
                            variable=variable,
                            source="hmrc_spi",
                            unit=Unit.COUNT,
                            values={_SPI_YEAR: float(row[count_col])},
                            is_count=True,
                            breakdown_variable="total_income",
                            lower_bound=float(lower),
                            upper_bound=float(upper),
                            reference_url=ref,
                        )
                    )
    except Exception as e:
        logger.error("Failed to download/parse HMRC SPI ODS: %s", e)

    # Also read projected future years from incomes_projection.csv
    # if it exists (generated by utils/incomes_projection.py)
    proj_path = STORAGE / "incomes_projection.csv"
    if proj_path.exists():
        targets.extend(_read_projection_csv(proj_path, ref))

    return targets


def _read_projection_csv(csv_path: Path, ref: str) -> list[Target]:
    """Read projected future year targets from incomes_projection.csv."""
    incomes = pd.read_csv(csv_path)
    # Drop only the aggregate row (lower=12570, upper=inf). The detailed
    # top band also has upper=inf and must remain a calibration target.
    is_aggregate = (incomes["total_income_lower_bound"] == _BAND_LOWER[0]) & (
        incomes["total_income_upper_bound"] == float("inf")
    )
    incomes = incomes[~is_aggregate]
    targets = []

    for year in incomes.year.unique():
        if year <= _SPI_YEAR:
            continue  # Skip base year — we have actuals from ODS
        year_df = incomes[incomes.year == year]

        for _, row in year_df.iterrows():
            lower = row.total_income_lower_bound
            upper = row.total_income_upper_bound
            band_label = _format_band_label(lower, upper)

            for variable in INCOME_VARIABLES:
                amount_col = f"{variable}_amount"
                count_col = f"{variable}_count"

                if amount_col in row.index and pd.notna(row[amount_col]):
                    amount = float(row[amount_col])
                    if variable == "property_income":
                        amount *= _PROPERTY_INCOME_SCALE
                    name = f"hmrc/{variable}_income_band_{band_label}"
                    targets.append(
                        Target(
                            name=name,
                            variable=variable,
                            source="hmrc_spi",
                            unit=Unit.GBP,
                            values={int(year): amount},
                            breakdown_variable="total_income",
                            lower_bound=float(lower),
                            upper_bound=float(upper),
                            reference_url=ref,
                        )
                    )

                if count_col in row.index and pd.notna(row[count_col]):
                    name = f"hmrc/{variable}_count_income_band_{band_label}"
                    targets.append(
                        Target(
                            name=name,
                            variable=variable,
                            source="hmrc_spi",
                            unit=Unit.COUNT,
                            values={int(year): float(row[count_col])},
                            is_count=True,
                            breakdown_variable="total_income",
                            lower_bound=float(lower),
                            upper_bound=float(upper),
                            reference_url=ref,
                        )
                    )

    # Merge targets with the same name across years
    merged: dict[str, Target] = {}
    for t in targets:
        if t.name in merged:
            merged[t.name].values.update(t.values)
        else:
            merged[t.name] = t

    return list(merged.values())
