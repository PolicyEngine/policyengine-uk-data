import pandas as pd
from pathlib import Path


def _parse_uc_national_payment_dist():
    """Parse UC national payment distribution into long format."""
    storage_path = Path(__file__).parent.parent / "storage"
    file_path = storage_path / "uc_national_payment_dist.xlsx"

    # Read the Excel file, skipping header rows
    df = pd.read_excel(file_path, header=None)

    # Extract family types from row 7 (index 7)
    family_types = df.iloc[7, 3:7].tolist()  # Columns 3-6: the 4 family types

    # Extract data rows (starting from row 9, index 9)
    data_rows = []

    for idx in range(9, len(df)):
        award_band = df.iloc[idx, 1]  # Monthly award amount band

        # Skip if not a valid award band
        if pd.isna(award_band) or award_band in ["No payment", "Total"]:
            continue

        for col_idx, family_type in enumerate(family_types, start=3):
            household_count = df.iloc[idx, col_idx]

            # Skip missing, ".." (suppressed), or zero values
            if (
                pd.isna(household_count)
                or household_count == ".."
                or household_count == 0
            ):
                continue

            data_rows.append(
                {
                    "monthly_award_band": award_band,
                    "family_type": family_type,
                    "household_count": int(household_count),
                }
            )

    result_df = pd.DataFrame(data_rows)

    # Parse monthly band into min and max, then convert to annual
    def parse_band(band):
        """Parse band like '£100.01 to £200.00' into (min, max)."""
        parts = band.replace("£", "").replace(",", "").split(" to ")
        if len(parts) == 2:
            return float(parts[0]) * 12, float(parts[1]) * 12
        return None, None

    result_df[["uc_annual_payment_min", "uc_annual_payment_max"]] = result_df[
        "monthly_award_band"
    ].apply(lambda x: pd.Series(parse_band(x)))

    # Map family types to constant names
    family_type_mapping = {
        "Single, no children": "SINGLE",
        "Single, with children": "LONE_PARENT",
        "Couple, no children": "COUPLE_NO_CHILDREN",
        "Couple, with children": "COUPLE_WITH_CHILDREN",
    }
    result_df["family_type"] = result_df["family_type"].map(
        family_type_mapping
    )

    # Reorder columns and drop monthly band
    result_df = result_df[
        [
            "uc_annual_payment_min",
            "uc_annual_payment_max",
            "family_type",
            "household_count",
        ]
    ]

    return result_df


def _parse_uc_pc_households():
    """Parse UC parliamentary constituency households (GB + NI)."""
    storage_path = Path(__file__).parent.parent / "storage"

    # Parse GB data
    gb_file_path = storage_path / "uc_pc_households.xlsx"
    df_gb = pd.read_excel(gb_file_path, header=None)

    gb_data_rows = []

    for idx in range(8, len(df_gb)):
        constituency = df_gb.iloc[idx, 1]  # Column 1: constituency name
        household_count = df_gb.iloc[idx, 3]  # Column 3: household count

        # Skip if empty, invalid, Total row, or Unknown
        if (
            pd.isna(constituency)
            or pd.isna(household_count)
            or constituency in ["Total", "Unknown"]
        ):
            continue

        gb_data_rows.append(
            {
                "constituency_name": constituency,
                "household_count": int(household_count),
            }
        )

    # Parse NI data
    ni_file_path = storage_path / "dfc-ni-uc-stats-supp-tables-may-2025.ods"
    df_ni = pd.read_excel(
        ni_file_path, sheet_name="5b", engine="odf", header=None
    )

    # Get constituency names from row 2, columns 1-18
    ni_constituencies = df_ni.iloc[2, 1:19].tolist()

    # Find May 2025 row
    may_2025_row = df_ni[df_ni[0] == "May 2025"].iloc[0]

    ni_data_rows = []
    for col_idx, constituency_name in enumerate(ni_constituencies, start=1):
        household_count = may_2025_row[col_idx]

        if pd.notna(household_count) and household_count != 0:
            ni_data_rows.append(
                {
                    "constituency_name": constituency_name,
                    "household_count": int(household_count),
                }
            )

    # Combine GB and NI data
    result_df = pd.DataFrame(gb_data_rows + ni_data_rows)

    # Scale constituency counts to match national total
    national_total = _parse_uc_national_payment_dist()["household_count"].sum()
    constituency_total = result_df["household_count"].sum()
    scaling_factor = national_total / constituency_total

    result_df["household_count"] = (
        (result_df["household_count"] * scaling_factor).round().astype(int)
    )

    return result_df


# Module-level dataframes for easy import
uc_national_payment_dist = _parse_uc_national_payment_dist()
uc_pc_households = _parse_uc_pc_households()
