import numpy as np
import pandas as pd
from pathlib import Path


def parse_monthly_award_band(band: str) -> tuple[float, float]:
    """Annual (lower, upper] payment bounds of a Stat-Xplore monthly award band.

    Awards are whole pence, so the band '£100.01 to £200.00' holds monthly
    awards over £100.00 and up to £200.00: annual bounds (1,200, 2,400]. The
    lower bound is the previous band's top, so consecutive bands meet with no
    gap. The open top band '£2500.01 or over' is (30,000, inf).
    """
    text = band.replace("£", "").replace(",", "").strip()
    if text.endswith(" or over"):
        lower = float(text.removesuffix(" or over"))
        return round((lower - 0.01) * 12, 2), np.inf
    parts = text.split(" to ")
    if len(parts) != 2:
        raise ValueError(f"Unrecognised UC monthly award band: {band!r}")
    lower, upper = (float(part) for part in parts)
    return round((lower - 0.01) * 12, 2), upper * 12


def _check_bands_disjoint(bands: pd.DataFrame) -> None:
    """Fail if any family type's payment bands overlap.

    Stat-Xplore also lists summary bands ('£1500.01 or over') that span the
    finer bands below them. They are suppressed ('..') in the committed
    extract; if a new extract filled one in, counting it as well would double
    count those households.
    """
    for family_type, group in bands.groupby("family_type"):
        group = group.sort_values("uc_annual_payment_min")
        lower = group.uc_annual_payment_min.to_numpy()
        upper = group.uc_annual_payment_max.to_numpy()
        if (lower[1:] < upper[:-1]).any():
            raise ValueError(
                f"UC payment bands for {family_type} overlap: check for "
                "summary bands spanning finer ones"
            )


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

    result_df[["uc_annual_payment_min", "uc_annual_payment_max"]] = [
        parse_monthly_award_band(band) for band in result_df["monthly_award_band"]
    ]
    _check_bands_disjoint(result_df)

    # Map family types to constant names
    family_type_mapping = {
        "Single, no children": "SINGLE",
        "Single, with children": "LONE_PARENT",
        "Couple, no children": "COUPLE_NO_CHILDREN",
        "Couple, with children": "COUPLE_WITH_CHILDREN",
    }
    result_df["family_type"] = result_df["family_type"].map(family_type_mapping)

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
    df_ni = pd.read_excel(ni_file_path, sheet_name="5b", engine="odf", header=None)

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


def _parse_uc_la_households():
    """Parse UC local authority households (GB + NI)."""
    storage_path = Path(__file__).parent.parent / "storage"

    # Parse GB data
    gb_file_path = storage_path / "uc_la_households.xlsx"
    df_gb = pd.read_excel(gb_file_path, header=None)

    gb_data_rows = []

    for idx in range(8, len(df_gb)):
        la_name = df_gb.iloc[idx, 2]  # Column 2: LA name
        household_count = df_gb.iloc[idx, 3]  # Column 3: household count

        # Skip if empty, invalid, Total row, or Unknown
        if (
            pd.isna(la_name)
            or pd.isna(household_count)
            or la_name in ["Total", "Unknown"]
        ):
            continue

        gb_data_rows.append(
            {
                "la_name": la_name,
                "household_count": int(household_count),
            }
        )

    # Parse NI data
    ni_file_path = storage_path / "dfc-ni-uc-stats-supp-tables-may-2025.ods"
    df_ni = pd.read_excel(ni_file_path, sheet_name="5c", engine="odf", header=None)

    # Get LGD names from row 2, columns 1-11
    ni_lgd_names = df_ni.iloc[2, 1:12].tolist()

    # Find May 2025 row
    may_2025_row = df_ni[df_ni[0] == "May 2025"].iloc[0]

    ni_data_rows = []
    for col_idx, lgd_name in enumerate(ni_lgd_names, start=1):
        if pd.notna(lgd_name):
            household_count = may_2025_row[col_idx]

            # Skip Ards and North Down to match target datasets (they have 10 NI LGDs, not 11)
            if lgd_name == "Ards and North Down":
                continue

            if pd.notna(household_count) and household_count != 0:
                ni_data_rows.append(
                    {
                        "la_name": lgd_name,
                        "household_count": int(household_count),
                    }
                )

    # Combine GB and NI data
    result_df = pd.DataFrame(gb_data_rows + ni_data_rows)

    # Scale LA counts to match national total
    national_total = _parse_uc_national_payment_dist()["household_count"].sum()
    la_total = result_df["household_count"].sum()
    scaling_factor = national_total / la_total

    result_df["household_count"] = (
        (result_df["household_count"] * scaling_factor).round().astype(int)
    )

    return result_df


# Module-level dataframes for easy import
uc_national_payment_dist = _parse_uc_national_payment_dist()
uc_pc_households = _parse_uc_pc_households()
uc_la_households = _parse_uc_la_households()
