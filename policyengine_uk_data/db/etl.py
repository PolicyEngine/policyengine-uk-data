"""ETL: load all calibration targets into the SQLite database.

Populates three tables:
    areas          Geographic hierarchy from OA crosswalk + area code CSVs.
    targets        Target definitions from the registry + local area CSVs.
    target_values  Year-indexed values for each target.

Usage:
    python -m policyengine_uk_data.db.etl          # full rebuild
    python -m policyengine_uk_data.db.etl --areas   # areas only
    python -m policyengine_uk_data.db.etl --targets  # targets only
"""

import argparse
import logging
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from policyengine_uk_data.db.schema import get_connection, DB_PATH
from policyengine_uk_data.storage import STORAGE_FOLDER

logger = logging.getLogger(__name__)

# ── Area hierarchy ──────────────────────────────────────────────────

_COUNTRY_ROWS = [
    ("E92000001", "England", "country", None, "England"),
    ("W92000004", "Wales", "country", None, "Wales"),
    ("S92000003", "Scotland", "country", None, "Scotland"),
    ("N92000002", "Northern Ireland", "country", None, "Northern Ireland"),
]

_COUNTRY_PREFIX_MAP = {
    "England": "E92000001",
    "Wales": "W92000004",
    "Scotland": "S92000003",
    "Northern Ireland": "N92000002",
}


def load_areas(conn: sqlite3.Connection) -> int:
    """Populate the areas table from OA crosswalk and area code CSVs.

    Returns the number of rows inserted.
    """
    conn.execute("DELETE FROM areas")
    count = 0

    # Countries
    conn.executemany(
        "INSERT OR IGNORE INTO areas (code, name, level, parent_code, country) "
        "VALUES (?, ?, ?, ?, ?)",
        _COUNTRY_ROWS,
    )
    count += len(_COUNTRY_ROWS)

    # Constituencies (2024 boundaries)
    const_path = STORAGE_FOLDER / "constituencies_2024.csv"
    if const_path.exists():
        df = pd.read_csv(const_path)
        rows = []
        for _, r in df.iterrows():
            country = r.get("country", "")
            parent = _COUNTRY_PREFIX_MAP.get(country)
            rows.append((r["code"], r["name"], "constituency", parent, country))
        conn.executemany(
            "INSERT OR IGNORE INTO areas (code, name, level, parent_code, country) "
            "VALUES (?, ?, ?, ?, ?)",
            rows,
        )
        count += len(rows)
        logger.info("Loaded %d constituencies", len(rows))

    # OA crosswalk (provides region → LA → MSOA → LSOA → OA hierarchy)
    # Loaded before standalone LA CSV so we can parent LAs to regions.
    xw_path = STORAGE_FOLDER / "oa_crosswalk.csv.gz"
    la_to_region = {}  # LA code → region code (from crosswalk)
    if xw_path.exists():
        xw = pd.read_csv(xw_path, dtype=str)

        # Regions (parent: country)
        regions = xw[["region_code", "country"]].drop_duplicates()
        region_rows = []
        for _, r in regions.iterrows():
            code = r["region_code"]
            if pd.isna(code) or code == "":
                continue
            country = r["country"]
            parent = _COUNTRY_PREFIX_MAP.get(country)
            region_rows.append((code, "", "region", parent, country))
        conn.executemany(
            "INSERT OR IGNORE INTO areas (code, name, level, parent_code, country) "
            "VALUES (?, ?, ?, ?, ?)",
            region_rows,
        )
        count += len(region_rows)

        # Build LA → region lookup from crosswalk
        la_region = xw[["la_code", "region_code"]].drop_duplicates(subset=["la_code"])
        for _, r in la_region.iterrows():
            if pd.notna(r["la_code"]) and pd.notna(r["region_code"]):
                la_to_region[r["la_code"]] = r["region_code"]

    # Local authorities (parent: region, falling back to country)
    la_path = STORAGE_FOLDER / "local_authorities_2021.csv"
    if la_path.exists():
        df = pd.read_csv(la_path)
        rows = []
        for _, r in df.iterrows():
            code = str(r["code"])
            prefix = code[0]
            country_map = {
                "E": "England",
                "W": "Wales",
                "S": "Scotland",
                "N": "Northern Ireland",
            }
            country = country_map.get(prefix, "")
            # Parent to region if available, else country
            parent = la_to_region.get(code, _COUNTRY_PREFIX_MAP.get(country))
            rows.append((code, r.get("name", ""), "la", parent, country))
        conn.executemany(
            "INSERT OR IGNORE INTO areas (code, name, level, parent_code, country) "
            "VALUES (?, ?, ?, ?, ?)",
            rows,
        )
        count += len(rows)
        logger.info("Loaded %d local authorities", len(rows))

    # Sub-LA areas from crosswalk (MSOA → LSOA → OA)
    if xw_path.exists():
        # MSOAs
        msoas = xw[["msoa_code", "la_code", "country"]].drop_duplicates(
            subset=["msoa_code"]
        )
        msoa_rows = [
            (r["msoa_code"], "", "msoa", r["la_code"], r["country"])
            for _, r in msoas.iterrows()
            if pd.notna(r["msoa_code"]) and r["msoa_code"] != ""
        ]
        conn.executemany(
            "INSERT OR IGNORE INTO areas (code, name, level, parent_code, country) "
            "VALUES (?, ?, ?, ?, ?)",
            msoa_rows,
        )
        count += len(msoa_rows)
        logger.info("Loaded %d MSOAs", len(msoa_rows))

        # LSOAs
        lsoas = xw[["lsoa_code", "msoa_code", "country"]].drop_duplicates(
            subset=["lsoa_code"]
        )
        lsoa_rows = [
            (r["lsoa_code"], "", "lsoa", r["msoa_code"], r["country"])
            for _, r in lsoas.iterrows()
            if pd.notna(r["lsoa_code"]) and r["lsoa_code"] != ""
        ]
        conn.executemany(
            "INSERT OR IGNORE INTO areas (code, name, level, parent_code, country) "
            "VALUES (?, ?, ?, ?, ?)",
            lsoa_rows,
        )
        count += len(lsoa_rows)
        logger.info("Loaded %d LSOAs", len(lsoa_rows))

        # OAs
        oas = xw[["oa_code", "lsoa_code", "country"]].drop_duplicates(
            subset=["oa_code"]
        )
        oa_rows = [
            (r["oa_code"], "", "oa", r["lsoa_code"], r["country"])
            for _, r in oas.iterrows()
            if pd.notna(r["oa_code"]) and r["oa_code"] != ""
        ]
        conn.executemany(
            "INSERT OR IGNORE INTO areas (code, name, level, parent_code, country) "
            "VALUES (?, ?, ?, ?, ?)",
            oa_rows,
        )
        count += len(oa_rows)
        logger.info("Loaded %d OAs", len(oa_rows))
    else:
        logger.warning("OA crosswalk not found at %s — skipping sub-LA areas", xw_path)

    conn.commit()
    logger.info("Total areas loaded: %d", count)
    return count


# ── Targets ─────────────────────────────────────────────────────────


def _insert_target(
    conn: sqlite3.Connection,
    *,
    name: str,
    variable: str,
    source: str,
    unit: str,
    geographic_level: str,
    geo_code: str | None = None,
    geo_name: str | None = None,
    breakdown_variable: str | None = None,
    lower_bound: float | None = None,
    upper_bound: float | None = None,
    is_count: bool = False,
    reference_url: str | None = None,
    forecast_vintage: str | None = None,
    values: dict[int, float] | None = None,
) -> int:
    """Insert a target and its year-values. Returns the target ID."""
    cur = conn.execute(
        """INSERT OR REPLACE INTO targets
           (name, variable, source, unit, geographic_level, geo_code,
            geo_name, breakdown_variable, lower_bound, upper_bound,
            is_count, reference_url, forecast_vintage)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            name,
            variable,
            source,
            unit,
            geographic_level,
            geo_code,
            geo_name,
            breakdown_variable,
            lower_bound,
            upper_bound,
            int(is_count),
            reference_url,
            forecast_vintage,
        ),
    )
    target_id = cur.lastrowid
    if values:
        conn.executemany(
            "INSERT OR REPLACE INTO target_values (target_id, year, value) "
            "VALUES (?, ?, ?)",
            [(target_id, year, val) for year, val in values.items()],
        )
    return target_id


def load_registry_targets(conn: sqlite3.Connection) -> int:
    """Load all targets from the target registry into the database.

    These are the national/country/region targets discovered via
    ``get_all_targets()``.
    """
    from policyengine_uk_data.targets.registry import get_all_targets

    targets = get_all_targets()
    count = 0
    for t in targets:
        _insert_target(
            conn,
            name=t.name,
            variable=t.variable,
            source=t.source,
            unit=t.unit.value,
            geographic_level=t.geographic_level.value,
            geo_code=t.geo_code,
            geo_name=t.geo_name,
            breakdown_variable=t.breakdown_variable,
            lower_bound=t.lower_bound,
            upper_bound=t.upper_bound,
            is_count=t.is_count,
            reference_url=t.reference_url,
            forecast_vintage=t.forecast_vintage,
            values=t.values,
        )
        count += 1

    conn.commit()
    logger.info("Loaded %d registry targets", count)
    return count


def _load_local_age_targets(conn: sqlite3.Connection) -> int:
    """Load constituency and LA age targets from CSVs."""
    count = 0
    base = STORAGE_FOLDER.parent / "datasets" / "local_areas"

    for level, subdir in [
        ("constituency", "constituencies"),
        ("local_authority", "local_authorities"),
    ]:
        age_path = base / subdir / "targets" / "age.csv"
        if not age_path.exists():
            logger.warning("Age CSV not found: %s", age_path)
            continue

        df = pd.read_csv(age_path)
        # Aggregate single-year ages to 10-year bands
        for _, row in df.iterrows():
            code = row["code"]
            area_name = row.get("name", "")
            for lower in range(0, 80, 10):
                upper = lower + 10
                age_cols = [str(a) for a in range(lower, upper) if str(a) in df.columns]
                value = float(row[age_cols].sum())
                target_name = f"ons/{level}/{code}/age_{lower}_{upper}"
                _insert_target(
                    conn,
                    name=target_name,
                    variable="age",
                    source="ons",
                    unit="count",
                    geographic_level=level,
                    geo_code=code,
                    geo_name=area_name,
                    breakdown_variable="age",
                    lower_bound=float(lower),
                    upper_bound=float(upper),
                    is_count=True,
                    values={2025: value},
                )
                count += 1

    conn.commit()
    logger.info("Loaded %d local age targets", count)
    return count


def _load_local_income_targets(conn: sqlite3.Connection) -> int:
    """Load constituency and LA income targets from SPI CSVs."""
    count = 0
    base = STORAGE_FOLDER.parent / "datasets" / "local_areas"
    income_vars = ["self_employment_income", "employment_income"]

    for level, subdir, filename in [
        ("constituency", "constituencies", "spi_by_constituency.csv"),
        ("local_authority", "local_authorities", "spi_by_la.csv"),
    ]:
        path = base / subdir / "targets" / filename
        if not path.exists():
            logger.warning("SPI CSV not found: %s", path)
            continue

        df = pd.read_csv(path)
        for _, row in df.iterrows():
            code = row["code"]
            area_name = row.get("name", "")
            for var in income_vars:
                for suffix, unit, is_count in [
                    ("_amount", "gbp", False),
                    ("_count", "count", True),
                ]:
                    col = f"{var}{suffix}"
                    if col not in df.columns:
                        continue
                    value = float(row[col])
                    if np.isnan(value):
                        continue
                    target_name = f"hmrc/{level}/{code}/{var}{suffix}"
                    _insert_target(
                        conn,
                        name=target_name,
                        variable=var,
                        source="hmrc",
                        unit=unit,
                        geographic_level=level,
                        geo_code=code,
                        geo_name=area_name,
                        is_count=is_count,
                        reference_url="https://www.gov.uk/government/statistics/income-and-tax-by-county-and-region-and-by-parliamentary-constituency",
                        values={2025: value},
                    )
                    count += 1

    conn.commit()
    logger.info("Loaded %d local income targets", count)
    return count


def _load_local_uc_targets(conn: sqlite3.Connection) -> int:
    """Load UC household targets from XLSX files."""
    count = 0

    for level, xlsx_name, loader_attr in [
        ("constituency", "uc_pc_households.xlsx", "uc_pc_households"),
        ("local_authority", "uc_la_households.xlsx", "uc_la_households"),
    ]:
        path = STORAGE_FOLDER / xlsx_name
        if not path.exists():
            logger.warning("UC XLSX not found: %s", path)
            continue

        from policyengine_uk_data.utils import uc_data

        df = getattr(uc_data, loader_attr)

        # Get area codes for alignment
        if level == "constituency":
            codes_df = pd.read_csv(STORAGE_FOLDER / "constituencies_2024.csv")
        else:
            codes_df = pd.read_csv(STORAGE_FOLDER / "local_authorities_2021.csv")

        for i, hh_count in enumerate(df.household_count.values):
            if i >= len(codes_df):
                break
            code = codes_df.iloc[i]["code"]
            area_name = codes_df.iloc[i].get("name", "")
            value = float(hh_count)
            if np.isnan(value):
                continue
            target_name = f"dwp/{level}/{code}/uc_households"
            _insert_target(
                conn,
                name=target_name,
                variable="universal_credit",
                source="dwp",
                unit="count",
                geographic_level=level,
                geo_code=code,
                geo_name=area_name,
                is_count=True,
                reference_url="https://stat-xplore.dwp.gov.uk",
                values={2025: value},
            )
            count += 1

    conn.commit()
    logger.info("Loaded %d local UC targets", count)
    return count


def _load_la_extras(conn: sqlite3.Connection) -> int:
    """Load LA-only targets: ONS income, tenure, private rent."""
    from policyengine_uk_data.targets.sources.local_la_extras import (
        load_ons_la_income,
        load_household_counts,
        load_tenure_data,
        load_private_rents,
    )

    count = 0

    ons_income = load_ons_la_income()
    if not ons_income.empty:
        for _, row in ons_income.iterrows():
            code = row["la_code"]
            for col in ["total_income", "net_income_bhc", "net_income_ahc"]:
                val = row.get(col)
                if val is None or np.isnan(val):
                    continue
                _insert_target(
                    conn,
                    name=f"ons/la/{code}/{col}",
                    variable=col,
                    source="ons",
                    unit="gbp",
                    geographic_level="local_authority",
                    geo_code=code,
                    values={2020: float(val)},
                )
                count += 1

    households = load_household_counts()
    if not households.empty:
        for _, row in households.iterrows():
            code = row["la_code"]
            val = row["households"]
            if np.isnan(val):
                continue
            _insert_target(
                conn,
                name=f"ons/la/{code}/households",
                variable="households",
                source="ons",
                unit="count",
                geographic_level="local_authority",
                geo_code=code,
                is_count=True,
                values={2021: float(val)},
            )
            count += 1

    tenure = load_tenure_data()
    if not tenure.empty:
        for _, row in tenure.iterrows():
            code = row["la_code"]
            for col in [
                "owned_outright_pct",
                "owned_mortgage_pct",
                "private_rent_pct",
                "social_rent_pct",
            ]:
                val = row.get(col)
                if val is None or np.isnan(val):
                    continue
                _insert_target(
                    conn,
                    name=f"ehs/la/{code}/{col}",
                    variable=col.replace("_pct", ""),
                    source="ehs",
                    unit="rate",
                    geographic_level="local_authority",
                    geo_code=code,
                    values={2023: float(val)},
                )
                count += 1

    rents = load_private_rents()
    if not rents.empty:
        for _, row in rents.iterrows():
            code = row["area_code"]
            val = row["median_annual_rent"]
            if np.isnan(val):
                continue
            _insert_target(
                conn,
                name=f"voa/la/{code}/median_annual_rent",
                variable="median_annual_rent",
                source="voa",
                unit="gbp",
                geographic_level="local_authority",
                geo_code=code,
                values={2024: float(val)},
            )
            count += 1

    conn.commit()
    logger.info("Loaded %d LA extra targets", count)
    return count


def load_all_targets(conn: sqlite3.Connection) -> int:
    """Load all targets (registry + local CSVs) into the database."""
    conn.execute("DELETE FROM target_values")
    conn.execute("DELETE FROM targets")
    conn.commit()

    count = 0
    count += load_registry_targets(conn)
    count += _load_local_age_targets(conn)
    count += _load_local_income_targets(conn)
    count += _load_local_uc_targets(conn)
    count += _load_la_extras(conn)
    logger.info("Total targets loaded: %d", count)
    return count


def build_database(db_path: Path | None = None) -> Path:
    """Full rebuild: create schema, load areas, load targets."""
    path = db_path or DB_PATH
    conn = get_connection(path)
    load_areas(conn)
    load_all_targets(conn)
    conn.close()
    logger.info("Database built at %s", path)
    return path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Build the target SQLite database.")
    parser.add_argument("--areas", action="store_true", help="Load areas only.")
    parser.add_argument("--targets", action="store_true", help="Load targets only.")
    parser.add_argument("--db", type=Path, help="Override database path.")
    args = parser.parse_args()

    path = args.db or DB_PATH
    conn = get_connection(path)

    if args.areas:
        load_areas(conn)
    elif args.targets:
        load_all_targets(conn)
    else:
        load_areas(conn)
        load_all_targets(conn)

    conn.close()
    logger.info("Done: %s", path)
