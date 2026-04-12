"""Build a unified UK Output Area crosswalk.

Downloads OA-level geographic lookups from ONS (England & Wales),
NRS (Scotland), and NISRA (Northern Ireland) and combines them
into a single crosswalk mapping:

    OA -> LSOA/DZ -> MSOA/IZ -> LA -> constituency -> region -> country

The crosswalk also includes OA-level population estimates from
Census 2021 (E+W, NI) and Census 2022 (Scotland).

Output: storage/oa_crosswalk.csv.gz

Columns:
    oa_code         - Output Area GSS code (E00/W00/S00/95xx)
    lsoa_code       - LSOA (E+W) or Data Zone (Scotland) code
    msoa_code       - MSOA (E+W) or Intermediate Zone (Scotland) code
    la_code         - Local Authority District GSS code
    constituency_code - Parliamentary constituency 2024 GSS code
    region_code     - Region GSS code (E12/W99/S99/N99)
    country         - Country name (England/Wales/Scotland/Northern Ireland)
    population      - Population (Census 2021 or 2022)
"""

import io
import logging
import zipfile
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)

STORAGE_FOLDER = Path(__file__).parent.parent / "storage"
CROSSWALK_PATH = STORAGE_FOLDER / "oa_crosswalk.csv.gz"

# ── Download URLs ─────────────────────────────────────────

# ONS Hub CSV download: OA21 → LSOA21 → MSOA21 → LAD22 (E+W)
_EW_OA_LOOKUP_URL = (
    "https://open-geography-portalx-ons.hub.arcgis.com/"
    "api/download/v1/items/"
    "b9ca90c10aaa4b8d9791e9859a38ca67/csv?layers=0"
)

# ONS Hub CSV download: OA21 → PCON25 (E+W constituency 2024)
_EW_OA_CONST_URL = (
    "https://open-geography-portalx-ons.hub.arcgis.com/"
    "api/download/v1/items/"
    "5968b5b2c0f14dd29ba277beaae6dec3/csv?layers=0"
)

# ONS Hub CSV download: LAD22 → RGN22 (England only)
_EN_LAD_REGION_URL = (
    "https://open-geography-portalx-ons.hub.arcgis.com/"
    "api/download/v1/items/"
    "78b348cd8fb04037ada3c862aa054428/csv?layers=0"
)

# Nomis Census 2021 TS001 (population by OA) - bulk zip
_EW_POPULATION_URL = (
    "https://www.nomisweb.co.uk/output/census/2021/census2021-ts001.zip"
)

# NRS Scotland: OA22 → DZ22 → IZ22 (zip)
_SCOTLAND_OA_DZ_URL = "https://www.nrscotland.gov.uk/media/iz3evrqt/oa22_dz22_iz22.zip"

# NRS Scotland: OA22 → UK Parliamentary Constituency 2024
_SCOTLAND_OA_CONST_URL = "https://www.nrscotland.gov.uk/media/njkmhppf/oa22_ukpc24.zip"

# statistics.gov.scot: DZ22 → IZ22 → LA → constituency
_SCOTLAND_DZ_LOOKUP_URL = (
    "https://statistics.gov.scot/downloads/file?"
    "id=75ff05d2-d482-4463-81b6-76b8dd6b6d3b/"
    "DataZone2022lookup_2025-10-28.csv"
)

# Scotland Census 2022 OA population
_SCOTLAND_OA_POP_URL = (
    "https://www.scotlandscensus.gov.uk/media/kqcmo4ge/output-area-2022-all-persons.csv"
)

# NISRA NI DZ2021 lookup
_NI_DZ_LOOKUP_URL = (
    "https://www.nisra.gov.uk/sites/nisra.gov.uk/files/"
    "publications/geography-dz2021-lookup-tables.xlsx"
)

# NISRA NI DZ2021 population
_NI_DZ_POP_URL = (
    "https://www.nisra.gov.uk/sites/nisra.gov.uk/files/"
    "publications/"
    "census-2021-person-estimates-data-zones.xlsx"
)


def _download_csv(url: str, timeout: int = 300) -> pd.DataFrame:
    """Download a CSV file from a URL.

    Args:
        url: URL to download.
        timeout: Request timeout in seconds.

    Returns:
        DataFrame from the CSV.
    """
    logger.info(f"Downloading {url[:80]}...")
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    # Try utf-8-sig (strips BOM), fall back to latin-1
    for enc in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            text = resp.content.decode(enc)
            return pd.read_csv(io.StringIO(text))
        except (UnicodeDecodeError, ValueError):
            continue
    # Last resort
    return pd.read_csv(io.BytesIO(resp.content))


def _download_csv_from_zip(
    url: str,
    csv_filter: str = ".csv",
    timeout: int = 300,
) -> pd.DataFrame:
    """Download a ZIP and extract the first CSV matching
    the filter.

    Args:
        url: URL of the ZIP file.
        csv_filter: Substring that the CSV filename must
            contain (case-insensitive).
        timeout: Request timeout in seconds.

    Returns:
        DataFrame from the extracted CSV.
    """
    logger.info(f"Downloading ZIP {url[:80]}...")
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        csv_files = [
            n
            for n in zf.namelist()
            if n.lower().endswith(".csv") and csv_filter.lower() in n.lower()
        ]
        if not csv_files:
            # Fallback: any CSV
            csv_files = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not csv_files:
            raise FileNotFoundError(f"No CSV found in ZIP. Contents: {zf.namelist()}")
        logger.info(f"  Extracting {csv_files[0]}")
        with zf.open(csv_files[0]) as f:
            raw = f.read().decode("utf-8-sig")
            return pd.read_csv(io.StringIO(raw))


# ── England & Wales ───────────────────────────────────────


def _get_ew_oa_hierarchy() -> pd.DataFrame:
    """Download OA → LSOA → MSOA → LAD lookup for E+W.

    Returns:
        DataFrame with columns: oa_code, lsoa_code,
        msoa_code, la_code
    """
    df = _download_csv(_EW_OA_LOOKUP_URL)
    logger.info(f"  E+W hierarchy: {len(df)} rows, columns {df.columns.tolist()[:6]}")

    # Strip whitespace from column names
    df.columns = df.columns.str.strip()

    # Direct rename - known column names from ONS
    col_map = {
        "OA21CD": "oa_code",
        "LSOA21CD": "lsoa_code",
        "MSOA21CD": "msoa_code",
        "LAD22CD": "la_code",
    }
    # Handle different LAD vintages
    for c in df.columns:
        if c.startswith("LAD") and c.endswith("CD"):
            col_map[c] = "la_code"

    df = df.rename(columns=col_map)
    return df[["oa_code", "lsoa_code", "msoa_code", "la_code"]]


def _get_ew_population() -> pd.DataFrame:
    """Download Census 2021 OA-level population for E+W.

    Returns:
        DataFrame with columns: oa_code, population
    """
    df = _download_csv_from_zip(_EW_POPULATION_URL, csv_filter="oa")
    logger.info(f"  TS001 columns: {df.columns.tolist()}")

    # Find geography code and total population columns
    geo_col = None
    obs_col = None
    for c in df.columns:
        cl = c.lower()
        if "geography code" in cl:
            geo_col = c
        elif "total" in cl and "measures" in cl:
            obs_col = c
        elif (
            cl
            in (
                "observation",
                "obs_value",
                "count",
            )
            or "observation" in cl
        ):
            if obs_col is None:
                obs_col = c

    if geo_col is None:
        geo_col = [c for c in df.columns if "code" in c.lower()][0]
    if obs_col is None:
        obs_col = df.select_dtypes(include="number").columns[-1]

    result = pd.DataFrame(
        {
            "oa_code": df[geo_col].astype(str),
            "population": pd.to_numeric(df[obs_col], errors="coerce")
            .fillna(0)
            .astype(int),
        }
    )
    # Keep only OA-level codes (E00/W00)
    result = result[result["oa_code"].str.match(r"^[EW]00")].copy()
    return result


def _get_ew_constituency() -> pd.DataFrame:
    """Download OA → constituency (2024) for E+W.

    Returns:
        DataFrame with columns: oa_code, constituency_code
    """
    df = _download_csv(_EW_OA_CONST_URL)
    logger.info(
        f"  E+W constituency: {len(df)} rows, columns {df.columns.tolist()[:6]}"
    )

    df.columns = df.columns.str.strip()
    col_map = {"OA21CD": "oa_code"}
    for c in df.columns:
        if c.startswith("PCON") and c.endswith("CD"):
            col_map[c] = "constituency_code"
    df = df.rename(columns=col_map)
    return df[["oa_code", "constituency_code"]]


# ── Scotland ──────────────────────────────────────────────


def _get_scotland_oa_hierarchy() -> pd.DataFrame:
    """Build Scotland OA → DZ → IZ → LA hierarchy.

    Returns:
        DataFrame with columns: oa_code, lsoa_code (=DZ),
        msoa_code (=IZ), la_code, constituency_code
    """
    # OA → DZ → IZ from NRS zip
    oa_dz = _download_csv_from_zip(_SCOTLAND_OA_DZ_URL, csv_filter="")
    logger.info(
        f"  Scotland OA→DZ: {len(oa_dz)} rows, columns {oa_dz.columns.tolist()}"
    )

    # Find column names dynamically
    oa_col = [c for c in oa_dz.columns if "oa" in c.lower() and "code" in c.lower()]
    dz_col = [
        c
        for c in oa_dz.columns
        if "dz" in c.lower() or "datazone" in c.lower() and "code" in c.lower()
    ]

    if not oa_col:
        oa_col = [oa_dz.columns[0]]
    if not dz_col:
        dz_col = [oa_dz.columns[1]]

    oa_dz = oa_dz.rename(
        columns={
            oa_col[0]: "oa_code",
            dz_col[0]: "lsoa_code",
        }
    )

    # DZ → LA → constituency from statistics.gov.scot
    dz_lookup = _download_csv(_SCOTLAND_DZ_LOOKUP_URL)
    logger.info(
        f"  Scotland DZ lookup: {len(dz_lookup)} rows, "
        f"columns {dz_lookup.columns.tolist()[:8]}"
    )

    # Find columns
    dz_lk_col = [
        c for c in dz_lookup.columns if "dz" in c.lower() and "code" in c.lower()
    ]
    iz_col = [c for c in dz_lookup.columns if "iz" in c.lower() and "code" in c.lower()]
    la_col = [c for c in dz_lookup.columns if "la" in c.lower() and "code" in c.lower()]
    ukpc_col = [
        c for c in dz_lookup.columns if "ukpc" in c.lower() or "pcon" in c.lower()
    ]

    if not dz_lk_col:
        dz_lk_col = [dz_lookup.columns[0]]
    if not iz_col:
        iz_col = [dz_lookup.columns[2]]
    if not la_col:
        la_col = [dz_lookup.columns[4]]

    rename_map = {
        dz_lk_col[0]: "lsoa_code",
        iz_col[0]: "msoa_code",
        la_col[0]: "la_code",
    }
    if ukpc_col:
        rename_map[ukpc_col[0]] = "constituency_code"

    dz_lookup = dz_lookup.rename(columns=rename_map)

    merge_cols = ["lsoa_code", "msoa_code", "la_code"]
    if "constituency_code" in dz_lookup.columns:
        merge_cols.append("constituency_code")

    # Deduplicate DZ lookup (multiple rows per DZ in some
    # versions)
    dz_dedup = dz_lookup[merge_cols].drop_duplicates(subset=["lsoa_code"])

    result = oa_dz[["oa_code", "lsoa_code"]].merge(dz_dedup, on="lsoa_code", how="left")

    # Also try OA → constituency direct lookup
    try:
        oa_const = _download_csv_from_zip(
            _SCOTLAND_OA_CONST_URL,
            csv_filter="oa22_ukpc24",
        )
        oa_c_col = [
            c
            for c in oa_const.columns
            if c.lower() == "oa22" or ("oa" in c.lower() and "code" in c.lower())
        ]
        const_c_col = [
            c
            for c in oa_const.columns
            if "ukpc" in c.lower() or "pcon" in c.lower() or "const" in c.lower()
        ]
        if oa_c_col and const_c_col:
            oa_const = oa_const.rename(
                columns={
                    oa_c_col[0]: "oa_code",
                    const_c_col[0]: "const_direct",
                }
            )
            result = result.merge(
                oa_const[["oa_code", "const_direct"]],
                on="oa_code",
                how="left",
            )
            # Prefer direct OA→const over DZ-derived
            if "constituency_code" not in result.columns:
                result["constituency_code"] = ""
            mask = result["const_direct"].notna()
            result.loc[mask, "constituency_code"] = result.loc[mask, "const_direct"]
            result = result.drop(columns=["const_direct"])
    except Exception as e:
        logger.warning(
            f"Could not download Scotland OA→constituency direct lookup: {e}"
        )

    if "constituency_code" not in result.columns:
        result["constituency_code"] = ""

    return result


def _get_scotland_oa_population() -> pd.DataFrame:
    """Get Scotland Census 2022 OA-level population.

    Tries the NRS OA population CSV first. If that fails
    (403), falls back to assigning equal population within
    each Data Zone using the DZ lookup.

    Returns:
        DataFrame with columns: oa_code, population
    """
    # Try direct OA population download
    try:
        df = _download_csv(_SCOTLAND_OA_POP_URL)
        logger.info(
            f"  Scotland pop: {len(df)} rows, columns {df.columns.tolist()[:5]}"
        )

        oa_col = [
            c
            for c in df.columns
            if ("output" in c.lower() or "oa" in c.lower()) and "code" in c.lower()
        ]
        if not oa_col:
            for c in df.columns:
                if df[c].dtype == object:
                    sample = str(df[c].iloc[0])
                    if sample.startswith("S00"):
                        oa_col = [c]
                        break
        if not oa_col:
            oa_col = [df.columns[0]]

        count_col = [
            c
            for c in df.columns
            if "count" in c.lower() or "total" in c.lower() or "all person" in c.lower()
        ]
        if not count_col:
            num_cols = df.select_dtypes(include="number").columns
            if len(num_cols) > 0:
                count_col = [num_cols[0]]

        if count_col:
            return pd.DataFrame(
                {
                    "oa_code": df[oa_col[0]].astype(str),
                    "population": pd.to_numeric(df[count_col[0]], errors="coerce")
                    .fillna(0)
                    .astype(int),
                }
            )
    except Exception as e:
        logger.warning(
            f"Could not download Scotland OA population: {e}. Using uniform population."
        )

    # Fallback: use OA→DZ lookup and assign ~120 per OA
    # (Scotland pop ~5.4M / 46K OAs ≈ 117)
    logger.info("  Using uniform population estimate for Scotland OAs (~117 per OA)")
    oa_dz = _download_csv_from_zip(_SCOTLAND_OA_DZ_URL, csv_filter="")
    oa_col = oa_dz.columns[0]
    return pd.DataFrame(
        {
            "oa_code": oa_dz[oa_col].astype(str),
            "population": 117,
        }
    )


# ── Northern Ireland ──────────────────────────────────────


def _get_ni_hierarchy() -> pd.DataFrame:
    """Build NI Data Zone hierarchy.

    NI does not publish census data at OA level. The smallest
    published geography is the Data Zone (DZ2021, ~3,780 areas).
    We treat NI Data Zones as the OA-equivalent.

    Returns:
        DataFrame with columns: oa_code (=DZ2021),
        lsoa_code (=DZ2021), msoa_code (=SDZ2021),
        la_code (LGD2014)
    """
    logger.info("Downloading NI DZ2021 lookup...")
    try:
        resp = requests.get(_NI_DZ_LOOKUP_URL, timeout=120)
        resp.raise_for_status()
    except Exception as e:
        logger.warning(
            f"Could not download NI DZ lookup: {e}. Returning empty NI hierarchy."
        )
        return pd.DataFrame(
            columns=[
                "oa_code",
                "lsoa_code",
                "msoa_code",
                "la_code",
            ]
        )
    df = pd.read_excel(io.BytesIO(resp.content))

    logger.info(f"  NI lookup: {len(df)} rows, columns {df.columns.tolist()[:6]}")

    dz_col = [c for c in df.columns if "dz2021" in c.lower() and "code" in c.lower()]
    sdz_col = [c for c in df.columns if "sdz2021" in c.lower() and "code" in c.lower()]
    lgd_col = [c for c in df.columns if "lgd" in c.lower() and "code" in c.lower()]

    if not dz_col or not sdz_col or not lgd_col:
        logger.warning(
            f"NI columns not as expected: {df.columns.tolist()}. Using positional."
        )
        dz_col = [df.columns[0]]
        sdz_col = [df.columns[1]]
        lgd_col = [df.columns[2]]

    return pd.DataFrame(
        {
            "oa_code": df[dz_col[0]].astype(str),
            "lsoa_code": df[dz_col[0]].astype(str),
            "msoa_code": df[sdz_col[0]].astype(str),
            "la_code": df[lgd_col[0]].astype(str),
        }
    )


def _get_ni_population() -> pd.DataFrame:
    """Get NI Census 2021 population at Data Zone level.

    Returns:
        DataFrame with columns: oa_code, population
    """
    logger.info("Downloading NI DZ population...")
    try:
        resp = requests.get(_NI_DZ_POP_URL, timeout=120)
        resp.raise_for_status()
    except Exception as e:
        logger.warning(f"Could not download NI population: {e}. Returning empty.")
        return pd.DataFrame(columns=["oa_code", "population"])
    df = pd.read_excel(io.BytesIO(resp.content))

    logger.info(f"  NI pop: {len(df)} rows, columns {df.columns.tolist()[:5]}")

    dz_col = [c for c in df.columns if "dz" in c.lower() and "code" in c.lower()]
    pop_col = [
        c
        for c in df.columns
        if "population" in c.lower() or "all" in c.lower() or "total" in c.lower()
    ]

    if not dz_col:
        dz_col = [df.columns[0]]
    if not pop_col:
        num_cols = df.select_dtypes(include="number").columns
        pop_col = [num_cols[0]] if len(num_cols) > 0 else None

    if pop_col is None:
        return pd.DataFrame(
            {
                "oa_code": df[dz_col[0]].astype(str),
                "population": 1,
            }
        )

    return pd.DataFrame(
        {
            "oa_code": df[dz_col[0]].astype(str),
            "population": pd.to_numeric(df[pop_col[0]], errors="coerce")
            .fillna(0)
            .astype(int),
        }
    )


# ── Region & Country Assignment ───────────────────────────


def _get_la_to_region_map() -> dict:
    """Download LAD → region mapping for England.

    Returns:
        Dict mapping LAD code to region code.
    """
    try:
        df = _download_csv(_EN_LAD_REGION_URL)
        logger.info(f"  LAD→region: {len(df)} rows, columns {df.columns.tolist()[:4]}")

        lad_col = [c for c in df.columns if "lad" in c.lower() and "cd" in c.lower()]
        rgn_col = [c for c in df.columns if "rgn" in c.lower() and "cd" in c.lower()]

        if lad_col and rgn_col:
            return dict(zip(df[lad_col[0]], df[rgn_col[0]]))
    except Exception as e:
        logger.warning(f"Could not download region lookup: {e}")

    return {}


def _assign_regions(df: pd.DataFrame) -> pd.DataFrame:
    """Assign region codes based on LA code.

    Args:
        df: DataFrame with la_code column.

    Returns:
        DataFrame with region_code column added.
    """
    la_to_region = _get_la_to_region_map()

    def get_region(la_code: str) -> str:
        if not isinstance(la_code, str):
            return ""
        if la_code in la_to_region:
            return la_to_region[la_code]
        if la_code.startswith("W"):
            return "W99999999"
        if la_code.startswith("S"):
            return "S99999999"
        if la_code.startswith("N"):
            return "N99999999"
        return ""

    df["region_code"] = df["la_code"].apply(get_region)
    missing_eng = (
        (df["la_code"].fillna("").str.startswith("E")) & (df["region_code"] == "")
    ).sum()
    if missing_eng:
        logger.warning("Could not assign region_code for %s English rows", missing_eng)
    return df


def _assign_country(df: pd.DataFrame) -> pd.DataFrame:
    """Assign country name based on OA code prefix.

    Args:
        df: DataFrame with oa_code column.

    Returns:
        DataFrame with country column added.
    """

    def get_country(oa_code: str) -> str:
        if not isinstance(oa_code, str):
            return "Unknown"
        if oa_code.startswith("E"):
            return "England"
        elif oa_code.startswith("W"):
            return "Wales"
        elif oa_code.startswith("S"):
            return "Scotland"
        else:
            return "Northern Ireland"

    df["country"] = df["oa_code"].apply(get_country)
    return df


# ── Main Build Function ──────────────────────────────────


def build_oa_crosswalk(
    save: bool = True,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Build the unified UK Output Area crosswalk.

    Downloads data from ONS, NRS, and NISRA, combines into
    a single crosswalk, and optionally saves to compressed
    CSV.

    Args:
        save: Whether to save to disk.
        output_path: Override output path.

    Returns:
        DataFrame with columns: oa_code, lsoa_code,
        msoa_code, la_code, constituency_code,
        region_code, country, population
    """
    if output_path is None:
        output_path = CROSSWALK_PATH

    # ── England & Wales ──
    logger.info("=== Building E+W OA hierarchy ===")
    ew_hierarchy = _get_ew_oa_hierarchy()
    ew_population = _get_ew_population()
    ew_const = _get_ew_constituency()

    ew = ew_hierarchy.merge(ew_population, on="oa_code", how="left")
    ew["population"] = ew["population"].fillna(0).astype(int)
    ew = ew.merge(ew_const, on="oa_code", how="left")
    ew["constituency_code"] = ew["constituency_code"].fillna("")

    logger.info(f"E+W: {len(ew):,} OAs, pop {ew['population'].sum():,}")

    # ── Scotland ──
    logger.info("=== Building Scotland OA hierarchy ===")
    scot = _get_scotland_oa_hierarchy()
    scot_pop = _get_scotland_oa_population()

    scot = scot.merge(scot_pop, on="oa_code", how="left")
    scot["population"] = scot["population"].fillna(0).astype(int)

    logger.info(f"Scotland: {len(scot):,} OAs, pop {scot['population'].sum():,}")

    # ── Northern Ireland ──
    logger.info("=== Building NI hierarchy ===")
    ni = _get_ni_hierarchy()
    ni_pop = _get_ni_population()

    ni = ni.merge(ni_pop, on="oa_code", how="left")
    ni["population"] = ni["population"].fillna(0).astype(int)
    ni["constituency_code"] = ""

    logger.info(f"NI: {len(ni):,} Data Zones, pop {ni['population'].sum():,}")

    # ── Combine ──
    combined = pd.concat([ew, scot, ni], ignore_index=True)
    combined = _assign_regions(combined)
    combined = _assign_country(combined)

    combined = combined[
        [
            "oa_code",
            "lsoa_code",
            "msoa_code",
            "la_code",
            "constituency_code",
            "region_code",
            "country",
            "population",
        ]
    ]

    logger.info(
        f"=== Total: {len(combined):,} areas, pop {combined['population'].sum():,} ==="
    )

    if save:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(output_path, index=False, compression="gzip")
        logger.info(f"Saved crosswalk to {output_path}")

    return combined


def load_oa_crosswalk(
    path: Optional[Path] = None,
) -> pd.DataFrame:
    """Load the pre-built OA crosswalk from disk.

    Args:
        path: Override path (default:
            storage/oa_crosswalk.csv.gz).

    Returns:
        DataFrame with crosswalk columns.

    Raises:
        FileNotFoundError: If crosswalk file doesn't exist.
    """
    if path is None:
        path = CROSSWALK_PATH

    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run build_oa_crosswalk() "
            "or 'python -m policyengine_uk_data.calibration."
            "oa_crosswalk' to generate."
        )

    df = pd.read_csv(
        path,
        dtype={
            "oa_code": str,
            "lsoa_code": str,
            "msoa_code": str,
            "la_code": str,
            "constituency_code": str,
            "region_code": str,
            "country": str,
        },
    )
    df["population"] = pd.to_numeric(df["population"], errors="coerce").astype("Int64")
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    build_oa_crosswalk()
