"""Diagnostics for local area calibration: target sparsity and effective sample size.

Loads all target data sources (CSVs, Excel files) used during calibration and
reports coverage, missing values, and effective FRS sample pool per area.

Usage:
    python policyengine_uk_data/datasets/local_areas/calibration_diagnostics.py
"""

import json

import h5py
import pandas as pd
import numpy as np
from pathlib import Path

STORAGE = Path(__file__).parent.parent.parent / "storage"
CONST_TARGETS = Path(__file__).parent / "constituencies" / "targets"
LA_TARGETS = Path(__file__).parent / "local_authorities" / "targets"


def _country_from_code(code: str) -> str:
    return {"E": "England", "W": "Wales", "S": "Scotland", "N": "Northern Ireland"}.get(
        code[0], "Unknown"
    )


def _load_ons_la_income() -> pd.DataFrame:
    """Load ONS income estimates by local authority (no package imports needed)."""
    xlsx = pd.ExcelFile(STORAGE / "local_authority_ons_income.xlsx")

    def load_sheet(sheet_name, value_col):
        df = pd.read_excel(xlsx, sheet_name=sheet_name, header=3)
        df.columns = [
            "msoa_code", "msoa_name", "la_code", "la_name",
            "region_code", "region_name", value_col,
            "upper_ci", "lower_ci", "ci_width",
        ]
        df = df.iloc[1:].dropna(subset=["msoa_code"])
        df[value_col] = pd.to_numeric(df[value_col])
        return df[["la_code", value_col]]

    bhc = load_sheet("Net income before housing costs", "net_income_bhc")
    ahc = load_sheet("Net income after housing costs", "net_income_ahc")
    la_bhc = bhc.groupby("la_code")["net_income_bhc"].mean().reset_index()
    la_ahc = ahc.groupby("la_code")["net_income_ahc"].mean().reset_index()
    return la_bhc.merge(la_ahc, on="la_code")


def _load_uc_pc_households() -> pd.DataFrame:
    """Load UC constituency household counts from Excel (standalone)."""
    gb_path = STORAGE / "uc_pc_households.xlsx"
    df_gb = pd.read_excel(gb_path, header=None)
    rows = []
    for idx in range(8, len(df_gb)):
        name = df_gb.iloc[idx, 1]
        count = df_gb.iloc[idx, 3]
        if pd.isna(name) or pd.isna(count) or name in ("Total", "Unknown"):
            continue
        rows.append({"constituency_name": name, "household_count": int(count)})

    ni_path = STORAGE / "dfc-ni-uc-stats-supp-tables-may-2025.ods"
    if ni_path.exists():
        try:
            df_ni = pd.read_excel(ni_path, sheet_name="5b", engine="odf", header=None)
            ni_names = df_ni.iloc[2, 1:19].tolist()
            may_row = df_ni[df_ni[0] == "May 2025"].iloc[0]
            for col_idx, name in enumerate(ni_names, start=1):
                count = may_row[col_idx]
                if pd.notna(count) and count != 0:
                    rows.append({"constituency_name": name, "household_count": int(count)})
        except Exception:
            pass

    return pd.DataFrame(rows)


def _load_uc_la_households() -> pd.DataFrame:
    """Load UC local authority household counts from Excel (standalone)."""
    gb_path = STORAGE / "uc_la_households.xlsx"
    df_gb = pd.read_excel(gb_path, header=None)
    rows = []
    for idx in range(8, len(df_gb)):
        name = df_gb.iloc[idx, 2]
        count = df_gb.iloc[idx, 3]
        if pd.isna(name) or pd.isna(count) or name in ("Total", "Unknown"):
            continue
        rows.append({"la_name": name, "household_count": int(count)})

    ni_path = STORAGE / "dfc-ni-uc-stats-supp-tables-may-2025.ods"
    if ni_path.exists():
        try:
            df_ni = pd.read_excel(ni_path, sheet_name="5c", engine="odf", header=None)
            ni_names = df_ni.iloc[2, 1:12].tolist()
            may_row = df_ni[df_ni[0] == "May 2025"].iloc[0]
            for col_idx, name in enumerate(ni_names, start=1):
                if pd.notna(name) and name != "Ards and North Down":
                    count = may_row[col_idx]
                    if pd.notna(count) and count != 0:
                        rows.append({"la_name": name, "household_count": int(count)})
        except Exception:
            pass

    return pd.DataFrame(rows)


def _load_constituency_targets() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (area_info, targets) DataFrames for constituencies."""
    areas = pd.read_csv(STORAGE / "constituencies_2024.csv")
    spi = pd.read_csv(CONST_TARGETS / "spi_by_constituency.csv")
    age = pd.read_csv(CONST_TARGETS / "age.csv")

    targets = pd.DataFrame({"code": areas["code"], "name": areas["name"]})
    targets["country"] = targets["code"].apply(_country_from_code)

    for var in ["self_employment_income", "employment_income"]:
        targets[f"hmrc/{var}/amount"] = spi[f"{var}_amount"].values
        targets[f"hmrc/{var}/count"] = spi[f"{var}_count"].values

    for lower in range(0, 80, 10):
        upper = lower + 10
        cols = [str(a) for a in range(lower, upper)]
        targets[f"age/{lower}_{upper}"] = age[cols].sum(axis=1).values

    try:
        uc = _load_uc_pc_households()
        targets["uc_households"] = uc["household_count"].values
    except Exception:
        targets["uc_households"] = np.nan

    return areas, targets


def _load_la_targets() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (area_info, targets) DataFrames for local authorities."""
    areas = pd.read_csv(STORAGE / "local_authorities_2021.csv")
    spi = pd.read_csv(LA_TARGETS / "spi_by_la.csv")
    age = pd.read_csv(LA_TARGETS / "age.csv")

    targets = pd.DataFrame({"code": areas["code"], "name": areas["name"]})
    targets["country"] = targets["code"].apply(_country_from_code)

    for var in ["self_employment_income", "employment_income"]:
        targets[f"hmrc/{var}/amount"] = spi[f"{var}_amount"].values
        targets[f"hmrc/{var}/count"] = spi[f"{var}_count"].values

    for lower in range(0, 80, 10):
        upper = lower + 10
        cols = [str(a) for a in range(lower, upper)]
        targets[f"age/{lower}_{upper}"] = age[cols].sum(axis=1).values

    try:
        uc = _load_uc_la_households()
        targets["uc_households"] = uc["household_count"].values
    except Exception:
        targets["uc_households"] = np.nan

    # ONS income targets
    try:
        ons = _load_ons_la_income()
        merged = areas.merge(ons, left_on="code", right_on="la_code", how="left")
        targets["ons/net_income_bhc"] = merged["net_income_bhc"].values
        targets["ons/net_income_ahc"] = merged["net_income_ahc"].values
        targets["ons/housing_costs"] = (
            merged["net_income_bhc"].values - merged["net_income_ahc"].values
        )
    except Exception:
        for col in ["ons/net_income_bhc", "ons/net_income_ahc", "ons/housing_costs"]:
            targets[col] = np.nan

    # Household counts per LA
    try:
        hh = pd.read_excel(STORAGE / "la_count_households.xlsx", sheet_name="Dataset")
        hh.columns = ["la_code", "la_name", "households"]
        merged_hh = areas.merge(hh, left_on="code", right_on="la_code", how="left")
        targets["_households"] = merged_hh["households"].values
    except Exception:
        targets["_households"] = np.nan

    # Tenure targets
    try:
        tenure = pd.read_excel(STORAGE / "la_tenure.xlsx", sheet_name="data download")
        tenure.columns = [
            "region_code", "region_name", "la_code", "la_name",
            "owned_outright_pct", "owned_mortgage_pct",
            "private_rent_pct", "social_rent_pct",
        ]
        merged_t = areas.merge(
            tenure[["la_code", "owned_outright_pct", "owned_mortgage_pct",
                     "private_rent_pct", "social_rent_pct"]],
            left_on="code", right_on="la_code", how="left",
        )
        for tenure_type in ["owned_outright", "owned_mortgage", "private_rent", "social_rent"]:
            pct = merged_t[f"{tenure_type}_pct"].values
            hh_vals = targets["_households"].values
            targets[f"tenure/{tenure_type}"] = np.where(
                pd.notna(pct) & pd.notna(hh_vals),
                pct / 100 * hh_vals,
                np.nan,
            )
    except Exception:
        for t in ["owned_outright", "owned_mortgage", "private_rent", "social_rent"]:
            targets[f"tenure/{t}"] = np.nan

    # Private rent amounts
    try:
        rent = pd.read_excel(
            STORAGE / "la_private_rents_median.xlsx",
            sheet_name="Figure 3", header=5,
        )
        rent.columns = [
            "col0", "la_code_old", "area_code", "area_name", "room",
            "studio", "one_bed", "two_bed", "three_bed", "four_plus",
            "median_monthly_rent",
        ]
        rent = rent[rent["area_code"].astype(str).str.match(r"^E0[6789]")]
        rent["median_monthly_rent"] = pd.to_numeric(
            rent["median_monthly_rent"], errors="coerce"
        )
        rent["median_annual_rent"] = rent["median_monthly_rent"] * 12
        merged_r = areas.merge(
            rent[["area_code", "median_annual_rent"]],
            left_on="code", right_on="area_code", how="left",
        )
        prp = targets.get("tenure/private_rent")
        if prp is not None:
            targets["rent/private_rent"] = np.where(
                merged_r["median_annual_rent"].notna() & prp.notna(),
                merged_r["median_annual_rent"].values * prp / targets["_households"].values,
                np.nan,
            )
        else:
            targets["rent/private_rent"] = np.nan
    except Exception:
        targets["rent/private_rent"] = np.nan

    return areas, targets


def _target_columns(targets: pd.DataFrame) -> list[str]:
    """Return columns that are calibration target variables (not metadata)."""
    return [c for c in targets.columns if c not in ("code", "name", "country", "_households")]


def sparsity_by_variable(targets: pd.DataFrame, level: str) -> pd.DataFrame:
    """For each variable, count areas with real vs missing/zero targets."""
    cols = _target_columns(targets)
    rows = []
    for col in cols:
        vals = targets[col]
        n_total = len(vals)
        n_missing = int(vals.isna().sum())
        n_zero = int((vals.notna() & (vals == 0)).sum())
        n_nonzero = n_total - n_missing - n_zero
        rows.append({
            "level": level,
            "variable": col,
            "n_areas": n_total,
            "n_with_target": n_nonzero,
            "n_missing": n_missing,
            "n_zero": n_zero,
            "pct_coverage": round(100 * n_nonzero / n_total, 1),
        })
    return pd.DataFrame(rows)


def sparsity_by_area(targets: pd.DataFrame, level: str) -> pd.DataFrame:
    """For each area, count variables with real vs missing/zero targets."""
    cols = _target_columns(targets)
    rows = []
    for _, row in targets.iterrows():
        vals = row[cols]
        n_total = len(cols)
        n_present = int(vals.notna().sum())
        n_zero = int((vals.notna() & (vals == 0)).sum())
        n_missing = n_total - n_present
        n_nonzero = n_present - n_zero
        rows.append({
            "level": level,
            "code": row["code"],
            "name": row["name"],
            "country": row["country"],
            "n_variables": n_total,
            "n_with_target": n_nonzero,
            "n_missing": n_missing,
            "n_zero": n_zero,
            "pct_coverage": round(100 * n_nonzero / n_total, 1),
        })
    return pd.DataFrame(rows)


def sample_size_by_country(targets: pd.DataFrame, level: str) -> pd.DataFrame:
    """Count areas per country (determines effective FRS sample pool)."""
    return (
        targets.groupby("country")
        .agg(n_areas=("code", "count"))
        .reset_index()
        .assign(level=level)
    )


def _load_household_variable_data() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Compute per-household boolean masks and numeric values for calibration variables.

    Returns (masks, values) where:
    - masks[var]: boolean, True if household contributes to that variable
    - values[var]: float, the household-level calibration column value
    """
    dataset_path = STORAGE / "enhanced_frs_2023_24.h5"
    if not dataset_path.exists():
        return {}, {}

    with h5py.File(dataset_path, "r") as f:
        persons = f["person/table"][:]
        households = f["household/table"][:]

    n_hh = len(households)
    hh_ids = households["household_id"]
    hh_id_to_idx = {int(hid): i for i, hid in enumerate(hh_ids)}
    person_hh_idx = np.array([hh_id_to_idx[int(pid)] for pid in persons["person_household_id"]])

    masks = {}
    values = {}
    n_total_households = n_hh

    # Income variables: sum person-level income to household
    for var in ["employment_income", "self_employment_income"]:
        person_vals = persons[var].astype(float)
        hh_sum = np.zeros(n_hh, dtype=float)
        np.add.at(hh_sum, person_hh_idx, person_vals)
        hh_has = hh_sum > 0

        person_count = (person_vals > 0).astype(float)
        hh_count = np.zeros(n_hh, dtype=float)
        np.add.at(hh_count, person_hh_idx, person_count)

        masks[f"hmrc/{var}/amount"] = hh_has
        values[f"hmrc/{var}/amount"] = hh_sum
        masks[f"hmrc/{var}/count"] = hh_has
        values[f"hmrc/{var}/count"] = hh_count

    # Age bands: count persons in band per household
    ages = persons["age"].astype(float)
    for lower in range(0, 80, 10):
        upper = lower + 10
        in_band = ((ages >= lower) & (ages < upper)).astype(float)
        hh_count = np.zeros(n_hh, dtype=float)
        np.add.at(hh_count, person_hh_idx, in_band)
        masks[f"age/{lower}_{upper}"] = hh_count > 0
        values[f"age/{lower}_{upper}"] = hh_count

    # UC: household has person reporting UC
    uc_reported = (persons["universal_credit_reported"] > 0).astype(float)
    hh_uc = np.zeros(n_hh, dtype=float)
    np.maximum.at(hh_uc, person_hh_idx, uc_reported)
    masks["uc_households"] = hh_uc > 0
    values["uc_households"] = hh_uc

    # ONS income / housing costs (household-level proxies)
    # Sum person incomes to household as gross income proxy
    income_cols = [
        "employment_income", "self_employment_income",
        "private_pension_income", "savings_interest_income",
        "dividend_income", "property_income",
        "maintenance_income", "miscellaneous_income",
    ]
    total_person_income = np.zeros(len(persons), dtype=float)
    for col in income_cols:
        if col in persons.dtype.names:
            total_person_income += persons[col].astype(float)
    hh_income = np.zeros(n_hh, dtype=float)
    np.add.at(hh_income, person_hh_idx, total_person_income)

    housing_costs = (
        households["rent"].astype(float)
        + households["mortgage_interest_repayment"].astype(float)
    )

    masks["ons/net_income_bhc"] = hh_income > 0
    values["ons/net_income_bhc"] = hh_income
    masks["ons/net_income_ahc"] = hh_income > 0
    values["ons/net_income_ahc"] = np.maximum(hh_income - housing_costs, 0)
    masks["ons/housing_costs"] = housing_costs > 0
    values["ons/housing_costs"] = housing_costs

    # Tenure types (household-level, 0/1)
    tenure = np.array([t.decode() for t in households["tenure_type"]])
    for name, match in [
        ("tenure/owned_outright", tenure == "OWNED_OUTRIGHT"),
        ("tenure/owned_mortgage", tenure == "OWNED_WITH_MORTGAGE"),
        ("tenure/private_rent", tenure == "RENT_PRIVATELY"),
        ("tenure/social_rent", (tenure == "RENT_FROM_COUNCIL") | (tenure == "RENT_FROM_HA")),
    ]:
        masks[name] = match
        values[name] = match.astype(float)

    # Private rent amount: rent for private renters only
    rent = households["rent"].astype(float)
    is_private = (tenure == "RENT_PRIVATELY")
    masks["rent/private_rent"] = is_private & (rent > 0)
    values["rent/private_rent"] = rent * is_private

    return masks, values, n_total_households


def compute_sample_sizes_and_errors(
    targets_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute per-area per-variable sample sizes and calibration errors.

    Returns (sample_sizes_df, errors_df).
    """
    masks, values, n_total_households = _load_household_variable_data()
    if not masks:
        return pd.DataFrame(), pd.DataFrame()

    weight_configs = [
        ("constituency", "parliamentary_constituency_weights.h5", "constituencies_2024.csv"),
        ("local_authority", "local_authority_weights.h5", "local_authorities_2021.csv"),
        ("country", "local_authority_weights.h5", "local_authorities_2021.csv"),
    ]

    ss_rows = []
    err_rows = []

    for level, weight_file, area_file in weight_configs:
        weight_path = STORAGE / weight_file
        area_path = STORAGE / area_file
        if not weight_path.exists() or not area_path.exists():
            continue

        with h5py.File(weight_path, "r") as f:
            weights = f["2025"][:]  # (n_areas, n_households)

        areas = pd.read_csv(area_path)

        # For country level, aggregate LA weights by country
        if level == "country":
            areas["country"] = areas["code"].apply(_country_from_code)
            countries = sorted(areas["country"].unique())
            country_weights = []
            country_areas = []
            for c in countries:
                mask = areas["country"] == c
                cw = weights[mask.values].sum(axis=0)
                country_weights.append(cw)
                country_areas.append({
                    "code": c[0],
                    "name": c,
                })
            weights = np.array(country_weights)
            areas = pd.DataFrame(country_areas)

        has_weight = weights > 0
        n_with_weight = has_weight.sum(axis=1)  # per area

        # Get target values for this level
        if level == "country" and "level" in targets_df.columns:
            la_targets = targets_df[targets_df["level"] == "local_authority"].copy()
            if not la_targets.empty:
                la_targets["_country"] = la_targets["code"].apply(_country_from_code)
                numeric_cols = [c for c in la_targets.columns if c not in ("code", "name", "country", "level", "_country", "_households")]
                country_agg = la_targets.groupby("_country")[numeric_cols].sum().reset_index()
                country_agg["code"] = country_agg["_country"].str[0]
                country_agg["name"] = country_agg["_country"]
                level_targets = country_agg
            else:
                level_targets = pd.DataFrame()
        else:
            level_targets = targets_df[targets_df["level"] == level] if "level" in targets_df.columns else pd.DataFrame()

        for var_name, var_mask in masks.items():
            # Skip variables not relevant to this level
            if level == "constituency" and var_name.startswith(("ons/", "tenure/", "rent/")):
                continue

            # Sample sizes
            relevant = has_weight & var_mask[np.newaxis, :]
            n_relevant = relevant.sum(axis=1)

            w_relevant = weights * var_mask[np.newaxis, :].astype(float)
            w_sum = w_relevant.sum(axis=1)
            w_sq_sum = (w_relevant ** 2).sum(axis=1)
            ess = np.where(w_sq_sum > 0, w_sum ** 2 / w_sq_sum, 0)

            # Calibration estimates: weights @ values
            var_vals = values[var_name]
            estimates = weights @ var_vals  # (n_areas,)

            for i in range(len(areas)):
                code = areas.iloc[i]["code"]
                name = areas.iloc[i]["name"]

                ss_rows.append({
                    "level": level,
                    "area_code": code,
                    "area_name": name,
                    "variable": var_name,
                    "n_total": n_total_households,
                    "n_in_area": int(n_with_weight[i]),
                    "n_with_value": int(var_mask.sum()),
                    "n_nonzero_weight": int(n_relevant[i]),
                    "weighted_total": round(float(weights[i].sum()), 1),
                    "weighted_with_value": round(float((weights[i] * var_mask).sum()), 1),
                    "ess": round(float(ess[i]), 1),
                })

                # Look up the target value
                target_val = None
                if not level_targets.empty and var_name in level_targets.columns:
                    if level == "country":
                        match = level_targets[level_targets["name"] == name]
                    else:
                        match = level_targets[level_targets["code"] == code]
                    if not match.empty:
                        tv = match.iloc[0][var_name]
                        if pd.notna(tv):
                            target_val = float(tv)

                estimate = float(estimates[i])
                if target_val is not None and target_val != 0:
                    err_rows.append({
                        "level": level,
                        "area_code": code,
                        "area_name": name,
                        "variable": var_name,
                        "target": round(target_val, 1),
                        "estimate": round(estimate, 1),
                        "error": round(estimate - target_val, 1),
                        "pct_error": round(100 * (estimate - target_val) / target_val, 2),
                    })

    return pd.DataFrame(ss_rows), pd.DataFrame(err_rows)


def generate_diagnostics() -> dict[str, pd.DataFrame]:
    """Generate all diagnostic tables and return as a dict of DataFrames."""
    _, const_targets = _load_constituency_targets()
    _, la_targets = _load_la_targets()

    results = {}

    # Sparsity by variable
    results["sparsity_by_variable"] = pd.concat([
        sparsity_by_variable(const_targets, "constituency"),
        sparsity_by_variable(la_targets, "local_authority"),
    ], ignore_index=True)

    # Sparsity by area
    results["sparsity_by_area"] = pd.concat([
        sparsity_by_area(const_targets, "constituency"),
        sparsity_by_area(la_targets, "local_authority"),
    ], ignore_index=True)

    # Areas per country
    results["areas_per_country"] = pd.concat([
        sample_size_by_country(const_targets, "constituency"),
        sample_size_by_country(la_targets, "local_authority"),
    ], ignore_index=True)

    # FRS approximate sample sizes (from the 2023-24 survey)
    # ~20,000 households, distributed roughly:
    #   England ~83%, Wales ~5%, Scotland ~9%, NI ~3%
    frs_approx = pd.DataFrame({
        "country": ["England", "Wales", "Scotland", "Northern Ireland"],
        "approx_frs_households": [16_600, 1_000, 1_800, 600],
    })
    areas_by_country = results["areas_per_country"].pivot(
        index="country", columns="level", values="n_areas"
    ).reset_index()
    sample_context = frs_approx.merge(areas_by_country, on="country", how="left")
    if "constituency" in sample_context.columns:
        sample_context["frs_per_constituency"] = (
            sample_context["approx_frs_households"] / sample_context["constituency"]
        ).round(1)
    if "local_authority" in sample_context.columns:
        sample_context["frs_per_la"] = (
            sample_context["approx_frs_households"] / sample_context["local_authority"]
        ).round(1)
    results["sample_size_context"] = sample_context

    # Raw target values per area (for the dashboard detail views)
    const_export = const_targets.copy()
    const_export["level"] = "constituency"
    la_export = la_targets.drop(columns=["_households"], errors="ignore").copy()
    la_export["level"] = "local_authority"
    results["targets"] = pd.concat([const_export, la_export], ignore_index=True)

    # Per-area, per-variable sample sizes and calibration errors
    sample_sizes, errors = compute_sample_sizes_and_errors(results["targets"])
    if not sample_sizes.empty:
        results["sample_sizes"] = sample_sizes
    if not errors.empty:
        results["errors"] = errors

    return results


def print_diagnostics(results: dict[str, pd.DataFrame] = None):
    """Print a summary report to stdout."""
    if results is None:
        results = generate_diagnostics()

    print("=" * 80)
    print("LOCAL AREA CALIBRATION DIAGNOSTICS")
    print("=" * 80)

    print("\n1. TARGET SPARSITY BY VARIABLE")
    print("-" * 80)
    df = results["sparsity_by_variable"]
    for level in df["level"].unique():
        subset = df[df["level"] == level]
        print(f"\n  {level.upper()} ({subset['n_areas'].iloc[0]} areas)")
        print(f"  {'Variable':<35} {'With target':>12} {'Missing':>8} {'Zero':>6} {'Coverage':>9}")
        for _, row in subset.iterrows():
            print(
                f"  {row['variable']:<35} {row['n_with_target']:>12} "
                f"{row['n_missing']:>8} {row['n_zero']:>6} {row['pct_coverage']:>8.1f}%"
            )

    print("\n\n2. TARGET SPARSITY BY AREA (summary)")
    print("-" * 80)
    df = results["sparsity_by_area"]
    for level in df["level"].unique():
        subset = df[df["level"] == level]
        print(f"\n  {level.upper()}")
        print(f"    Total areas: {len(subset)}")
        print(f"    Variables per area: {subset['n_variables'].iloc[0]}")
        print(f"    Mean coverage: {subset['pct_coverage'].mean():.1f}%")
        print(f"    Min coverage: {subset['pct_coverage'].min():.1f}% "
              f"({subset.loc[subset['pct_coverage'].idxmin(), 'name']})")
        print(f"    Max coverage: {subset['pct_coverage'].max():.1f}%")

        worst = subset.nsmallest(5, "pct_coverage")
        if worst["pct_coverage"].iloc[0] < 100:
            print(f"\n    Worst-covered areas:")
            for _, row in worst.iterrows():
                print(
                    f"      {row['name']:<35} {row['country']:<20} "
                    f"{row['n_with_target']}/{row['n_variables']} "
                    f"({row['pct_coverage']:.1f}%)"
                )

    print("\n\n3. EFFECTIVE FRS SAMPLE SIZE PER AREA")
    print("-" * 80)
    df = results["sample_size_context"]
    print(
        f"\n  {'Country':<20} {'FRS HHs':>10} {'Constituencies':>15} "
        f"{'FRS/const':>10} {'LAs':>6} {'FRS/LA':>8}"
    )
    for _, row in df.iterrows():
        const = row.get("constituency", 0)
        la = row.get("local_authority", 0)
        fpc = row.get("frs_per_constituency", 0)
        fla = row.get("frs_per_la", 0)
        print(
            f"  {row['country']:<20} {row['approx_frs_households']:>10,} "
            f"{int(const) if pd.notna(const) else 0:>15} {fpc:>10.1f} "
            f"{int(la) if pd.notna(la) else 0:>6} {fla:>8.1f}"
        )

    print("\n  Note: The optimiser can assign non-zero weight to any same-country")
    print("  household for any area, so these ratios indicate the pool size, not")
    print("  a hard constraint. Weight dropout (5%) regularises against overfitting.")


def export_json(results: dict[str, pd.DataFrame] = None, output_path: str = None) -> Path:
    """Export all diagnostic DataFrames as a single JSON file for the dashboard."""
    if output_path is None:
        output_path = Path(__file__).parent / "calibration_diagnostics.json"
    else:
        output_path = Path(output_path)

    if results is None:
        results = generate_diagnostics()
    data = {
        name: json.loads(df.to_json(orient="records"))
        for name, df in results.items()
    }
    output_path.write_text(json.dumps(data, indent=2))
    return output_path


def save_diagnostics(output_dir: str = None):
    """Save all diagnostic tables as CSV files."""
    if output_dir is None:
        output_dir = Path(__file__).parent
    else:
        output_dir = Path(output_dir)

    results = generate_diagnostics()
    for name, df in results.items():
        path = output_dir / f"{name}.csv"
        df.to_csv(path, index=False)
        print(f"Saved {path}")


if __name__ == "__main__":
    results = generate_diagnostics()
    print_diagnostics(results)

    json_path = export_json(results)
    print(f"\nExported JSON: {json_path}")

    dashboard_path = Path(__file__).parent / "calibration_dashboard.html"
    print(f"Open dashboard:  file://{dashboard_path.resolve()}")
