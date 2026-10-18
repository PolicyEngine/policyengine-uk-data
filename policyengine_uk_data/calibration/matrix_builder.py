"""Sparse calibration matrix builder for cloned FRS datasets.

Bridges Phase 2 (clone-and-assign) and Phase 3 (L0 calibration) by
building target matrices that exploit the OA geography columns on
each cloned household.

Provides two interfaces:
- ``create_cloned_target_matrix``: backward-compatible (metrics, targets,
  country_mask) triple for use as ``matrix_fn`` in both the dense Adam
  optimizer and the L0 calibrator.
- ``build_sparse_calibration_matrix``: direct sparse path that skips the
  dense country_mask, producing the (M_csr, y, group_ids) triple that
  ``calibrate_l0`` consumes internally.

Phase 4 of the OA calibration pipeline.
US reference: policyengine-us-data PRs #456, #489.
"""

import logging

import numpy as np
import pandas as pd
from scipy import sparse as sp

from policyengine_uk_data.storage import STORAGE_FOLDER

logger = logging.getLogger(__name__)

AREA_TYPES = ("constituency", "la")


def _load_area_codes(area_type: str) -> pd.DataFrame:
    if area_type == "constituency":
        return pd.read_csv(STORAGE_FOLDER / "constituencies_2024.csv")
    elif area_type == "la":
        return pd.read_csv(STORAGE_FOLDER / "local_authorities_2021.csv")
    raise ValueError(f"Unknown area_type: {area_type}")


def _geo_column(area_type: str) -> str:
    return "constituency_code_oa" if area_type == "constituency" else "la_code_oa"


def build_assignment_matrix(
    dataset,
    area_type: str,
    area_codes: pd.Series,
) -> sp.csr_matrix:
    """Build sparse (n_areas, n_households) binary assignment matrix.

    Each household is assigned to exactly one area via its OA geography
    column. Households with empty/unmatched codes get no assignment
    (they contribute to national targets only).

    Args:
        dataset: Cloned dataset with la_code_oa/constituency_code_oa.
        area_type: "constituency" or "la".
        area_codes: Series of canonical area codes (defines row order).

    Returns:
        Sparse CSR matrix of shape (n_areas, n_households).
    """
    geo_col = _geo_column(area_type)
    hh_codes = dataset.household[geo_col].values

    code_to_idx = {code: i for i, code in enumerate(area_codes)}
    n_areas = len(area_codes)
    n_households = len(hh_codes)

    rows = []
    cols = []
    for j, code in enumerate(hh_codes):
        code_str = str(code)
        if code_str in code_to_idx:
            rows.append(code_to_idx[code_str])
            cols.append(j)

    n_assigned = len(rows)
    logger.info(
        "Assignment matrix: %d/%d households assigned to %d %s areas",
        n_assigned,
        n_households,
        n_areas,
        area_type,
    )

    return sp.csr_matrix(
        (np.ones(n_assigned, dtype=np.float64), (rows, cols)),
        shape=(n_areas, n_households),
    )


def _compute_household_metrics(sim, area_type: str) -> pd.DataFrame:
    """Compute household-level metric columns from a Microsimulation.

    Consolidates the metric computation currently duplicated between
    constituencies/loss.py and local_authorities/loss.py.
    """
    from policyengine_uk_data.targets.sources.local_income import INCOME_VARIABLES

    matrix = pd.DataFrame()

    # Income metrics
    for income_variable in INCOME_VARIABLES:
        income_values = sim.calculate(income_variable).values
        in_spi_frame = sim.calculate("income_tax").values > 0
        matrix[f"hmrc/{income_variable}/amount"] = sim.map_result(
            income_values * in_spi_frame, "person", "household"
        )
        matrix[f"hmrc/{income_variable}/count"] = sim.map_result(
            (income_values != 0) * in_spi_frame, "person", "household"
        )

    # Age metrics
    age = sim.calculate("age").values
    for lower in range(0, 80, 10):
        upper = lower + 10
        in_band = (age >= lower) & (age < upper)
        matrix[f"age/{lower}_{upper}"] = sim.map_result(in_band, "person", "household")

    # UC metrics
    on_uc = (sim.calculate("universal_credit").values > 0).astype(int)
    matrix["uc_households"] = sim.map_result(on_uc, "benunit", "household")

    if area_type == "constituency":
        # UC by children breakdown (constituency only)
        is_child = sim.calculate("is_child").values
        children_per_hh = sim.map_result(is_child, "person", "household")
        on_uc_hh = sim.map_result(on_uc, "benunit", "household") > 0
        matrix["uc_hh_0_children"] = (on_uc_hh & (children_per_hh == 0)).astype(float)
        matrix["uc_hh_1_child"] = (on_uc_hh & (children_per_hh == 1)).astype(float)
        matrix["uc_hh_2_children"] = (on_uc_hh & (children_per_hh == 2)).astype(float)
        matrix["uc_hh_3plus_children"] = (on_uc_hh & (children_per_hh >= 3)).astype(
            float
        )

    if area_type == "la":
        # LA-only metrics: ONS income, tenure, rent
        hbai_net_income = sim.calculate("equiv_hbai_household_net_income").values
        hbai_net_income_ahc = sim.calculate(
            "equiv_hbai_household_net_income_ahc"
        ).values
        housing_costs = hbai_net_income - hbai_net_income_ahc

        matrix["ons/equiv_net_income_bhc"] = hbai_net_income
        matrix["ons/equiv_net_income_ahc"] = hbai_net_income_ahc
        matrix["ons/equiv_housing_costs"] = housing_costs

        tenure_type = sim.calculate("tenure_type").values
        matrix["tenure/owned_outright"] = (tenure_type == "OWNED_OUTRIGHT").astype(
            float
        )
        matrix["tenure/owned_mortgage"] = (tenure_type == "OWNED_WITH_MORTGAGE").astype(
            float
        )
        matrix["tenure/private_rent"] = (tenure_type == "RENT_PRIVATELY").astype(float)
        matrix["tenure/social_rent"] = (
            (tenure_type == "RENT_FROM_COUNCIL") | (tenure_type == "RENT_FROM_HA")
        ).astype(float)

        is_private_renter = (tenure_type == "RENT_PRIVATELY").astype(float)
        benunit_rent = sim.calculate("benunit_rent").values
        household_rent = sim.map_result(benunit_rent, "benunit", "household")
        matrix["rent/private_rent"] = household_rent * is_private_renter

    return matrix


def _load_area_targets(
    area_type: str,
    area_codes: pd.DataFrame,
    dataset,
    sim,
) -> pd.DataFrame:
    """Load and align target values for an area type.

    Returns a DataFrame with shape (n_areas, n_metrics) aligned to
    the same column order as ``_compute_household_metrics``.
    """
    from policyengine_uk_data.targets.sources.local_income import (
        INCOME_VARIABLES,
        get_constituency_income_targets,
        get_la_income_targets,
        get_national_income_projections,
    )
    from policyengine_uk_data.targets.sources.local_age import (
        get_constituency_age_targets,
        get_la_age_targets,
        get_uk_total_population,
    )

    y = pd.DataFrame()
    time_period = dataset.time_period

    # ── Income targets ──────────────────────────────────────────────
    if area_type == "constituency":
        incomes = get_constituency_income_targets()
    else:
        incomes = get_la_income_targets()

    national_incomes = get_national_income_projections(int(time_period))

    for income_variable in INCOME_VARIABLES:
        local_targets = incomes[f"{income_variable}_amount"].values
        local_target_sum = local_targets.sum()
        national_target = national_incomes[
            (national_incomes.total_income_lower_bound == 12_570)
            & (national_incomes.total_income_upper_bound == np.inf)
        ][income_variable + "_amount"].iloc[0]
        adjustment = national_target / local_target_sum

        y[f"hmrc/{income_variable}/amount"] = local_targets * adjustment
        y[f"hmrc/{income_variable}/count"] = (
            incomes[f"{income_variable}_count"].values * adjustment
        )

    # ── Age targets ─────────────────────────────────────────────────
    if area_type == "constituency":
        age_targets = get_constituency_age_targets()
    else:
        age_targets = get_la_age_targets()

    uk_total_population = get_uk_total_population(int(time_period))
    age_cols = [c for c in age_targets.columns if c.startswith("age/")]

    targets_total_pop = 0
    for col in age_cols:
        y[col] = age_targets[col].values
        targets_total_pop += age_targets[col].values.sum()

    for col in age_cols:
        y[col] *= uk_total_population / targets_total_pop * 0.9

    # ── UC targets ──────────────────────────────────────────────────
    if area_type == "constituency":
        from policyengine_uk_data.targets.sources.local_uc import (
            get_constituency_uc_targets,
            get_constituency_uc_by_children_targets,
        )

        y["uc_households"] = get_constituency_uc_targets().values
        uc_by_children = get_constituency_uc_by_children_targets()
        for col in uc_by_children.columns:
            y[col] = uc_by_children[col].values
    else:
        from policyengine_uk_data.targets.sources.local_uc import get_la_uc_targets

        y["uc_households"] = get_la_uc_targets().values

    # ── Boundary mapping (constituency 2010 → 2024 only) ───────────
    if area_type == "constituency":
        from policyengine_uk_data.datasets.local_areas.constituencies.boundary_changes.mapping_matrix import (
            mapping_matrix,
        )

        y_columns = list(y.columns)
        y_values = mapping_matrix @ y.values
        y = pd.DataFrame(y_values, columns=y_columns)

    # ── LA extras: ONS income, tenure, rent ─────────────────────────
    if area_type == "la":
        from policyengine_uk_data.targets.sources.local_la_extras import (
            load_ons_la_income,
            load_household_counts,
            load_tenure_data,
            load_private_rents,
            UPRATING_NET_INCOME_BHC_2020_TO_2025,
            UPRATING_HOUSING_COSTS_2020_TO_2025,
        )

        original_weights = sim.calculate("household_weight", 2025).values

        ons_income = load_ons_la_income()
        households_by_la = load_household_counts()

        ons_merged = area_codes.merge(
            ons_income, left_on="code", right_on="la_code", how="left"
        ).merge(
            households_by_la,
            left_on="code",
            right_on="la_code",
            how="left",
            suffixes=("", "_hh"),
        )

        hbai_net_income = sim.calculate("equiv_hbai_household_net_income").values
        hbai_net_income_ahc = sim.calculate(
            "equiv_hbai_household_net_income_ahc"
        ).values
        housing_costs = hbai_net_income - hbai_net_income_ahc

        ons_merged["equiv_net_income_bhc_target"] = (
            ons_merged["net_income_bhc"]
            * ons_merged["households"]
            * UPRATING_NET_INCOME_BHC_2020_TO_2025
        )
        ons_merged["equiv_housing_costs_target"] = (
            ons_merged["net_income_bhc"] * ons_merged["households"]
            - ons_merged["net_income_ahc"] * ons_merged["households"]
        ) * UPRATING_HOUSING_COSTS_2020_TO_2025
        ons_merged["equiv_net_income_ahc_target"] = (
            ons_merged["equiv_net_income_bhc_target"]
            - ons_merged["equiv_housing_costs_target"]
        )

        has_ons_data = (
            ons_merged["net_income_bhc"].notna() & ons_merged["households"].notna()
        ).values
        total_households = ons_merged["households"].sum()
        la_household_share = np.where(
            ons_merged["households"].notna(),
            ons_merged["households"].values / total_households,
            1 / len(area_codes),
        )

        national_bhc = (original_weights * hbai_net_income).sum()
        national_ahc = (original_weights * hbai_net_income_ahc).sum()
        national_hc = (original_weights * housing_costs).sum()

        y["ons/equiv_net_income_bhc"] = np.where(
            has_ons_data,
            ons_merged["equiv_net_income_bhc_target"].values,
            national_bhc * la_household_share,
        )
        y["ons/equiv_net_income_ahc"] = np.where(
            has_ons_data,
            ons_merged["equiv_net_income_ahc_target"].values,
            national_ahc * la_household_share,
        )
        y["ons/equiv_housing_costs"] = np.where(
            has_ons_data,
            ons_merged["equiv_housing_costs_target"].values,
            national_hc * la_household_share,
        )

        # Tenure targets
        tenure_data = load_tenure_data()
        tenure_merged = area_codes.merge(
            tenure_data, left_on="code", right_on="la_code", how="left"
        ).merge(
            households_by_la,
            left_on="code",
            right_on="la_code",
            how="left",
            suffixes=("", "_hh"),
        )

        has_tenure = (
            tenure_merged["owned_outright_pct"].notna()
            & tenure_merged["households"].notna()
        ).values

        metrics = _compute_household_metrics(sim, area_type)
        for tenure_key, pct_col in [
            ("owned_outright", "owned_outright_pct"),
            ("owned_mortgage", "owned_mortgage_pct"),
            ("private_rent", "private_rent_pct"),
            ("social_rent", "social_rent_pct"),
        ]:
            targets = tenure_merged[pct_col] / 100 * tenure_merged["households"]
            national = (original_weights * metrics[f"tenure/{tenure_key}"].values).sum()
            y[f"tenure/{tenure_key}"] = np.where(
                has_tenure, targets.values, national * la_household_share
            )

        # Private rent amounts
        tenure_merged = tenure_merged.merge(
            load_private_rents(), left_on="code", right_on="area_code", how="left"
        )
        tenure_merged["private_rent_target"] = (
            tenure_merged["median_annual_rent"]
            * tenure_merged["private_rent_pct"]
            / 100
            * tenure_merged["households"]
        )
        has_rent = (
            tenure_merged["median_annual_rent"].notna()
            & tenure_merged["private_rent_pct"].notna()
            & tenure_merged["households"].notna()
        ).values
        national_rent = (original_weights * metrics["rent/private_rent"].values).sum()
        y["rent/private_rent"] = np.where(
            has_rent,
            tenure_merged["private_rent_target"].values,
            national_rent * la_household_share,
        )

    return y


def create_cloned_target_matrix(
    dataset,
    area_type: str = "constituency",
    time_period=None,
    reform=None,
):
    """Build (metrics, targets, country_mask) for a cloned dataset.

    Uses the OA geography columns from clone-and-assign to build a
    sparse assignment matrix, then densifies country_mask for backward
    compatibility with the existing calibration interface.

    Args:
        dataset: Cloned UKSingleYearDataset with OA geography columns.
        area_type: "constituency" or "la".
        time_period: Override calculation period.
        reform: PolicyEngine reform to apply.

    Returns:
        (metrics, targets, country_mask) triple matching the interface
        of create_constituency_target_matrix / create_local_authority_target_matrix.
    """
    from policyengine_uk import Microsimulation

    if time_period is None:
        time_period = dataset.time_period

    area_codes_df = _load_area_codes(area_type)
    area_codes = area_codes_df["code"]

    sim = Microsimulation(dataset=dataset, reform=reform)
    sim.default_calculation_period = time_period

    metrics = _compute_household_metrics(sim, area_type)
    targets = _load_area_targets(area_type, area_codes_df, dataset, sim)

    # Build sparse assignment and densify for backward compat
    assignment = build_assignment_matrix(dataset, area_type, area_codes)
    country_mask = assignment.toarray()

    return metrics, targets, country_mask


def build_sparse_calibration_matrix(
    dataset,
    area_type: str,
    national_matrix_fn,
    time_period=None,
    reform=None,
):
    """Build sparse (M, y, group_ids) directly from cloned dataset.

    Memory-efficient path that skips the dense country_mask entirely.
    Each household is in exactly one area, so we iterate over metrics
    and use the assignment column to place entries directly.

    Args:
        dataset: Cloned UKSingleYearDataset with OA geography columns.
        area_type: "constituency" or "la".
        national_matrix_fn: Function returning (metrics, targets) for
            national-level calibration.
        time_period: Override calculation period.
        reform: PolicyEngine reform to apply.

    Returns:
        (M, y, group_ids): sparse CSR matrix, target vector, group IDs.
    """
    from policyengine_uk import Microsimulation

    if time_period is None:
        time_period = dataset.time_period

    area_codes_df = _load_area_codes(area_type)
    area_codes = area_codes_df["code"]

    sim = Microsimulation(dataset=dataset, reform=reform)
    sim.default_calculation_period = time_period

    metrics = _compute_household_metrics(sim, area_type)
    targets = _load_area_targets(area_type, area_codes_df, dataset, sim)

    # Build household → area index mapping (no dense matrix needed)
    geo_col = _geo_column(area_type)
    hh_codes = dataset.household[geo_col].values
    code_to_idx = {code: i for i, code in enumerate(area_codes)}

    hh_area_idx = np.full(len(hh_codes), -1, dtype=np.int32)
    for j, code in enumerate(hh_codes):
        idx = code_to_idx.get(str(code), -1)
        if idx >= 0:
            hh_area_idx[j] = idx

    assigned_mask = hh_area_idx >= 0

    metric_values = metrics.values
    target_values = targets.values
    n_areas = len(area_codes)
    n_metrics = metric_values.shape[1]
    n_records = metric_values.shape[0]

    rows = []
    cols = []
    data = []
    y_list = []
    group_ids = []

    for j in range(n_metrics):
        metric_col = metric_values[:, j]
        for i in range(n_areas):
            target_val = target_values[i, j]
            if np.isnan(target_val) or target_val == 0:
                continue

            # Households assigned to this area with non-zero metric
            in_area = (hh_area_idx == i) & assigned_mask
            candidate_indices = np.where(in_area)[0]
            values = metric_col[candidate_indices]

            nonzero = values != 0
            if not nonzero.any():
                continue

            row_idx = len(y_list)
            record_indices = candidate_indices[nonzero]
            values = values[nonzero]

            rows.extend([row_idx] * len(record_indices))
            cols.extend(record_indices.tolist())
            data.extend(values.tolist())
            y_list.append(target_val)
            group_ids.append(j)

    # Add national targets
    national_metrics, national_targets = national_matrix_fn(dataset)
    national_metric_values = (
        national_metrics.values
        if hasattr(national_metrics, "values")
        else np.array(national_metrics)
    )
    national_target_values = (
        national_targets.values
        if hasattr(national_targets, "values")
        else np.array(national_targets)
    )

    national_group_start = n_metrics
    for j in range(len(national_target_values)):
        target_val = national_target_values[j]
        if np.isnan(target_val) or target_val == 0:
            continue

        metric_col = national_metric_values[:, j]
        nonzero = metric_col != 0
        if not nonzero.any():
            continue

        row_idx = len(y_list)
        record_indices = np.where(nonzero)[0]
        values = metric_col[record_indices]

        rows.extend([row_idx] * len(record_indices))
        cols.extend(record_indices.tolist())
        data.extend(values.tolist())
        y_list.append(target_val)
        group_ids.append(national_group_start + j)

    n_total_targets = len(y_list)
    M = sp.csr_matrix(
        (data, (rows, cols)),
        shape=(n_total_targets, n_records),
    )
    y = np.array(y_list, dtype=np.float64)
    group_ids = np.array(group_ids, dtype=np.int64)

    logger.info(
        "Sparse matrix (direct): %d targets x %d records, %.2f%% non-zero",
        n_total_targets,
        n_records,
        100 * M.nnz / (n_total_targets * n_records) if n_total_targets > 0 else 0,
    )

    return M, y, group_ids
