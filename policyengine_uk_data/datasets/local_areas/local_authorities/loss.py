"""Local authority calibration target matrix.

Constructs the (matrix, y, country_mask) triple for calibrating
household weights across 360 local authorities. Target data is
loaded from source modules in the targets system.

Sources:
- Age: ONS mid-year population estimates
- Income: HMRC SPI table 3.15
- UC: DWP Stat-Xplore
- ONS income: ONS small area income estimates
- Tenure: English Housing Survey
- Private rent: VOA/ONS private rental market statistics
- Council tax bands A-H: VOA Council Tax Stock of Properties (per LA)
- Council tax £ paid (net of CTR): MHCLG taxbase × Band D (England),
  Welsh Government Council Tax Income (Wales)

Missing-source policy: local target cells stay NaN when no direct LA
source is available. The local-area calibrator masks those cells out of
the local loss. National targets are supplied by a separate national
target matrix, so this module should not fabricate local targets by
allocating national totals across missing-source LAs.
"""

from policyengine_uk import Microsimulation
import pandas as pd
import numpy as np

from policyengine_uk.data import UKSingleYearDataset
from policyengine_uk_data.storage import STORAGE_FOLDER
from policyengine_uk_data.targets.sources.local_age import (
    get_la_age_targets,
    get_uk_total_population,
)
from policyengine_uk_data.targets.sources.local_income import (
    get_la_income_targets,
    get_national_income_projections,
    INCOME_VARIABLES,
)
from policyengine_uk_data.targets.sources.local_uc import (
    get_la_uc_targets,
)
from policyengine_uk_data.targets.sources.local_la_extras import (
    get_ons_income_uprating_factors,
    load_ons_la_income,
    load_household_counts,
    load_tenure_data,
    load_private_rents,
)


def create_local_authority_target_matrix(
    dataset: UKSingleYearDataset,
    time_period: int = None,
    reform=None,
):
    if time_period is None:
        time_period = dataset.time_period

    la_codes = pd.read_csv(STORAGE_FOLDER / "local_authorities_2021.csv")

    sim = Microsimulation(dataset=dataset, reform=reform)
    sim.default_calculation_period = time_period

    matrix = pd.DataFrame()
    y = pd.DataFrame()

    # ── Income targets ─────────────────────────────────────────────
    incomes = get_la_income_targets()
    national_incomes = get_national_income_projections(int(time_period))

    for income_variable in INCOME_VARIABLES:
        income_values = sim.calculate(income_variable).values
        in_spi_frame = sim.calculate("income_tax").values > 0
        matrix[f"hmrc/{income_variable}/amount"] = sim.map_result(
            income_values * in_spi_frame, "person", "household"
        )
        local_targets = incomes[f"{income_variable}_amount"].values
        local_target_sum = local_targets.sum()
        national_target = national_incomes[
            (national_incomes.total_income_lower_bound == 12_570)
            & (national_incomes.total_income_upper_bound == np.inf)
        ][income_variable + "_amount"].iloc[0]
        adjustment = national_target / local_target_sum
        y[f"hmrc/{income_variable}/amount"] = local_targets * adjustment

        matrix[f"hmrc/{income_variable}/count"] = sim.map_result(
            (income_values != 0) * in_spi_frame, "person", "household"
        )
        y[f"hmrc/{income_variable}/count"] = (
            incomes[f"{income_variable}_count"].values * adjustment
        )

    # ── Age targets ────────────────────────────────────────────────
    age_targets = get_la_age_targets()
    uk_total_population = get_uk_total_population(int(time_period))

    age = sim.calculate("age").values
    targets_total_pop = 0
    age_cols = [c for c in age_targets.columns if c.startswith("age/")]
    for col in age_cols:
        lower, upper = col.removeprefix("age/").split("_")
        lower, upper = int(lower), int(upper)
        in_band = (age >= lower) & (age < upper)
        matrix[col] = sim.map_result(in_band, "person", "household")
        y[col] = age_targets[col].values
        targets_total_pop += age_targets[col].values.sum()

    for col in age_cols:
        y[col] *= uk_total_population / targets_total_pop * 0.9

    # ── UC targets ─────────────────────────────────────────────────
    y["uc_households"] = get_la_uc_targets().values
    matrix["uc_households"] = sim.map_result(
        (sim.calculate("universal_credit").values > 0).astype(int),
        "benunit",
        "household",
    )

    # ── ONS income targets ─────────────────────────────────────────
    ons_income = load_ons_la_income()
    households_by_la = load_household_counts()

    ons_merged = la_codes.merge(
        ons_income, left_on="code", right_on="la_code", how="left"
    ).merge(
        households_by_la,
        left_on="code",
        right_on="la_code",
        how="left",
        suffixes=("", "_hh"),
    )

    hbai_net_income = sim.calculate("equiv_hbai_household_net_income").values
    hbai_net_income_ahc = sim.calculate("equiv_hbai_household_net_income_ahc").values
    housing_costs = hbai_net_income - hbai_net_income_ahc
    income_bhc_uprating_factor, housing_costs_uprating_factor = (
        get_ons_income_uprating_factors(int(time_period))
    )

    matrix["ons/equiv_net_income_bhc"] = hbai_net_income
    matrix["ons/equiv_net_income_ahc"] = hbai_net_income_ahc
    matrix["ons/equiv_housing_costs"] = housing_costs

    ons_merged["equiv_net_income_bhc_target"] = (
        ons_merged["net_income_bhc"]
        * ons_merged["households"]
        * income_bhc_uprating_factor
    )
    ons_merged["equiv_housing_costs_target"] = (
        ons_merged["net_income_bhc"] * ons_merged["households"]
        - ons_merged["net_income_ahc"] * ons_merged["households"]
    ) * housing_costs_uprating_factor
    ons_merged["equiv_net_income_ahc_target"] = (
        ons_merged["equiv_net_income_bhc_target"]
        - ons_merged["equiv_housing_costs_target"]
    )

    has_ons_data = (
        ons_merged["net_income_bhc"].notna() & ons_merged["households"].notna()
    ).values
    y["ons/equiv_net_income_bhc"] = np.where(
        has_ons_data,
        ons_merged["equiv_net_income_bhc_target"].values,
        np.nan,
    )
    y["ons/equiv_net_income_ahc"] = np.where(
        has_ons_data,
        ons_merged["equiv_net_income_ahc_target"].values,
        np.nan,
    )
    y["ons/equiv_housing_costs"] = np.where(
        has_ons_data,
        ons_merged["equiv_housing_costs_target"].values,
        np.nan,
    )

    # ── Tenure targets ─────────────────────────────────────────────
    tenure_data = load_tenure_data()

    tenure_merged = la_codes.merge(
        tenure_data, left_on="code", right_on="la_code", how="left"
    ).merge(
        households_by_la,
        left_on="code",
        right_on="la_code",
        how="left",
        suffixes=("", "_hh"),
    )

    tenure_type = sim.calculate("tenure_type").values
    matrix["tenure/owned_outright"] = (tenure_type == "OWNED_OUTRIGHT").astype(float)
    matrix["tenure/owned_mortgage"] = (tenure_type == "OWNED_WITH_MORTGAGE").astype(
        float
    )
    matrix["tenure/private_rent"] = (tenure_type == "RENT_PRIVATELY").astype(float)
    matrix["tenure/social_rent"] = (
        (tenure_type == "RENT_FROM_COUNCIL") | (tenure_type == "RENT_FROM_HA")
    ).astype(float)

    has_tenure = (
        tenure_merged["owned_outright_pct"].notna()
        & tenure_merged["households"].notna()
    ).values

    for tenure_key, pct_col in [
        ("owned_outright", "owned_outright_pct"),
        ("owned_mortgage", "owned_mortgage_pct"),
        ("private_rent", "private_rent_pct"),
        ("social_rent", "social_rent_pct"),
    ]:
        targets = tenure_merged[pct_col] / 100 * tenure_merged["households"]
        y[f"tenure/{tenure_key}"] = np.where(
            has_tenure,
            targets.values,
            np.nan,
        )

    # ── Private rent amounts ───────────────────────────────────────
    rent_data = load_private_rents()

    tenure_merged = tenure_merged.merge(
        rent_data, left_on="code", right_on="area_code", how="left"
    )

    is_private_renter = (tenure_type == "RENT_PRIVATELY").astype(float)
    benunit_rent = sim.calculate("benunit_rent").values
    household_rent = sim.map_result(benunit_rent, "benunit", "household")
    private_rent_amount = household_rent * is_private_renter

    matrix["rent/private_rent"] = private_rent_amount

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
    y["rent/private_rent"] = np.where(
        has_rent,
        tenure_merged["private_rent_target"].values,
        np.nan,
    )

    # ── Council tax band counts (LA targets) ───────────────────────
    # Derived/proxy targets: per-LA VOA dwellings in each band A-H.
    # Lineage drift vs the matrix-side household council_tax_band:
    # VOA counts dwellings (incl. exempt / empty / second homes);
    # matrix counts households. See la_council_tax.py for full
    # caveat. Missing cells stay NaN and are masked out by the
    # calibrator; this keeps the target direct instead of fabricating
    # national-share fallbacks for Scotland or Northern Ireland. Band I
    # is Wales-only and rarely populated, so it is intentionally
    # excluded.
    ct_path = STORAGE_FOLDER / "la_council_tax.csv"
    if ct_path.exists():
        ct_data = pd.read_csv(ct_path)
        ct_columns = ["code"] + [f"count_band_{b}" for b in "ABCDEFGH"]
        if "total_council_tax_net" in ct_data.columns:
            ct_columns.append("total_council_tax_net")
        ct_merged = la_codes.merge(ct_data[ct_columns], on="code", how="left")
        ct_band = sim.calculate("council_tax_band").values
        for band in "ABCDEFGH":
            col = f"voa/council_tax/{band}"
            matrix[col] = (ct_band == band).astype(float)
            csv_col = f"count_band_{band}"
            has_count = ct_merged[csv_col].notna().values
            direct = ct_merged[csv_col].values
            y[col] = np.where(
                has_count,
                direct,
                np.nan,
            )

        # ── Council tax £ paid, net of CTR (LA targets) ────────────
        # Derived/proxy target: y = MHCLG taxbase × Band D (E) or WG
        # Council Tax Income (W). Matrix col is FRS-reported
        # council_tax_less_benefit (gross − reported CTB). Same
        # intent (household council tax paid net of CTR), different
        # construction paths — see la_council_tax.py for the lineage
        # caveat flagged in review by @MaxGhenis. Both sides are net
        # of CTR, per Max's 28 Apr standup decision on FRS alignment.
        # Missing cells remain NaN and are masked out by the calibrator.
        if "total_council_tax_net" in ct_merged.columns:
            matrix["housing/council_tax_net"] = sim.calculate(
                "council_tax_less_benefit"
            ).values
            has_ct_net = ct_merged["total_council_tax_net"].notna().values
            direct_net = ct_merged["total_council_tax_net"].values
            y["housing/council_tax_net"] = np.where(
                has_ct_net,
                direct_net,
                np.nan,
            )

    # ── Country mask ───────────────────────────────────────────────
    country_mask = create_country_mask(
        household_countries=sim.calculate("country").values,
        codes=la_codes.code,
    )

    return matrix, y, country_mask


def create_country_mask(
    household_countries: np.ndarray, codes: pd.Series
) -> np.ndarray:
    """Country mask: R[i,j] = 1 iff household j is in same country as area i."""
    area_countries = codes.apply(lambda code: code[0]).map(
        {
            "E": "ENGLAND",
            "W": "WALES",
            "S": "SCOTLAND",
            "N": "NORTHERN_IRELAND",
        }
    )
    r = np.zeros((len(codes), len(household_countries)))
    for i in range(len(codes)):
        r[i] = household_countries == area_countries.iloc[i]
    return r
