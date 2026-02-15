"""Constituency-level calibration target matrix.

Constructs the (matrix, y, country_mask) triple for calibrating
household weights across 650 parliamentary constituencies. Target
data is loaded from source modules in the targets system.

Sources:
- Age: ONS mid-year population estimates
- Income: HMRC SPI table 3.15
- UC: DWP Stat-Xplore
"""

from policyengine_uk import Microsimulation
import pandas as pd
import numpy as np

from policyengine_uk.data import UKSingleYearDataset
from policyengine_uk_data.storage import STORAGE_FOLDER
from policyengine_uk_data.datasets.local_areas.constituencies.boundary_changes.mapping_matrix import (
    mapping_matrix,
)
from policyengine_uk_data.targets.sources.local_age import (
    get_constituency_age_targets,
    get_uk_total_population,
)
from policyengine_uk_data.targets.sources.local_income import (
    get_constituency_income_targets,
    get_national_income_projections,
    INCOME_VARIABLES,
)
from policyengine_uk_data.targets.sources.local_uc import (
    get_constituency_uc_targets,
)


def create_constituency_target_matrix(
    dataset: UKSingleYearDataset,
    time_period: int = None,
    reform=None,
):
    if time_period is None:
        time_period = dataset.time_period

    sim = Microsimulation(dataset=dataset, reform=reform)
    sim.default_calculation_period = dataset.time_period

    matrix = pd.DataFrame()
    y = pd.DataFrame()

    # ── Income targets ─────────────────────────────────────────────
    incomes = get_constituency_income_targets()
    national_incomes = get_national_income_projections(
        int(dataset.time_period)
    )

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
    age_targets = get_constituency_age_targets()
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

    # National consistency adjustment
    for col in age_cols:
        y[col] *= uk_total_population / targets_total_pop * 0.9

    # ── UC targets ─────────────────────────────────────────────────
    y["uc_households"] = get_constituency_uc_targets().values
    matrix["uc_households"] = sim.map_result(
        (sim.calculate("universal_credit").values > 0).astype(int),
        "benunit",
        "household",
    )

    # ── Boundary mapping (2010 → 2024) ────────────────────────────
    const_2024 = pd.read_csv(STORAGE_FOLDER / "constituencies_2024.csv")

    y_columns = list(y.columns)
    y_values = mapping_matrix @ y.values
    y = pd.DataFrame(y_values, columns=y_columns)

    country_mask = create_country_mask(
        household_countries=sim.calculate("country").values,
        codes=const_2024.code,
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
