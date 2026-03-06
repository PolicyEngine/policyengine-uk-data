"""
Validates that imputed electricity and gas spending/kWh on the FRS match
NEED 2023 admin data across four dimensions: income band, tenure, accommodation
type, and region.

Runs impute_consumption() on the base FRS (post-wealth imputation) so the
test reflects what the QRF + raking calibration actually produces for real
FRS households at 2023 price levels.
"""

import numpy as np
import pytest
from policyengine_uk import Microsimulation
from policyengine_uk.data import UKSingleYearDataset

from policyengine_uk_data.datasets.imputations.consumption import (
    ACCOMM_TO_NEED,
    NEED_ACCOMM_ELEC,
    NEED_ACCOMM_GAS,
    NEED_INCOME_BANDS,
    NEED_REGION_ELEC,
    NEED_REGION_GAS,
    NEED_TENURE_ELEC,
    NEED_TENURE_GAS,
    OFGEM_Q4_2023_ELEC_RATE,
    OFGEM_Q4_2023_GAS_RATE,
    TENURE_TO_NEED,
    impute_consumption,
)
from policyengine_uk_data.datasets.imputations.wealth import impute_wealth
from policyengine_uk_data.storage import STORAGE_FOLDER

BAND_TOL = 0.11  # 11% per cell (raking tension between dimensions can push ~10%)
HIGH_INC_TOL = 0.15  # 15% for £100k+ bands (thin FRS sample, raking tension)


@pytest.fixture(scope="module")
def imputed():
    """Base FRS with wealth then consumption imputed, at 2023 price levels."""
    try:
        ds = UKSingleYearDataset(STORAGE_FOLDER / "frs_2023_24.h5")
    except FileNotFoundError:
        pytest.skip("frs_2023_24.h5 not available")
    ds = impute_wealth(ds)
    return impute_consumption(ds)


@pytest.fixture(scope="module")
def arrays(imputed):
    sim = Microsimulation(dataset=imputed)
    return dict(
        income=sim.calculate(
            "hbai_household_net_income", map_to="household", period=2023
        ).values,
        tenure=sim.calculate("tenure_type", map_to="household", period=2023).values,
        accomm=sim.calculate(
            "accommodation_type", map_to="household", period=2023
        ).values,
        region=sim.calculate("region", map_to="household", period=2023).values,
        weights=sim.calculate(
            "household_weight", map_to="household", period=2023
        ).values,
        elec=imputed.household["electricity_consumption"].values,
        gas=imputed.household["gas_consumption"].values,
    )


def _wmean(values, weights):
    return float((values * weights).sum() / weights.sum())


def _check(label, rows):
    _print_table(label, rows)
    for band, imputed, target, pct_err in rows:
        assert pct_err < BAND_TOL, (
            f"{label} [{band}]: imputed {imputed:.0f} vs NEED {target:.0f} ({pct_err:.1%})"
        )


def test_electricity_by_income(arrays):
    elec, income, w = arrays["elec"], arrays["income"], arrays["weights"]
    rows = []
    for lo, hi, band, _, elec_kwh in NEED_INCOME_BANDS:
        target = elec_kwh * OFGEM_Q4_2023_ELEC_RATE
        mask = (income >= lo) & (income < hi)
        if mask.sum() == 0:
            continue
        imp = _wmean(elec[mask], w[mask])
        tol = HIGH_INC_TOL if lo >= 100_000 else BAND_TOL
        rows.append((band, imp, target, abs(imp - target) / target, tol))
    _print_table(
        "Electricity £/yr by income band",
        [(b, i, t, e) for b, i, t, e, _ in rows],
    )
    for band, imp, target, pct_err, tol in rows:
        assert pct_err < tol, (
            f"Electricity by income [{band}]: {imp:.0f} vs {target:.0f} ({pct_err:.1%})"
        )


def test_gas_by_income(arrays):
    gas, income, w = arrays["gas"], arrays["income"], arrays["weights"]
    rows = []
    for lo, hi, band, gas_kwh, _ in NEED_INCOME_BANDS:
        target = gas_kwh * OFGEM_Q4_2023_GAS_RATE
        mask = (income >= lo) & (income < hi)
        if mask.sum() == 0:
            continue
        imp = _wmean(gas[mask], w[mask])
        tol = HIGH_INC_TOL if lo >= 100_000 else BAND_TOL
        rows.append((band, imp, target, abs(imp - target) / target, tol))
    _print_table("Gas £/yr by income band", [(b, i, t, e) for b, i, t, e, _ in rows])
    for band, imp, target, pct_err, tol in rows:
        assert pct_err < tol, (
            f"Gas by income [{band}]: {imp:.0f} vs {target:.0f} ({pct_err:.1%})"
        )


def test_electricity_by_tenure(arrays):
    elec, tenure, w = arrays["elec"], arrays["tenure"], arrays["weights"]
    rows = []
    for frs_val, need_key in TENURE_TO_NEED.items():
        target = NEED_TENURE_ELEC[need_key] * OFGEM_Q4_2023_ELEC_RATE
        mask = tenure == frs_val
        if mask.sum() == 0:
            continue
        imp = _wmean(elec[mask], w[mask])
        rows.append((frs_val, imp, target, abs(imp - target) / target))
    _check("Electricity £/yr by tenure", rows)


def test_gas_by_tenure(arrays):
    gas, tenure, w = arrays["gas"], arrays["tenure"], arrays["weights"]
    rows = []
    for frs_val, need_key in TENURE_TO_NEED.items():
        target = NEED_TENURE_GAS[need_key] * OFGEM_Q4_2023_GAS_RATE
        mask = tenure == frs_val
        if mask.sum() == 0:
            continue
        imp = _wmean(gas[mask], w[mask])
        rows.append((frs_val, imp, target, abs(imp - target) / target))
    _check("Gas £/yr by tenure", rows)


def test_electricity_by_accommodation(arrays):
    elec, accomm, w = arrays["elec"], arrays["accomm"], arrays["weights"]
    rows = []
    for frs_val, need_key in ACCOMM_TO_NEED.items():
        target = NEED_ACCOMM_ELEC[need_key] * OFGEM_Q4_2023_ELEC_RATE
        mask = accomm == frs_val
        if mask.sum() == 0:
            continue
        imp = _wmean(elec[mask], w[mask])
        rows.append((frs_val, imp, target, abs(imp - target) / target))
    _check("Electricity £/yr by accommodation type (excl. OTHER)", rows)


def test_gas_by_accommodation(arrays):
    gas, accomm, w = arrays["gas"], arrays["accomm"], arrays["weights"]
    rows = []
    for frs_val, need_key in ACCOMM_TO_NEED.items():
        target = NEED_ACCOMM_GAS[need_key] * OFGEM_Q4_2023_GAS_RATE
        mask = accomm == frs_val
        if mask.sum() == 0:
            continue
        imp = _wmean(gas[mask], w[mask])
        rows.append((frs_val, imp, target, abs(imp - target) / target))
    _check("Gas £/yr by accommodation type (excl. OTHER)", rows)


def test_electricity_by_region(arrays):
    elec, region, w = arrays["elec"], arrays["region"], arrays["weights"]
    rows = []
    for reg, target_kwh in NEED_REGION_ELEC.items():
        target = target_kwh * OFGEM_Q4_2023_ELEC_RATE
        mask = region == reg
        if mask.sum() == 0:
            continue
        imp = _wmean(elec[mask], w[mask])
        rows.append((reg, imp, target, abs(imp - target) / target))
    _check("Electricity £/yr by region", rows)


def test_gas_by_region(arrays):
    gas, region, w = arrays["gas"], arrays["region"], arrays["weights"]
    rows = []
    for reg, target_kwh in NEED_REGION_GAS.items():
        target = target_kwh * OFGEM_Q4_2023_GAS_RATE
        mask = region == reg
        if mask.sum() == 0:
            continue
        imp = _wmean(gas[mask], w[mask])
        rows.append((reg, imp, target, abs(imp - target) / target))
    _check("Gas £/yr by region", rows)


def test_non_negative_energy(imputed):
    """All households should have non-negative electricity and gas spend."""
    elec = imputed.household["electricity_consumption"].values
    gas = imputed.household["gas_consumption"].values
    assert (elec >= 0).all(), f"{(elec < 0).sum()} households have negative electricity"
    assert (gas >= 0).all(), f"{(gas < 0).sum()} households have negative gas"


def test_national_mean(arrays):
    """Weighted national mean should be within 15% of NEED 2023 overall mean."""
    w = arrays["weights"]
    # NEED 2023 overall mean: unweighted average across income bands as proxy
    need_elec_national = (
        sum(e for *_, e in NEED_INCOME_BANDS) / len(NEED_INCOME_BANDS)
    ) * OFGEM_Q4_2023_ELEC_RATE
    need_gas_national = (
        sum(g for *_, g, _ in NEED_INCOME_BANDS) / len(NEED_INCOME_BANDS)
    ) * OFGEM_Q4_2023_GAS_RATE

    imp_elec = _wmean(arrays["elec"], w)
    imp_gas = _wmean(arrays["gas"], w)

    elec_err = abs(imp_elec - need_elec_national) / need_elec_national
    gas_err = abs(imp_gas - need_gas_national) / need_gas_national
    print(
        f"\nNational mean electricity: imputed £{imp_elec:.0f} vs NEED £{need_elec_national:.0f} ({elec_err:.1%})"
    )
    print(
        f"National mean gas: imputed £{imp_gas:.0f} vs NEED £{need_gas_national:.0f} ({gas_err:.1%})"
    )
    assert elec_err < 0.15, f"National electricity mean off by {elec_err:.1%}"
    assert gas_err < 0.15, f"National gas mean off by {gas_err:.1%}"


def test_energy_sum_approx_domestic(imputed):
    """Electricity + gas should roughly equal the legacy domestic_energy_consumption."""
    elec = imputed.household["electricity_consumption"].values
    gas = imputed.household["gas_consumption"].values
    domestic = imputed.household["domestic_energy_consumption"].values

    # Compare medians rather than means to be robust to outliers
    combined = np.median(elec + gas)
    legacy = np.median(domestic)
    if legacy > 0:
        ratio = combined / legacy
        print(
            f"\nMedian(elec+gas)={combined:.0f}, median(domestic_energy)={legacy:.0f}, ratio={ratio:.2f}"
        )
        assert 0.5 < ratio < 2.0, (
            f"elec+gas median (£{combined:.0f}) diverges from domestic_energy "
            f"median (£{legacy:.0f}) by ratio {ratio:.2f}"
        )


def _print_table(title, rows):
    print(f"\n{'─' * 68}")
    print(f"  {title}")
    print(f"{'─' * 68}")
    print(f"  {'Group':<26} {'Imputed':>10} {'NEED 2023':>10} {'Error':>8}")
    print(f"  {'─' * 26} {'─' * 10} {'─' * 10} {'─' * 8}")
    for group, imp, target, pct_err in rows:
        flag = " !" if pct_err >= BAND_TOL else ""
        print(f"  {str(group):<26} {imp:>10.0f} {target:>10.0f} {pct_err:>7.1%}{flag}")
    print(f"{'─' * 68}")
