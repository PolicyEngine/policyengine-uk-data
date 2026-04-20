"""Regression test for issue #367: housing-cost pass-through.

Before this fix, ``impute_over_incomes`` multiplied ``rent``,
``mortgage_interest_repayment`` and ``mortgage_capital_repayment`` by
``new_income_total / original_income_total``. Because FRS under-reports
dividends while the SPI-trained QRF predicts realistic dividend values,
the ratio inflated housing costs ~2.5× in the built enhanced FRS,
pushing AHC poverty rates 10-18 pp above HBAI for non-pensioners.

These tests guard against the rescaling coming back in any form.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


class _FakeQRFModel:
    """Minimal stub with the interface `impute_over_incomes` expects."""

    def __init__(self, imputations, multiplier: float = 1.0):
        self._imputations = list(imputations)
        self._multiplier = multiplier

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        # Plausible non-negative predictions, scaled by a caller-controlled
        # multiplier so we can simulate huge dividend imputations on FRS
        # donor rows without needing a real SPI-trained QRF.
        n = len(X)
        rng = np.random.default_rng(0)
        return pd.DataFrame(
            {
                col: rng.exponential(1_000 * self._multiplier, size=n)
                for col in self._imputations
            }
        )


def _tiny_frs_dataset():
    """Load the committed tiny FRS dataset (1 000 households) if available."""
    from policyengine_uk.data import UKSingleYearDataset
    from policyengine_uk_data.storage import STORAGE_FOLDER

    path = STORAGE_FOLDER / "frs_2023_24_tiny.h5"
    if not path.exists():
        pytest.skip("Tiny FRS dataset not available")
    return UKSingleYearDataset(path)


def test_housing_costs_pass_through_unchanged():
    """Housing-cost columns must be byte-identical on exit.

    Even when imputed dividends dwarf the FRS baseline (multiplier=100),
    rent and mortgage values must not be rescaled.
    """
    from policyengine_uk_data.datasets.imputations.income import (
        IMPUTATIONS,
        impute_over_incomes,
    )

    ds = _tiny_frs_dataset()
    rent_in = ds.household["rent"].to_numpy().copy()
    mi_in = ds.household["mortgage_interest_repayment"].to_numpy().copy()
    mc_in = ds.household["mortgage_capital_repayment"].to_numpy().copy()

    # If the old rescaling logic were still here, rent/mortgage would come
    # out ~100× larger.
    model = _FakeQRFModel(IMPUTATIONS, multiplier=100.0)
    result = impute_over_incomes(ds, model, ["dividend_income"])

    np.testing.assert_array_equal(result.household["rent"].to_numpy(), rent_in)
    np.testing.assert_array_equal(
        result.household["mortgage_interest_repayment"].to_numpy(), mi_in
    )
    np.testing.assert_array_equal(
        result.household["mortgage_capital_repayment"].to_numpy(), mc_in
    )


def test_only_listed_outputs_are_overwritten():
    """`output_variables` may be touched; other income columns must not be."""
    from policyengine_uk_data.datasets.imputations.income import (
        IMPUTATIONS,
        impute_over_incomes,
    )

    ds = _tiny_frs_dataset()
    employment_in = ds.person["employment_income"].to_numpy().copy()
    dividend_in = ds.person["dividend_income"].to_numpy().copy()

    model = _FakeQRFModel(IMPUTATIONS, multiplier=1.0)
    result = impute_over_incomes(ds, model, ["dividend_income"])

    np.testing.assert_array_equal(
        result.person["employment_income"].to_numpy(), employment_in
    )
    # dividend_income was listed — prediction output should differ from the
    # near-zero FRS baseline for at least most rows.
    assert not np.array_equal(result.person["dividend_income"].to_numpy(), dividend_in)


def test_housing_costs_preserved_when_income_baseline_is_zero():
    """Covers the zero-baseline shape the old `_safe_rescale_factor` guarded."""
    from policyengine_uk_data.datasets.imputations.income import (
        IMPUTATIONS,
        INCOME_COMPONENTS,
        impute_over_incomes,
    )

    ds = _tiny_frs_dataset()
    for col in INCOME_COMPONENTS:
        if col in ds.person.columns:
            ds.person[col] = 0.0

    rent_in = ds.household["rent"].to_numpy().copy()
    model = _FakeQRFModel(IMPUTATIONS, multiplier=1.0)
    result = impute_over_incomes(ds, model, ["dividend_income"])

    out = result.household["rent"].to_numpy()
    assert np.all(np.isfinite(out))
    np.testing.assert_array_equal(out, rent_in)


def test_built_enhanced_frs_housing_costs_track_raw_frs():
    """Regression: after build, enhanced FRS per-renter rent should be close
    to raw FRS per-renter rent (modulo small uprating / calibration effects).

    Pre-fix, the built enhanced FRS had rent values 2.5× raw FRS. The tight
    tolerance here (30 %) will fail on any dataset rebuilt with the old
    rescaling logic.
    """
    from policyengine_uk.data import UKSingleYearDataset
    from policyengine_uk_data.storage import STORAGE_FOLDER

    raw_path = STORAGE_FOLDER / "frs_2023_24.h5"
    enh_path = STORAGE_FOLDER / "enhanced_frs_2023_24.h5"
    if not (raw_path.exists() and enh_path.exists()):
        pytest.skip("Full raw and enhanced FRS datasets not available")

    raw = UKSingleYearDataset(raw_path)
    enh = UKSingleYearDataset(enh_path)

    for col in ("rent", "mortgage_interest_repayment", "mortgage_capital_repayment"):
        r = raw.household[col].to_numpy()
        e = enh.household[col].to_numpy()
        r_med = float(np.median(r[r > 0])) if (r > 0).any() else 0.0
        e_med = float(np.median(e[e > 0])) if (e > 0).any() else 0.0
        assert r_med > 0, f"Raw FRS has no positive {col}"
        ratio = e_med / r_med
        assert 0.7 < ratio < 1.3, (
            f"Enhanced {col} median (£{e_med:,.0f}) diverges from raw "
            f"(£{r_med:,.0f}) by ratio {ratio:.2f}x; expected near 1.0. "
            "Housing-cost rescaling may have been reintroduced (see #367)."
        )
