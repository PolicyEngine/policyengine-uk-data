"""`pension_credit_reported_capital` from the FRS benefit-unit capital
measure (TOTCAPB3).

Invariants:
1. A finite, non-negative TOTCAPB3 is carried over unchanged.
2. A missing, non-numeric or negative value gives -1 (policyengine-uk's
   "none recorded" sentinel), so the household proxy applies.
3. Without a TOTCAPB3 column every benefit unit gets -1.
4. The output is always -1 or a non-negative number, one per benefit unit.
"""

import numpy as np
import pandas as pd

from policyengine_uk_data.datasets.frs import derive_pension_credit_reported_capital


def test_values_carry_over_and_invalid_values_fall_back():
    benunit = pd.DataFrame(
        {"totcapb3": [0.0, 300.0, 2_900.0, 1_250_000.0, np.nan, -5.0, "x"]}
    )
    result = derive_pension_credit_reported_capital(benunit)
    assert result.tolist() == [0.0, 300.0, 2_900.0, 1_250_000.0, -1.0, -1.0, -1.0]


def test_missing_column_gives_sentinel():
    benunit = pd.DataFrame({"benunit_id": [101, 102, 201]})
    assert derive_pension_credit_reported_capital(benunit).tolist() == [-1.0] * 3


def test_output_is_sentinel_or_non_negative_for_random_inputs():
    rng = np.random.default_rng(1_792)
    for _ in range(50):
        n = int(rng.integers(1, 200))
        values = rng.normal(5_000, 20_000, n)
        values[rng.random(n) < 0.1] = np.nan
        result = derive_pension_credit_reported_capital(
            pd.DataFrame({"totcapb3": values})
        )
        assert len(result) == n
        assert np.all((result == -1) | (result >= 0))
        keep = np.isfinite(values) & (values >= 0)
        np.testing.assert_array_equal(result[keep], values[keep])
        assert np.all(result[~keep] == -1)


def test_uprating_rows_match_savings():
    """The column grows with ``savings`` (policyengine-uk uprates it with the
    same per-capita GDP index), so the build's uprating keeps them in step."""
    from policyengine_uk_data.storage import STORAGE_FOLDER

    for name in ["uprating_factors.csv", "uprating_growth_factors.csv"]:
        table = pd.read_csv(STORAGE_FOLDER / name).set_index("Variable")
        pd.testing.assert_series_equal(
            table.loc["pension_credit_reported_capital"],
            table.loc["savings"],
            check_names=False,
        )


def test_spi_copy_records_no_capital():
    """SPI-synthetic copies carry SPI-imputed incomes, so the FRS donor's
    capital is cleared to -1 (household proxy) on them."""
    from types import SimpleNamespace

    from policyengine_uk_data.datasets.imputations.income import (
        clear_frs_reported_capital,
    )

    copy = SimpleNamespace(
        benunit=pd.DataFrame({"pension_credit_reported_capital": [0.0, 300.0, -1.0]})
    )
    assert clear_frs_reported_capital(copy).benunit[
        "pension_credit_reported_capital"
    ].tolist() == [-1.0, -1.0, -1.0]
    without = SimpleNamespace(benunit=pd.DataFrame({"benunit_id": [1, 2]}))
    assert "pension_credit_reported_capital" not in (
        clear_frs_reported_capital(without).benunit.columns
    )
