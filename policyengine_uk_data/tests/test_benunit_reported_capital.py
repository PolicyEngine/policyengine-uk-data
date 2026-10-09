"""`benunit_reported_capital` from the FRS benefit-unit capital measure
(TOTCAPB4, falling back to TOTCAPB3). policyengine-uk's Universal Credit and
Pension Credit reported capital default to it.

Invariants:
1. A finite, non-negative TOTCAPB4 is carried over unchanged; where TOTCAPB4
   is absent or invalid, a valid TOTCAPB3 is used.
2. A missing, non-numeric or negative value in both gives -1
   (policyengine-uk's "none recorded" sentinel), so the household proxy
   applies.
3. Without either column every benefit unit gets -1.
4. The output is always -1 or a non-negative number, one per benefit unit.
"""

import numpy as np
import pandas as pd

from policyengine_uk_data.datasets.frs import derive_benunit_reported_capital


def test_values_carry_over_and_invalid_values_fall_back():
    benunit = pd.DataFrame(
        {"totcapb3": [0.0, 300.0, 2_900.0, 1_250_000.0, np.nan, -5.0, "x"]}
    )
    result = derive_benunit_reported_capital(benunit)
    assert result.tolist() == [0.0, 300.0, 2_900.0, 1_250_000.0, -1.0, -1.0, -1.0]


def test_totcapb4_takes_precedence_with_totcapb3_fallback():
    benunit = pd.DataFrame(
        {
            "totcapb3": [100.0, 200.0, 300.0, np.nan, -1.0],
            "totcapb4": [150.0, np.nan, -5.0, 400.0, np.nan],
        }
    )
    result = derive_benunit_reported_capital(benunit)
    assert result.tolist() == [150.0, 200.0, 300.0, 400.0, -1.0]


def test_nullable_inputs_keep_the_totcapb3_fallback():
    """Nullable (pd.NA) columns must not wipe a valid TOTCAPB3 where TOTCAPB4
    is missing."""
    benunit = pd.DataFrame(
        {
            "totcapb3": pd.array([0.0, 250.0, pd.NA, -3.0], dtype="Float64"),
            "totcapb4": pd.array([pd.NA, pd.NA, 400.0, pd.NA], dtype="Float64"),
        }
    )
    result = derive_benunit_reported_capital(benunit)
    assert result.tolist() == [0.0, 250.0, 400.0, -1.0]


def test_missing_column_gives_sentinel():
    benunit = pd.DataFrame({"benunit_id": [101, 102, 201]})
    assert derive_benunit_reported_capital(benunit).tolist() == [-1.0] * 3


def test_output_is_sentinel_or_non_negative_for_random_inputs():
    rng = np.random.default_rng(1_792)
    for _ in range(50):
        n = int(rng.integers(1, 200))
        values = rng.normal(5_000, 20_000, n)
        values[rng.random(n) < 0.1] = np.nan
        result = derive_benunit_reported_capital(pd.DataFrame({"totcapb4": values}))
        assert len(result) == n
        assert np.all((result == -1) | (result >= 0))
        keep = np.isfinite(values) & (values >= 0)
        np.testing.assert_array_equal(result[keep], values[keep])
        assert np.all(result[~keep] == -1)


def test_uprating_rows_match_the_model():
    """The build uprates the column with these rows (calibration materialises
    the calibration year from them), and policyengine-uk projects the saved
    dataset with the variable's own uprating index at runtime. The rows must
    equal what ``create_policyengine_uprating_factors_table`` derives from the
    locked policyengine-uk, or a unit near a capital threshold (Pension
    Credit's 10,000 pound deemed-income disregard, Universal Credit's 6,000
    pound tariff threshold and 16,000 pound limit) can be assessed differently
    in calibration and at runtime."""
    from policyengine_uk.system import system

    from policyengine_uk_data.storage import STORAGE_FOLDER
    from policyengine_uk_data.utils.uprating import END_YEAR, START_YEAR

    variable = system.variables["benunit_reported_capital"]
    index = system.parameters.get_child(variable.uprating)
    years = range(START_YEAR, END_YEAR + 1)
    expected = {y: round(index(y) / index(START_YEAR), 3) for y in years}
    factors = pd.read_csv(STORAGE_FOLDER / "uprating_factors.csv").set_index("Variable")
    growth = pd.read_csv(STORAGE_FOLDER / "uprating_growth_factors.csv").set_index(
        "Variable"
    )
    for y in years:
        assert factors.loc["benunit_reported_capital", str(y)] == expected[y]
        expected_growth = (
            0 if y == START_YEAR else round(expected[y] / expected[y - 1] - 1, 3)
        )
        assert growth.loc["benunit_reported_capital", str(y)] == expected_growth


def test_programme_capital_defaults_to_the_recorded_input():
    """policyengine-uk derives both programmes' reported capital from the one
    input this build fills, so the dataset stores only the observed figure and
    the uprating tables carry no rows for the programme variables."""
    from policyengine_uk.system import system

    from policyengine_uk_data.storage import STORAGE_FOLDER

    assert not system.variables["benunit_reported_capital"].formulas
    factors = pd.read_csv(STORAGE_FOLDER / "uprating_factors.csv")
    for name in ("pension_credit_reported_capital", "uc_reported_capital"):
        assert system.variables[name].formulas
        assert name not in set(factors["Variable"])


def test_spi_copy_records_no_capital():
    """SPI-synthetic copies carry SPI-imputed incomes, so the FRS donor's
    capital is cleared to -1 (household proxy) on them."""
    from types import SimpleNamespace

    from policyengine_uk_data.datasets.imputations.income import (
        clear_frs_reported_capital,
    )

    copy = SimpleNamespace(
        benunit=pd.DataFrame(
            {
                "benunit_reported_capital": [0.0, 300.0, -1.0],
                "uc_reported_capital": [5.0, -1.0, 7.0],
            }
        )
    )
    cleared = clear_frs_reported_capital(copy).benunit
    assert cleared["benunit_reported_capital"].tolist() == [-1.0, -1.0, -1.0]
    assert cleared["uc_reported_capital"].tolist() == [-1.0, -1.0, -1.0]
    without = SimpleNamespace(benunit=pd.DataFrame({"benunit_id": [1, 2]}))
    assert list(clear_frs_reported_capital(without).benunit.columns) == ["benunit_id"]
